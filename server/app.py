"""FastAPI 入口：对话 SSE、会话列表/历史、trace 与用量查询。

启动阶段才构造 ReactAgent（向量库/模型），导入 app 本身不触发知识库加载；
启动失败不直接崩溃，而是让 /health 返回可读错误，便于页面提示（P4）。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.session_index import is_session_thread
from observability.store import TraceStore
from server.chat_runner import ChatRunner
from server.sse import encode_sse
from utils.logger_handler import logger

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    thread_id: str | None = Field(default=None, max_length=64)


def create_app(
    runner: ChatRunner | None = None,
    store: TraceStore | None = None,
) -> FastAPI:
    trace_store = store or TraceStore()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if runner is not None:
            app.state.runner = runner
            app.state.startup_error = None
            yield
            return

        try:
            # 延迟导入：确保导入 server.app 时不会拉起向量库和模型。
            from agent.react_agent import ReactAgent

            app.state.runner = ChatRunner(ReactAgent(), store=trace_store)
            app.state.startup_error = None
            logger.info("[server] Agent 启动完成")
        except Exception as exc:
            app.state.runner = None
            app.state.startup_error = str(exc)
            logger.error("[server] Agent 启动失败", exc_info=True)
        yield

    app = FastAPI(
        title="智扫通机器人智能客服",
        version="0.3.0",
        description="扫地/扫拖机器人客服 Agent：SSE 流式对话、trace、token 成本账本。",
        lifespan=lifespan,
    )
    app.state.store = trace_store
    app.state.runner = runner
    app.state.startup_error = None

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health():
        current = getattr(app.state, "runner", None)
        if current is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "error": getattr(app.state, "startup_error", None)
                    or "Agent 尚未初始化",
                },
            )
        return {
            "status": "ok",
            "model": current.model_name,
            "trace_db": str(trace_store.db_path),
        }

    @app.get("/history/{thread_id}")
    def history(thread_id: str):
        current = getattr(app.state, "runner", None)
        if current is None:
            raise HTTPException(status_code=503, detail="Agent 尚未初始化")
        try:
            messages = current.agent.load_history(thread_id)
        except Exception as exc:
            logger.error(f"[server] 读取会话 {thread_id} 失败", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"thread_id": thread_id, "messages": messages}

    @app.get("/sessions")
    def sessions(limit: int = Query(default=20, ge=1, le=50)):
        current = getattr(app.state, "runner", None)
        if current is None:
            raise HTTPException(status_code=503, detail="Agent 尚未初始化")
        try:
            items = current.agent.list_sessions(limit=limit)
        except Exception as exc:
            logger.error("[server] 读取会话列表失败", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"sessions": items}

    @app.delete("/sessions/{thread_id}")
    def delete_session(thread_id: str):
        if not is_session_thread(thread_id):
            raise HTTPException(status_code=422, detail="无效的会话 ID")
        current = getattr(app.state, "runner", None)
        if current is None:
            raise HTTPException(status_code=503, detail="Agent 尚未初始化")

        # 先删不可见的 trace，再删可见的 checkpoint：后者失败时会话仍在列表里，
        # 重试即可；反过来会出现「列表没了、usage 还在」的脏状态。
        try:
            traces = trace_store.delete_thread(thread_id)
        except Exception as exc:
            logger.error(f"[server] 清理会话 {thread_id} 的 trace 失败", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        try:
            checkpoints = current.agent.delete_session(thread_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.error(f"[server] 删除会话 {thread_id} 失败", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        deleted = {**checkpoints, **traces}
        if not deleted.get("checkpoints") and not deleted.get("traces"):
            raise HTTPException(status_code=404, detail="会话不存在")

        logger.info(f"[server] 已删除会话 {thread_id}：{deleted}")
        return {"thread_id": thread_id, "deleted": deleted}

    @app.post("/chat/stream")
    def chat_stream(payload: ChatRequest):
        current = getattr(app.state, "runner", None)
        thread_id = (payload.thread_id or "").strip() or "default"
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query 不能为空")

        if current is None:
            error = getattr(app.state, "startup_error", None) or "Agent 尚未初始化"

            def failed_stream():
                yield encode_sse(
                    "error",
                    {
                        "code": "not_ready",
                        "message": f"服务未就绪：{error}",
                    },
                )

            return StreamingResponse(
                failed_stream(),
                media_type="text/event-stream",
                headers=_sse_headers(),
            )

        def event_stream():
            yield from (
                encode_sse(event["event"], event["data"])
                for event in current.stream(query, thread_id=thread_id)
            )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers=_sse_headers(),
        )

    @app.get("/traces")
    def traces(limit: int = Query(default=20, ge=1, le=100)):
        return {"traces": trace_store.list_traces(limit=limit)}

    @app.get("/trace/{trace_id}")
    def trace_detail(trace_id: str):
        trace = trace_store.get_trace(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="trace 不存在")
        return trace

    @app.get("/usage")
    def usage(days: int = Query(default=7, ge=1, le=90)):
        return trace_store.usage_summary(days=days)

    return app


def _sse_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


app = create_app()
