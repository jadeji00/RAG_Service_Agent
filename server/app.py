"""FastAPI 入口：登录、对话 SSE、会话列表/历史、trace 与用量查询。

启动阶段才构造 ReactAgent（向量库/模型），导入 app 本身不触发知识库加载；
启动失败不直接崩溃，而是让 /health 返回可读错误，便于页面提示（P4）。

身份是演示级的：账号/令牌/会话归属都在 ``server.auth``，业务接口靠
``current_user`` 依赖拿登录态，会话按 ``thread_id`` 校验归属，避免换个
线程 ID 就读到别人的对话。
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.session_index import is_session_thread, list_checkpoint_threads
from observability.store import TraceStore
from server.auth import ANONYMOUS_USER, AuthStore, User
from server.chat_runner import ChatRunner
from server.rate_limit import SlidingWindowLimiter
from server.sse import encode_sse
from utils.logger_handler import logger

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    thread_id: str | None = Field(default=None, max_length=64)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


def _bearer(authorization: str | None) -> str | None:
    """只认 ``Authorization: Bearer <token>``；裸令牌不放行，避免口径含糊。"""
    value = str(authorization or "").strip()
    if not value:
        return None
    prefix = "bearer "
    if not value.lower().startswith(prefix):
        return None
    return value[len(prefix) :].strip() or None


def _adopt_legacy_threads(runner: ChatRunner | None, auth_store: AuthStore) -> int:
    """把登录功能上线前产生的会话归到 ``legacy_owner`` 名下。

    这些线程在归属表里没有行，不迁移的话开启鉴权后侧栏会变成空的。
    """
    if not auth_store.enabled or not auth_store.legacy_owner or runner is None:
        return 0
    db_path = getattr(runner.agent, "checkpoint_db_path", None)
    if not db_path or not os.path.exists(db_path):
        return 0
    threads = [item["thread_id"] for item in list_checkpoint_threads(db_path)]
    return auth_store.adopt_threads(threads, auth_store.legacy_owner)


def create_app(
    runner: ChatRunner | None = None,
    store: TraceStore | None = None,
    auth: AuthStore | None = None,
    limiter: SlidingWindowLimiter | None = None,
) -> FastAPI:
    trace_store = store or TraceStore()
    auth_store = auth if auth is not None else AuthStore()
    rate_limiter = limiter if limiter is not None else SlidingWindowLimiter()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if runner is not None:
            app.state.runner = runner
            app.state.startup_error = None
            _log_adopted(_adopt_legacy_threads(runner, auth_store))
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
        _log_adopted(_adopt_legacy_threads(app.state.runner, auth_store))
        yield

    app = FastAPI(
        title="智扫通机器人智能客服",
        version="0.4.0",
        description="扫地/扫拖机器人客服 Agent：登录、SSE 流式对话、trace、token 成本账本。",
        lifespan=lifespan,
    )
    app.state.store = trace_store
    app.state.auth = auth_store
    app.state.limiter = rate_limiter
    app.state.runner = runner
    app.state.startup_error = None

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def current_user(authorization: str | None = Header(default=None)) -> User:
        """所有业务接口的登录校验；鉴权关闭时返回占位身份。"""
        if not auth_store.enabled:
            return ANONYMOUS_USER
        token = _bearer(authorization)
        user = auth_store.user_by_token(token) if token else None
        if user is None:
            raise HTTPException(
                status_code=401,
                detail="未登录或登录已过期",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user

    def require_thread_owner(thread_id: str, user: User) -> None:
        """会话归属校验：别人的线程直接 403，没有归属行则放行（无内容可读）。"""
        if not auth_store.enabled:
            return
        owner = auth_store.thread_owner(thread_id)
        if owner is not None and owner != user.username:
            raise HTTPException(status_code=403, detail="这个会话属于其他账号")

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
            "auth": auth_store.enabled,
        }

    # ------------------------------------------------------------- 身份

    @app.get("/auth/bootstrap")
    def auth_bootstrap():
        """公开接口：告诉前端要不要登录、有哪些演示账号（不含任何密码字段）。"""
        return {
            "enabled": auth_store.enabled,
            "users": auth_store.demo_users() if auth_store.enabled else [],
            "hint": auth_store.login_hint if auth_store.enabled else "",
        }

    @app.post("/auth/login")
    def auth_login(payload: LoginRequest):
        if not auth_store.enabled:
            raise HTTPException(status_code=400, detail="登录未启用")
        result = auth_store.login(payload.username, payload.password)
        if result is None:
            logger.warning(f"[auth] 登录失败：{payload.username[:32]}")
            raise HTTPException(status_code=401, detail="账号或密码不正确")
        user, token = result
        logger.info(f"[auth] 登录成功：{user.username}")
        return {
            "token": token,
            "user": user.as_dict(),
            "expires_in_hours": auth_store.token_ttl_hours,
        }

    @app.post("/auth/logout")
    def auth_logout(
        user: User = Depends(current_user),
        authorization: str | None = Header(default=None),
    ):
        auth_store.logout(_bearer(authorization) or "")
        return {"ok": True, "username": user.username}

    @app.get("/auth/me")
    def auth_me(user: User = Depends(current_user)):
        return {"enabled": auth_store.enabled, "user": user.as_dict()}

    # ------------------------------------------------------------- 会话

    @app.get("/history/{thread_id}")
    def history(thread_id: str, user: User = Depends(current_user)):
        current = getattr(app.state, "runner", None)
        if current is None:
            raise HTTPException(status_code=503, detail="Agent 尚未初始化")
        require_thread_owner(thread_id, user)
        try:
            messages = current.agent.load_history(thread_id)
        except Exception as exc:
            logger.error(f"[server] 读取会话 {thread_id} 失败", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"thread_id": thread_id, "messages": messages}

    @app.get("/sessions")
    def sessions(
        limit: int = Query(default=20, ge=1, le=50),
        user: User = Depends(current_user),
    ):
        current = getattr(app.state, "runner", None)
        if current is None:
            raise HTTPException(status_code=503, detail="Agent 尚未初始化")
        thread_filter = (
            auth_store.owner_filter(user.username) if auth_store.enabled else None
        )
        try:
            items = current.agent.list_sessions(limit=limit, thread_filter=thread_filter)
        except Exception as exc:
            logger.error("[server] 读取会话列表失败", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"sessions": items}

    @app.delete("/sessions/{thread_id}")
    def delete_session(thread_id: str, user: User = Depends(current_user)):
        if not is_session_thread(thread_id):
            raise HTTPException(status_code=422, detail="无效的会话 ID")
        current = getattr(app.state, "runner", None)
        if current is None:
            raise HTTPException(status_code=503, detail="Agent 尚未初始化")
        require_thread_owner(thread_id, user)

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

        auth_store.release_thread(thread_id)
        logger.info(f"[server] 已删除会话 {thread_id}：{deleted}")
        return {"thread_id": thread_id, "deleted": deleted}

    # ------------------------------------------------------------- 对话

    @app.post("/chat/stream")
    def chat_stream(
        payload: ChatRequest,
        request: Request,
        user: User = Depends(current_user),
    ):
        current = getattr(app.state, "runner", None)
        # 不给 thread_id 时发一个新线程，不再共用 "default"：
        # 共用会让第一个用户把 default 占下来，其他人一律 403。
        thread_id = (payload.thread_id or "").strip() or uuid.uuid4().hex
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

        # 限流放在这里：空 query 和服务未就绪都不该消耗配额（它们不会调模型），
        # 而认领线程是有副作用的写操作，被限流的请求不该留下痕迹。
        # 鉴权关闭时所有人都是同一个占位身份，按用户名计数等于全局一个桶，
        # 所以退回按客户端 IP。
        limit_key = user.username if auth_store.enabled else (
            request.client.host if request.client else "anonymous"
        )
        decision = rate_limiter.hit(limit_key)
        if not decision.allowed:
            logger.warning(
                f"[server] 限流拦截：{limit_key} 超过 {decision.limit} 次/"
                f"{int(rate_limiter.window_seconds)} 秒"
            )
            raise HTTPException(
                status_code=429,
                detail=f"请求太频繁了，请 {decision.retry_after} 秒后再试",
                headers={"Retry-After": str(decision.retry_after)},
            )

        if auth_store.enabled:
            owner = auth_store.claim_thread(thread_id, user.username)
            if owner != user.username:
                logger.warning(
                    f"[server] 越权访问被拦截：{user.username} 想用 {owner} 的会话"
                )
                raise HTTPException(status_code=403, detail="这个会话属于其他账号")

        owner_user_id = user.business_user_id if auth_store.enabled else None

        def event_stream():
            yield from (
                encode_sse(event["event"], event["data"])
                for event in current.stream(
                    query, thread_id=thread_id, owner_user_id=owner_user_id
                )
            )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers=_sse_headers(),
        )

    # ------------------------------------------------------------- 观测

    @app.get("/traces")
    def traces(
        limit: int = Query(default=20, ge=1, le=100),
        user: User = Depends(current_user),
    ):
        return {"traces": trace_store.list_traces(limit=limit)}

    @app.get("/trace/{trace_id}")
    def trace_detail(trace_id: str, user: User = Depends(current_user)):
        trace = trace_store.get_trace(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="trace 不存在")
        return trace

    @app.get("/usage")
    def usage(
        days: int = Query(default=7, ge=1, le=90),
        user: User = Depends(current_user),
    ):
        return trace_store.usage_summary(days=days)

    return app


def _log_adopted(count: int) -> None:
    if count:
        logger.info(f"[auth] 已把 {count} 个旧会话归到 legacy_owner 名下")


def _sse_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


app = create_app()
