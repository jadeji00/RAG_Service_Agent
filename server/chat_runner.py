"""把 Agent 的事件流包装成可观测的 SSE 事件流。

职责边界：
- Agent 只负责「路由、增量文本、工具状态」；
- 这里负责 trace/span、token 累加、成本估算和最终 done/error 事件。
这样 Streamlit、FastAPI、测试可以复用同一套底层事件。
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Iterator

from observability.pricing import estimate_cost, pricing_snapshot
from observability.store import TraceStore, utc_now_iso
from observability.usage import TokenAccount
from utils.config_handler import rag_conf
from utils.logger_handler import logger


class ChatRunner:
    def __init__(
        self,
        agent,
        store: TraceStore | None = None,
        model_name: str | None = None,
    ):
        self.agent = agent
        self.store = store or TraceStore()
        self.model_name = model_name or rag_conf.get("chat_model_name", "unknown")

    def stream(
        self,
        query: str,
        thread_id: str = "default",
        owner_user_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """产出 ``{"event": name, "data": {...}}``，由 SSE 层编码。

        ``owner_user_id`` 透传给 Agent 的报告链路做归属校验；为空表示没有
        登录态（评测/CLI），不拦。
        """
        trace_id = uuid.uuid4().hex
        started = time.perf_counter()
        started_at = utc_now_iso()
        self.store.start_trace(trace_id, thread_id, query, self.model_name)

        account = TokenAccount()
        answer_parts: list[str] = []
        tool_starts: dict[str, float] = {}
        tool_count = 0
        route = "unknown"
        result: dict | None = None
        status = "ok"
        error_detail: str | None = None
        first_token_ms: float | None = None
        last_delta_at: float | None = None

        try:
            for event in self.agent.iter_chat(
                query,
                thread_id=thread_id,
                token_account=account,
                owner_user_id=owner_user_id,
            ):
                event_type = event.get("type")

                if event_type == "route":
                    route = event.get("route", "unknown")
                    self.store.update_route(trace_id, route)
                    self.store.add_span(
                        trace_id,
                        "route",
                        "route",
                        duration_ms=round((time.perf_counter() - started) * 1000, 2),
                        detail={"route": route},
                        started_at=started_at,
                    )
                    yield {
                        "event": "meta",
                        "data": {
                            "trace_id": trace_id,
                            "thread_id": thread_id,
                            "route": route,
                            "model": self.model_name,
                        },
                    }
                    continue

                if event_type == "delta":
                    text = event.get("text", "")
                    if not text:
                        continue
                    now = time.perf_counter()
                    if first_token_ms is None:
                        first_token_ms = (now - started) * 1000
                    last_delta_at = now
                    answer_parts.append(text)
                    yield {"event": "delta", "data": {"text": text}}
                    continue

                if event_type == "tool":
                    name = event.get("name", "tool")
                    call_id = event.get("id") or name
                    if event.get("status") == "start":
                        tool_starts[call_id] = time.perf_counter()
                        tool_count += 1
                        yield {
                            "event": "tool",
                            "data": {
                                "name": name,
                                "status": "start",
                                "args": event.get("args", {}),
                                "id": call_id,
                            },
                        }
                        continue

                    started_tool = tool_starts.pop(call_id, None)
                    if started_tool is None:
                        tool_count += 1
                    duration_ms = event.get("duration_ms")
                    if duration_ms is None and started_tool is not None:
                        duration_ms = (time.perf_counter() - started_tool) * 1000
                    ok = bool(event.get("ok", True))
                    self.store.add_span(
                        trace_id,
                        f"tool:{name}",
                        "tool",
                        duration_ms=round(float(duration_ms), 2)
                        if duration_ms is not None
                        else None,
                        status="ok" if ok else "error",
                        detail={
                            "args": event.get("args", {}),
                            "ok": ok,
                            "blocked": bool(event.get("blocked")),
                            "empty": event.get("empty"),
                            "preview": event.get("preview"),
                        },
                    )
                    yield {
                        "event": "tool",
                        "data": {
                            "name": name,
                            "status": "end",
                            "args": event.get("args", {}),
                            "id": call_id,
                            "ok": ok,
                            "blocked": bool(event.get("blocked")),
                            "duration_ms": round(float(duration_ms), 2)
                            if duration_ms is not None
                            else None,
                            "empty": event.get("empty"),
                        },
                    }
                    continue

                if event_type == "notice":
                    yield {"event": "notice", "data": event}
                    continue

                if event_type == "result":
                    result = event
                    history_stats = event.get("history")
                    if history_stats:
                        # 压缩统计单独成 span：既能看省了多少条消息，也能看
                        # 摘要本身花了多少 token / 多少毫秒。
                        self.store.add_span(
                            trace_id,
                            "history:compress",
                            "compress",
                            duration_ms=history_stats.get("duration_ms"),
                            status="fallback"
                            if history_stats.get("fallback_calls")
                            else "ok",
                            detail=history_stats,
                        )

        except GeneratorExit:
            # 客户端断开：保留已有回答，trace 标记 cancelled。
            status = "cancelled"
            raise
        except Exception as exc:
            status = "error"
            error_detail = f"{type(exc).__name__}: {exc}"
            logger.error(f"[chat] trace={trace_id} 执行失败", exc_info=True)
            yield {
                "event": "error",
                "data": {
                    "trace_id": trace_id,
                    "code": "internal_error",
                    "message": "抱歉，服务暂时不可用，请稍后再试。",
                },
            }
        finally:
            answer = str((result or {}).get("answer") or "".join(answer_parts)).strip()
            latency_ms = (time.perf_counter() - started) * 1000
            usage = account.as_dict()
            has_usage = usage.get("source") == "provider"
            # 本来就不会调用模型的路径（越界拒答、报告越权拒答、空记录）没有用量，
            # 属于「本来就没有」，不能算作供应商漏报。
            usage_source = usage.get("source", "missing")
            no_model_call = bool((result or {}).get("no_model_call"))
            if not has_usage and (route == "out_of_scope" or no_model_call):
                usage_source = "none"
            usage["source"] = usage_source
            cost = (
                estimate_cost(
                    self.model_name,
                    usage.get("input_tokens"),
                    usage.get("output_tokens"),
                )
                if has_usage
                else None
            )
            llm_ms = (
                (last_delta_at - started) * 1000 - first_token_ms
                if last_delta_at is not None and first_token_ms is not None
                else None
            )

            try:
                self.store.finish_trace(
                    trace_id,
                    answer=answer,
                    status=status,
                    error=error_detail,
                    latency_ms=round(latency_ms, 2),
                    first_token_ms=round(first_token_ms, 2)
                    if first_token_ms is not None
                    else None,
                    llm_ms=round(llm_ms, 2) if llm_ms is not None else None,
                    input_tokens=usage.get("input_tokens") if has_usage else None,
                    output_tokens=usage.get("output_tokens") if has_usage else None,
                    total_tokens=usage.get("total_tokens") if has_usage else None,
                    cached_tokens=usage.get("cached_tokens") if has_usage else None,
                    cost_cny=cost,
                    usage_source=usage_source,
                    tool_calls=tool_count,
                )
            except Exception:
                logger.error(f"[chat] trace={trace_id} 落库失败", exc_info=True)

        if status != "ok":
            return

        yield {
            "event": "done",
            "data": {
                "trace_id": trace_id,
                "thread_id": thread_id,
                "route": route,
                "answer": str((result or {}).get("answer") or "".join(answer_parts)).strip(),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "first_token_ms": round(first_token_ms, 2)
                if first_token_ms is not None
                else None,
                "llm_ms": round(llm_ms, 2) if llm_ms is not None else None,
                "usage": usage,
                "cost_cny": cost,
                "currency": pricing_snapshot().get("currency", "CNY"),
                "tool_calls": tool_count,
                "history": (result or {}).get("history"),
            },
        }
