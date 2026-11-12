"""确定性报告子流程：参数解析 → 取数 → 单次写作。

旧机制把报告流程交给 ReAct 循环：提示词要求模型先调用
``fill_context_for_report`` 再取数，模型一旦跳过标记工具，报告提示词不会
切换，字段就缺失（P7）。这里把报告请求从工具选择里拿出来，用代码固定顺序，
模型只负责最后一步写作，无法跳过取数，也不会多跑一轮工具。
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Any

from langchain_community.chat_models.tongyi import ChatTongyi

from agent.report_router import (
    ReportRequest,
    is_report_query,
    parse_month,
    parse_user_id,
)
from observability.usage import extract_usage
from utils.config_handler import rag_conf
from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts

_EMPTY_RECORD_ANSWER = "未找到用户 {user_id} 在 {month} 的使用记录，无法生成报告。"

# 登录账号绑定了业务用户 ID 时，查别人的记录直接拒答，不取数也不调模型。
OWNER_DENIED_ANSWER = (
    "只能查询当前登录账号自己的使用记录（本账号绑定用户 {owner}），"
    "不能读取用户 {requested} 的数据。"
    "如果 {requested} 也是你的账号，请用它登录后再试。"
)


@lru_cache(maxsize=1)
def _report_model():
    """报告写作单独用 temperature=0，尽量让同一记录的可复现性更好。"""
    return ChatTongyi(model=rag_conf["chat_model_name"], temperature=0)


@lru_cache(maxsize=1)
def _report_stream_model():
    """流式写作专用：强制增量输出，SSE 才有 token 级效果。"""
    return ChatTongyi(
        model=rag_conf["chat_model_name"],
        temperature=0,
        streaming=True,
        model_kwargs={"incremental_output": True},
    )


def _fetch_record(user_id: str, month: str) -> str:
    from agent.tools.agent_tools import fetch_external_data

    return str(fetch_external_data.invoke({"user_id": user_id, "month": month}) or "").strip()


def _mock_user_id() -> str:
    from agent.tools.agent_tools import get_user_id

    return str(get_user_id.invoke({})).strip()


def _mock_month() -> str:
    from agent.tools.agent_tools import get_current_month

    return str(get_current_month.invoke({})).strip()


def _safe_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(part if isinstance(part, str) else str(part) for part in value)
    return str(value)


def resolve_report_request(
    query: str,
    user_id: str | None = None,
    month: str | None = None,
    owner_user_id: str | None = None,
) -> ReportRequest:
    """补齐报告参数：显式入参 > 问题里抽的 > 登录身份 > mock 工具。

    登录身份排在 mock 之前：登录后问「我的报告」没带 ID，用 mock 抽一个随机
    用户号会撞上归属校验，变成拒绝自己的查询。
    """
    resolved_id = (
        str(user_id or "").strip()
        or parse_user_id(query)
        or str(owner_user_id or "").strip()
        or _mock_user_id()
    )
    resolved_month = (
        parse_month(month) if month else parse_month(query)
    ) or _mock_month()

    if not resolved_id or not resolved_month:
        raise ValueError("无法确定报告所需的 user_id / month")
    return ReportRequest(user_id=resolved_id, month=resolved_month)


def _write_report(system_prompt: str, question: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    response = _report_model().invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]
    )
    return _safe_text(getattr(response, "content", "")).strip()


def _iter_report_text(system_prompt: str, question: str):
    """流式写作；没有工具，DashScope 可以稳定做到增量输出。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    yield from _report_stream_model().stream(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]
    )


def check_report_owner(request: ReportRequest, owner_user_id: str | None) -> str | None:
    """归属校验：返回拒答文本，``None`` 表示放行。

    ``owner_user_id`` 为空代表调用方没有登录态（评测脚本、CLI、Streamlit），
    这时不做校验，保持旧行为。
    """
    owner = str(owner_user_id or "").strip()
    if not owner or request.user_id == owner:
        return None
    logger.warning(
        f"[report pipeline] 越权查询被拦截：登录绑定 {owner}，请求 {request.user_id}"
    )
    return OWNER_DENIED_ANSWER.format(owner=owner, requested=request.user_id)


def _prepare_report(request: ReportRequest) -> tuple[str, list[dict], float]:
    fetch_started = time.perf_counter()
    record_raw = _fetch_record(request.user_id, request.month)
    fetch_ms = (time.perf_counter() - fetch_started) * 1000
    tool_calls = [
        {
            "name": "fetch_external_data",
            "args": {"user_id": request.user_id, "month": request.month},
        }
    ]
    return record_raw, tool_calls, fetch_ms


def generate_report(
    query: str,
    user_id: str | None = None,
    month: str | None = None,
    owner_user_id: str | None = None,
) -> dict:
    """跑固定报告流程，返回回答和可审计的执行信息。

    ``tool_calls`` 只记录真实的取数动作，供评测核对参数；这里没有
    ``fill_context_for_report``，因为提示词切换由本函数直接决定。
    """
    started = time.perf_counter()
    request = resolve_report_request(
        query, user_id=user_id, month=month, owner_user_id=owner_user_id
    )

    denied = check_report_owner(request, owner_user_id)
    if denied:
        return {
            "route": "report_pipeline",
            "request": request,
            "answer": denied,
            "record": {},
            "tool_calls": [],
            "fetch_args_ok": True,
            "owner_denied": True,
            "no_model_call": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    record_raw, tool_calls, _ = _prepare_report(request)

    if not record_raw:
        answer = _EMPTY_RECORD_ANSWER.format(
            user_id=request.user_id, month=request.month
        )
        record: dict = {}
        logger.warning(
            f"[report pipeline] 未找到 {request.user_id}/{request.month} 的使用记录"
        )
    else:
        try:
            record = json.loads(record_raw)
        except json.JSONDecodeError:
            record = {}
        system_prompt = (
            load_report_prompts().rstrip()
            + "\n\n【本次使用记录】\n"
            + record_raw
            + "\n\n请只依据以上记录和用户问题，直接输出报告。"
        )
        answer = _write_report(system_prompt, query)

    latency_ms = (time.perf_counter() - started) * 1000
    logger.info(
        f"[report pipeline] user={request.user_id} month={request.month} "
        f"empty={int(not record_raw)} latency={latency_ms:.0f}ms"
    )
    return {
        "route": "report_pipeline",
        "request": request,
        "answer": answer,
        "record": record,
        "tool_calls": tool_calls,
        "fetch_args_ok": True,
        "no_model_call": not bool(record_raw),
        "latency_ms": round(latency_ms, 2),
    }


def stream_report(
    query: str,
    user_id: str | None = None,
    month: str | None = None,
    usage_sink=None,
    owner_user_id: str | None = None,
):
    """流式版报告流程，产出 meta / tool / delta / result 事件。

    与 ``generate_report`` 共用参数解析和取数逻辑，区别只在写作阶段：
    这里用增量输出，让前端能边生成边显示。
    """
    started = time.perf_counter()
    request = resolve_report_request(
        query, user_id=user_id, month=month, owner_user_id=owner_user_id
    )

    denied = check_report_owner(request, owner_user_id)
    if denied:
        yield {
            "type": "meta",
            "route": "report",
            "request": {"user_id": request.user_id, "month": request.month},
            "tool_calls": [],
            "fetch_args_ok": True,
            "owner_denied": True,
        }
        yield {"type": "delta", "text": denied}
        yield {
            "type": "result",
            "route": "report",
            "answer": denied,
            "record": {},
            "tool_calls": [],
            "fetch_args_ok": True,
            "owner_denied": True,
            "no_model_call": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        return

    record_raw, tool_calls, fetch_ms = _prepare_report(request)
    args = {"user_id": request.user_id, "month": request.month}

    yield {
        "type": "meta",
        "route": "report",
        "request": {"user_id": request.user_id, "month": request.month},
        "tool_calls": tool_calls,
        "fetch_args_ok": True,
    }
    yield {
        "type": "tool",
        "name": "fetch_external_data",
        "status": "end",
        "args": args,
        "ok": True,
        "empty": not bool(record_raw),
        "duration_ms": round(fetch_ms, 2),
    }

    if not record_raw:
        answer = _EMPTY_RECORD_ANSWER.format(
            user_id=request.user_id, month=request.month
        )
        record: dict = {}
        logger.warning(
            f"[report pipeline] 未找到 {request.user_id}/{request.month} 的使用记录"
        )
        yield {"type": "delta", "text": answer}
        yield {
            "type": "result",
            "route": "report",
            "answer": answer,
            "record": record,
            "tool_calls": tool_calls,
            "fetch_args_ok": True,
            "empty": True,
            "no_model_call": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        return

    try:
        record = json.loads(record_raw)
    except json.JSONDecodeError:
        record = {}

    system_prompt = (
        load_report_prompts().rstrip()
        + "\n\n【本次使用记录】\n"
        + record_raw
        + "\n\n请只依据以上记录和用户问题，直接输出报告。"
    )

    parts: list[str] = []
    for chunk in _iter_report_text(system_prompt, query):
        usage = extract_usage(chunk)
        if usage and usage_sink:
            usage_sink(usage)
        text = _safe_text(getattr(chunk, "content", ""))
        if text:
            parts.append(text)
            yield {"type": "delta", "text": text}

    answer = "".join(parts).strip()
    if not answer:
        raise RuntimeError("报告模型没有返回内容")

    latency_ms = (time.perf_counter() - started) * 1000
    logger.info(
        f"[report pipeline] stream user={request.user_id} month={request.month} "
        f"latency={latency_ms:.0f}ms chars={len(answer)}"
    )
    yield {
        "type": "result",
        "route": "report",
        "answer": answer,
        "record": record,
        "tool_calls": tool_calls,
        "fetch_args_ok": True,
        "empty": False,
        "latency_ms": round(latency_ms, 2),
    }


__all__ = [
    "ReportRequest",
    "check_report_owner",
    "generate_report",
    "is_report_query",
    "resolve_report_request",
    "stream_report",
]
