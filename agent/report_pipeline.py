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

from agent.report_router import (
    ReportRequest,
    is_report_query,
    parse_month,
    parse_user_id,
)
from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts

_EMPTY_RECORD_ANSWER = "未找到用户 {user_id} 在 {month} 的使用记录，无法生成报告。"


@lru_cache(maxsize=1)
def _report_model():
    """报告写作单独用 temperature=0，尽量让同一记录的可复现性更好。"""
    from langchain_community.chat_models.tongyi import ChatTongyi
    from utils.config_handler import rag_conf

    return ChatTongyi(model=rag_conf["chat_model_name"], temperature=0)


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
) -> ReportRequest:
    """补齐报告参数：显式入参优先，其次从问题里抽，最后用 mock 工具补。"""
    resolved_id = str(user_id or "").strip() or parse_user_id(query) or _mock_user_id()
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


def generate_report(
    query: str,
    user_id: str | None = None,
    month: str | None = None,
) -> dict:
    """跑固定报告流程，返回回答和可审计的执行信息。

    ``tool_calls`` 只记录真实的取数动作，供评测核对参数；这里没有
    ``fill_context_for_report``，因为提示词切换由本函数直接决定。
    """
    started = time.perf_counter()
    request = resolve_report_request(query, user_id=user_id, month=month)
    record_raw = _fetch_record(request.user_id, request.month)
    tool_calls = [
        {
            "name": "fetch_external_data",
            "args": {"user_id": request.user_id, "month": request.month},
        }
    ]

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
        "latency_ms": round(latency_ms, 2),
    }


__all__ = [
    "ReportRequest",
    "generate_report",
    "is_report_query",
    "resolve_report_request",
]
