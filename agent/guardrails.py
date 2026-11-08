"""工具护栏：参数校验、单轮预算、重复调用拦截。

全部是纯函数，状态放在每次运行的 ``runtime.context["guard"]`` 里，
不跨请求共享；方便单测，也方便在日志/trace 里复盘为什么被拦截。
"""

from __future__ import annotations

import json
import re
from typing import Any

MAX_TOOL_CALLS = 6
MAX_IDENTICAL_ATTEMPTS = 2

_USER_ID_RE = re.compile(r"^\d{4}$")
_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_NO_ARG_TOOLS = {"get_user_location", "get_user_id", "get_current_month"}


def new_guard_state() -> dict[str, Any]:
    return {
        "executed": 0,
        "blocked": 0,
        "attempts": {},
        "completed": {},
        "force_injected": False,
    }


def canonical_args(args: dict | None) -> str:
    try:
        return json.dumps(args or {}, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(args)


def validate_tool_args(name: str, args: dict | None) -> str | None:
    """返回错误说明；None 表示通过。"""
    args = args or {}

    if name == "fetch_external_data":
        user_id = str(args.get("user_id", "")).strip()
        month = str(args.get("month", "")).strip()
        if not _USER_ID_RE.fullmatch(user_id):
            return "user_id 必须是 4 位数字字符串"
        if not _MONTH_RE.fullmatch(month):
            return "month 必须是 YYYY-MM 格式"
        return None

    if name == "get_weather":
        city = str(args.get("city", "")).strip()
        if not city:
            return "city 不能为空"
        if len(city) > 20 or "\n" in city:
            return "city 名称过长或包含换行"
        return None

    if name == "rag_summarize":
        query = str(args.get("query", "")).strip()
        if not query:
            return "query 不能为空"
        if len(query) > 500:
            return "query 过长（上限 500 字）"
        return None

    if name in _NO_ARG_TOOLS and args:
        return f"{name} 不接受参数"

    return None


def authorize_tool_call(
    name: str,
    args: dict | None,
    guard: dict[str, Any],
) -> tuple[bool, str, str]:
    """判断这次调用是否放行，并更新预算/次数状态。"""
    key = f"{name}:{canonical_args(args)}"

    error = validate_tool_args(name, args)
    if error:
        guard["blocked"] = int(guard.get("blocked", 0)) + 1
        return False, f"参数校验失败：{error}", key

    if int(guard.get("executed", 0)) >= MAX_TOOL_CALLS:
        guard["blocked"] = int(guard.get("blocked", 0)) + 1
        return (
            False,
            f"本轮工具调用已达上限（{MAX_TOOL_CALLS} 次），请直接基于已有信息回答。",
            key,
        )

    completed = guard.setdefault("completed", {})
    previous = completed.get(key)
    if previous == "ok":
        guard["blocked"] = int(guard.get("blocked", 0)) + 1
        return (
            False,
            "检测到重复工具调用（相同工具 + 相同参数），已拦截；请直接使用上一次结果。",
            key,
        )

    attempts = guard.setdefault("attempts", {})
    if previous == "error" and attempts.get(key, 0) >= MAX_IDENTICAL_ATTEMPTS:
        guard["blocked"] = int(guard.get("blocked", 0)) + 1
        return (
            False,
            "同一调用已失败多次，已拦截；请换参数或直接说明无法完成。",
            key,
        )

    guard["executed"] = int(guard.get("executed", 0)) + 1
    attempts[key] = attempts.get(key, 0) + 1
    return True, "", key


def record_tool_result(guard: dict[str, Any], key: str, ok: bool) -> None:
    guard.setdefault("completed", {})[key] = "ok" if ok else "error"


__all__ = [
    "MAX_TOOL_CALLS",
    "authorize_tool_call",
    "canonical_args",
    "new_guard_state",
    "record_tool_result",
    "validate_tool_args",
]
