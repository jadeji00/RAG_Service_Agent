"""报告请求的纯函数路由：判断意图、抽取用户 ID 与月份。

这里不导入模型、向量库或工具，方便单测；真正的取数与写作在
``agent.report_pipeline`` 里完成。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_USER_ID_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")
_MONTH_RE = re.compile(r"(?<!\d)(20\d{2})\s*[-/.年]\s*(\d{1,2})(?!\d)\s*月?")
_REPORT_WORDS = ("报告", "使用记录", "使用情况")
_REPORT_DATA_WORDS = ("清洁效率", "耗材", "对比", "特征", "数据")


@dataclass(frozen=True)
class ReportRequest:
    user_id: str
    month: str


def parse_user_id(text: str) -> str | None:
    """从文本里取 4 位用户 ID；排除 19xx / 20xx 年份。"""
    for match in _USER_ID_RE.finditer(str(text or "")):
        value = match.group(1)
        if not value.startswith(("19", "20")):
            return value
    return None


def parse_month(text: str) -> str | None:
    """把 2026-10 / 2026/10 / 2026年10月 统一成 YYYY-MM。"""
    match = _MONTH_RE.search(str(text or ""))
    if not match:
        return None
    year, month = match.group(1), int(match.group(2))
    if not 1 <= month <= 12:
        return None
    return f"{year}-{month:02d}"


def is_report_query(text: str) -> bool:
    """判断一句话是否在要个人使用报告。

    规则优先覆盖明确词，其次要求「用户 ID + 月份 + 报告数据词」同时出现，
    避免把「耗材多久换一次」这类知识问答误判成报告。
    """
    value = str(text or "").strip()
    if not value:
        return False
    if any(word in value for word in _REPORT_WORDS):
        return True

    user_id = parse_user_id(value)
    month = parse_month(value)
    return bool(
        user_id
        and month
        and any(word in value for word in _REPORT_DATA_WORDS)
    )


def extract_report_request(text: str) -> ReportRequest | None:
    """只从文本抽取参数；缺 ID/月份时返回 None，由调用方决定怎么补。"""
    user_id = parse_user_id(text)
    month = parse_month(text)
    if not user_id or not month:
        return None
    return ReportRequest(user_id=user_id, month=month)
