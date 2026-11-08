"""会话索引：从 LangGraph checkpoint 库整理出可恢复的历史会话。

只读 SQLite，不改 checkpoint 的存储格式。线程 ID 只保留两个前端
（FastAPI 页面 / Streamlit）生成的 ``uuid4().hex``，评测脚本的
``answer-eval-*``、``smoke-*`` 等线程会被自然排除，避免侧栏被测试数据塞满。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

# 两个前端都是 uuid.uuid4().hex：32 位小写十六进制，不带连字符。
_SESSION_THREAD = r"^[0-9a-f]{32}$"

# UUIDv6 的时间戳是「1582-10-15 起的 100ns 数」，换算到 Unix 纪元要减掉这段。
_GREGORIAN_OFFSET_SECONDS = 12219292800


def is_session_thread(thread_id: str) -> bool:
    value = str(thread_id or "").strip()
    return len(value) == 32 and value.isascii() and all(
        c in "0123456789abcdef" for c in value
    )


def checkpoint_time_ms(checkpoint_id: str | None) -> int | None:
    """从 UUIDv6 / v7 的 checkpoint_id 前缀解出时间；解析失败返回 None。

    LangGraph 1.x 用 UUIDv6（Gregorian 纪元，100ns），旧版本可能是 UUIDv1；
    UUIDv1 的字段顺序不同，这里不做猜测，直接放弃时间。
    """
    if not checkpoint_id:
        return None
    raw = str(checkpoint_id).replace("-", "")
    if len(raw) < 32:
        return None
    try:
        version = int(raw[12], 16)
        if version == 7:  # Unix 毫秒
            return int(raw[:12], 16)
        if version == 6:  # 1582-10-15 起的 100ns
            ticks = int(raw[:15], 16)
            return ticks // 10000 - _GREGORIAN_OFFSET_SECONDS * 1000
    except ValueError:
        return None
    return None


def iso_from_ms(value: int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat(
        timespec="milliseconds"
    )


def list_checkpoint_threads(db_path: str) -> list[dict]:
    """按最后写入顺序返回真实会话线程（只读）。

    用 ``MAX(rowid)`` 排序而不是 ``checkpoint_id``，避免依赖 UUID 的具体版本；
    最近一次写入对应最近一次对话。
    """
    conn = sqlite3.connect(db_path, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT thread_id,
                   MAX(rowid) AS last_row,
                   MAX(checkpoint_id) AS last_checkpoint,
                   COUNT(*) AS checkpoints
            FROM checkpoints
            WHERE thread_id IS NOT NULL AND thread_id <> ''
            GROUP BY thread_id
            ORDER BY last_row DESC
            """
        ).fetchall()
    finally:
        conn.close()

    threads = []
    for row in rows:
        thread_id = str(row["thread_id"] or "")
        if not is_session_thread(thread_id):
            continue
        threads.append(
            {
                "thread_id": thread_id,
                "last_checkpoint": row["last_checkpoint"],
                "checkpoints": int(row["checkpoints"] or 0),
            }
        )
    return threads


def make_title(messages: list[dict], limit: int = 40) -> str:
    """用第一条用户消息做会话名；折叠换行并截断，不引入模型调用。"""
    for message in messages:
        if message.get("role") != "user":
            continue
        text = " ".join(str(message.get("content") or "").split())
        if not text:
            continue
        return text if len(text) <= limit else text[:limit] + "…"
    return ""


def count_turns(messages: list[dict]) -> int:
    return sum(1 for message in messages if message.get("role") == "user")


__all__ = [
    "checkpoint_time_ms",
    "count_turns",
    "is_session_thread",
    "iso_from_ms",
    "list_checkpoint_threads",
    "make_title",
]
