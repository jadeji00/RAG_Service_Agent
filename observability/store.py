"""SQLite 版 trace / span / token 账本。

单机演示足够：每次操作独立连接，WAL 模式，避免多线程共享连接的问题。
生产环境可换成 Postgres 或 OpenTelemetry，接口保持一致即可。
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from observability.pricing import pricing_snapshot
from utils.config_handler import observability_conf
from utils.path_tool import get_abs_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    thread_id TEXT,
    route TEXT NOT NULL DEFAULT 'unknown',
    question TEXT NOT NULL,
    answer TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    error TEXT,
    model TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    latency_ms REAL,
    first_token_ms REAL,
    llm_ms REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    cached_tokens INTEGER,
    cost_cny REAL,
    usage_source TEXT,
    tool_calls INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS spans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ok',
    started_at TEXT NOT NULL,
    duration_ms REAL,
    detail TEXT
);

CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_traces_started ON traces(started_at);
"""

_ALLOWED_FINISH_FIELDS = {
    "answer",
    "status",
    "error",
    "route",
    "latency_ms",
    "first_token_ms",
    "llm_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_tokens",
    "cost_cny",
    "usage_source",
    "tool_calls",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _truncate(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    return value if len(value) <= limit else value[:limit] + "…"


class TraceStore:
    def __init__(self, db_path: str | None = None):
        configured = db_path or (observability_conf or {}).get(
            "db_path", "checkpoints/observability.sqlite3"
        )
        if not os.path.isabs(configured):
            configured = get_abs_path(configured)
        self.db_path = configured
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()

    @contextmanager
    def _db(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._db() as conn:
            conn.executescript(_SCHEMA)

    def start_trace(
        self,
        trace_id: str,
        thread_id: str,
        question: str,
        model: str | None,
        route: str = "unknown",
    ) -> None:
        with self._db() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO traces
                    (trace_id, thread_id, route, question, model, started_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (trace_id, thread_id, route, question, model, utc_now_iso()),
            )

    def update_route(self, trace_id: str, route: str) -> None:
        with self._db() as conn:
            conn.execute(
                "UPDATE traces SET route = ? WHERE trace_id = ?",
                (route, trace_id),
            )

    def add_span(
        self,
        trace_id: str,
        name: str,
        kind: str,
        duration_ms: float | None = None,
        status: str = "ok",
        detail: Any = None,
        started_at: str | None = None,
    ) -> None:
        payload = None
        if detail is not None:
            payload = json.dumps(detail, ensure_ascii=False, default=str)
        with self._db() as conn:
            conn.execute(
                """
                INSERT INTO spans
                    (trace_id, name, kind, status, started_at, duration_ms, detail)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    name,
                    kind,
                    status,
                    started_at or utc_now_iso(),
                    duration_ms,
                    payload,
                ),
            )

    def finish_trace(self, trace_id: str, **fields: Any) -> None:
        unknown = set(fields) - _ALLOWED_FINISH_FIELDS
        if unknown:
            raise ValueError(f"不支持的 trace 字段：{sorted(unknown)}")

        fields.setdefault("status", "ok")
        fields["ended_at"] = utc_now_iso()
        if "answer" in fields:
            fields["answer"] = _truncate(fields["answer"], 8000)

        columns = ", ".join(f"{key} = ?" for key in fields)
        with self._db() as conn:
            conn.execute(
                f"UPDATE traces SET {columns} WHERE trace_id = ?",
                (*fields.values(), trace_id),
            )

    def get_trace(self, trace_id: str) -> dict | None:
        with self._db() as conn:
            trace = conn.execute(
                "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
            if trace is None:
                return None
            spans = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM spans WHERE trace_id = ? ORDER BY id",
                    (trace_id,),
                ).fetchall()
            ]

        result = dict(trace)
        for span in spans:
            if span.get("detail"):
                try:
                    span["detail"] = json.loads(span["detail"])
                except json.JSONDecodeError:
                    pass
        result["spans"] = spans
        return result

    def list_traces(self, limit: int = 20) -> list[dict]:
        limit = max(1, min(int(limit), 100))
        with self._db() as conn:
            rows = conn.execute(
                """
                SELECT trace_id, thread_id, route, status, model,
                       substr(question, 1, 120) AS question,
                       substr(answer, 1, 160) AS answer_preview,
                       started_at, ended_at, latency_ms,
                       total_tokens, cost_cny
                FROM traces
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def usage_summary(self, days: int = 7) -> dict:
        days = max(1, int(days))
        since = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).isoformat(timespec="milliseconds")

        with self._db() as conn:
            totals = conn.execute(
                """
                SELECT COUNT(*) AS requests,
                       COALESCE(SUM(total_tokens), 0) AS total_tokens,
                       COALESCE(SUM(input_tokens), 0) AS input_tokens,
                       COALESCE(SUM(output_tokens), 0) AS output_tokens,
                       COALESCE(SUM(cost_cny), 0) AS cost_cny,
                       AVG(latency_ms) AS avg_latency_ms,
                       SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS ok_requests,
                       SUM(CASE WHEN usage_source = 'missing'
                                THEN 1 ELSE 0 END) AS missing_usage
                FROM traces
                WHERE started_at >= ?
                """,
                (since,),
            ).fetchone()
            by_route = conn.execute(
                """
                SELECT route,
                       COUNT(*) AS requests,
                       COALESCE(SUM(total_tokens), 0) AS total_tokens,
                       COALESCE(SUM(cost_cny), 0) AS cost_cny,
                       AVG(latency_ms) AS avg_latency_ms
                FROM traces
                WHERE started_at >= ?
                GROUP BY route
                ORDER BY requests DESC
                """,
                (since,),
            ).fetchall()
            by_day = conn.execute(
                """
                SELECT substr(started_at, 1, 10) AS day,
                       COUNT(*) AS requests,
                       COALESCE(SUM(total_tokens), 0) AS total_tokens,
                       COALESCE(SUM(cost_cny), 0) AS cost_cny
                FROM traces
                WHERE started_at >= ?
                GROUP BY day
                ORDER BY day DESC
                """,
                (since,),
            ).fetchall()

        totals_dict = dict(totals) if totals else {}
        if totals_dict.get("avg_latency_ms") is not None:
            totals_dict["avg_latency_ms"] = round(
                float(totals_dict["avg_latency_ms"]), 2
            )
        totals_dict["cost_cny"] = round(
            float(totals_dict.get("cost_cny") or 0), 6
        )

        route_rows = []
        for row in by_route:
            item = dict(row)
            if item.get("avg_latency_ms") is not None:
                item["avg_latency_ms"] = round(float(item["avg_latency_ms"]), 2)
            route_rows.append(item)

        day_rows = []
        for row in by_day:
            item = dict(row)
            item["cost_cny"] = round(float(item.get("cost_cny") or 0), 6)
            day_rows.append(item)

        return {
            "days": days,
            "since": since,
            "totals": totals_dict,
            "by_route": route_rows,
            "by_day": day_rows,
            "pricing": pricing_snapshot(),
        }
