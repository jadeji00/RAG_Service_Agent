"""SSE 编码：事件名 + 单行 JSON data。

前端用 fetch 读流，按 ``\\n\\n`` 分块解析；JSON 里的换行会被保留为
多行 ``data:``，这里把每一行都加上前缀，符合 SSE 规范。
"""

from __future__ import annotations

import json
from typing import Any


def encode_sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    lines = payload.splitlines() or [""]
    body = "".join(f"data: {line}\n" for line in lines)
    return f"event: {event}\n{body}\n"
