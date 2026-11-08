"""从模型响应里提取 token 用量，并做单次请求内的累加。

DashScope 的 ChatTongyi 在流式最后一帧把用量放在
``response_metadata["token_usage"]``，而不是 ``usage_metadata``；
这里两种都兼容，并用 ``request_id`` 去重，避免同一帧被重复累加。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def normalize_usage(usage: Any) -> dict[str, int] | None:
    """把不同供应商的字段名统一成 input/output/total。"""
    if not isinstance(usage, dict):
        return None

    input_tokens = usage.get("input_tokens")
    if input_tokens is None:
        input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("output_tokens")
    if output_tokens is None:
        output_tokens = usage.get("completion_tokens")

    if input_tokens is None and output_tokens is None:
        return None

    input_tokens = int(input_tokens or 0)
    output_tokens = int(output_tokens or 0)
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = int(details.get("cached_tokens") or 0)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": int(total_tokens),
        "cached_tokens": cached_tokens,
    }


def extract_usage(message: Any) -> dict[str, Any] | None:
    """从 AIMessage / AIMessageChunk 里取一次模型调用的用量。

    返回 ``None`` 表示这个 chunk 没有用量信息（大多数中间 chunk 都是这样）。
    """
    if message is None:
        return None

    metadata = getattr(message, "response_metadata", None) or {}
    usage = metadata.get("token_usage") or metadata.get("usage")
    normalized = normalize_usage(usage)
    if normalized is None:
        normalized = normalize_usage(getattr(message, "usage_metadata", None))
    if normalized is None:
        return None

    normalized["request_id"] = metadata.get("request_id")
    normalized["model"] = metadata.get("model_name")
    return normalized


@dataclass
class TokenAccount:
    """一次请求内的 token 累加器，跨多轮模型调用。"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 0
    source: str = "missing"
    _request_ids: set[str] = field(default_factory=set, repr=False)

    def add(self, usage: dict[str, Any] | None) -> bool:
        if not usage:
            return False

        request_id = usage.get("request_id")
        if request_id and request_id in self._request_ids:
            return False
        if request_id:
            self._request_ids.add(request_id)

        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or 0)
        self.total_tokens += int(usage.get("total_tokens") or 0)
        self.cached_tokens += int(usage.get("cached_tokens") or 0)
        self.calls += 1
        self.source = "provider"
        return True

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("_request_ids", None)
        return data
