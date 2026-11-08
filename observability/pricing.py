"""按配置单价把 token 折算成成本。

金额只用于演示和前后对比，真实账单以供应商控制台为准。
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.config_handler import observability_conf


@dataclass(frozen=True)
class ModelPrice:
    input_per_1k: float
    output_per_1k: float


def get_price(model_name: str | None) -> ModelPrice | None:
    pricing = (observability_conf or {}).get("pricing") or {}
    row = pricing.get(model_name or "")
    if not isinstance(row, dict):
        return None
    try:
        return ModelPrice(
            input_per_1k=float(row["input_per_1k"]),
            output_per_1k=float(row["output_per_1k"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def estimate_cost(
    model_name: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    """返回元；模型没配单价或没有用量时返回 None，不猜。"""
    price = get_price(model_name)
    if price is None:
        return None
    if input_tokens is None and output_tokens is None:
        return None
    cost = (int(input_tokens or 0) / 1000) * price.input_per_1k + (
        int(output_tokens or 0) / 1000
    ) * price.output_per_1k
    return round(cost, 6)


def pricing_snapshot() -> dict:
    return {
        "currency": (observability_conf or {}).get("currency", "CNY"),
        "models": (observability_conf or {}).get("pricing", {}),
    }
