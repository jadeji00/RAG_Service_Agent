"""DashScope gte-rerank-v2 重排：给融合后的候选逐条重新打分。

召回阶段只回答「有没有可能相关」，向量分和 BM25 分都不能直接比；重排用
交叉编码器算 query×chunk 的相关度，把真正的答案顶到前面。任何失败（网络、
限流、鉴权、返回体异常）都返回 None，调用方退回 RRF 顺序并记日志，
保证重排挂掉时检索链路仍然可用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from utils.logger_handler import logger


def _read(obj: Any, name: str, default: Any = None) -> Any:
    """SDK 的返回体有时是对象、有时是 dict，两种都兼容。"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_results(response: Any) -> list[tuple[int, float]]:
    output = _read(response, "output")
    results = _read(output, "results") or []
    parsed: list[tuple[int, float]] = []
    for item in results:
        index = _read(item, "index")
        score = _read(item, "relevance_score")
        if index is None or score is None:
            continue
        parsed.append((int(index), float(score)))
    return parsed


@dataclass(frozen=True)
class RerankOutcome:
    scores: list[float]
    order: list[int]
    model: str
    latency_ms: float
    usage_tokens: int


class DashScopeReranker:
    def __init__(self, model: str = "gte-rerank-v2"):
        self.model = model

    def _call(self, query: str, documents: list[str]) -> Any:
        from dashscope import TextReRank

        return TextReRank.call(
            model=self.model,
            query=query,
            documents=documents,
            top_n=len(documents),
            return_documents=False,
        )

    def rerank(self, query: str, documents: list[str]) -> RerankOutcome | None:
        if not documents:
            return None

        started = time.perf_counter()
        try:
            response = self._call(query, documents)
        except Exception as exc:
            logger.warning(f"[rerank] 调用异常，退回融合顺序：{exc}")
            return None
        latency_ms = (time.perf_counter() - started) * 1000

        status_code = _read(response, "status_code")
        if status_code != 200:
            logger.warning(
                f"[rerank] 状态码 {status_code}（{_read(response, 'message')}），退回融合顺序"
            )
            return None

        parsed = _extract_results(response)
        if not parsed:
            logger.warning("[rerank] 返回结果为空，退回融合顺序")
            return None

        scores = [0.0] * len(documents)
        for index, score in parsed:
            if 0 <= index < len(scores):
                scores[index] = score

        order = sorted(range(len(documents)), key=lambda i: (-scores[i], i))
        usage_tokens = int(_read(_read(response, "usage"), "total_tokens", 0) or 0)
        return RerankOutcome(
            scores=scores,
            order=order,
            model=self.model,
            latency_ms=round(latency_ms, 2),
            usage_tokens=usage_tokens,
        )
