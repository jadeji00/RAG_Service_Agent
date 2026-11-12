"""引用溯源：把检索到的 chunk 压成前端能渲染、trace 能落库的引用条目。

单独成模块是因为这三件事都得能离线单测——``build_sources`` 依赖 Document 的
元数据约定，``merge_sources`` 依赖「一轮里可能检索多次」的行为，
``summarize_retrieval_stats`` 是把多次统计合成一条的口径。放在
``RagSummarizeService`` 里就必须先构造向量库才能测，代价太大。
"""

from __future__ import annotations

from langchain_core.documents import Document

# 一轮里最多展示多少条引用。k=3 时两次检索也就 6 条，12 是防呆上限。
MAX_SOURCES = 12
# 片段预览长度：够在界面上认出是哪一段，又不会把整块正文塞进 checkpoint。
PREVIEW_CHARS = 60


def _file_name(source: object) -> str:
    """``source`` 元数据是本地绝对路径，只取文件名。

    完整路径会把服务器目录结构推到浏览器上，没有必要也不安全。
    """
    return str(source or "").replace("\\", "/").rsplit("/", 1)[-1]


def build_sources(documents: list[Document]) -> list[dict]:
    """按检索顺序把 Document 转成引用条目。

    ``score`` 是重排模型给的相关度，由 ``HybridRetriever`` 写进 metadata；
    重排关闭或降级时它是 None，界面就不显示分数——不能填 0 冒充「不相关」。
    """
    sources = []
    for counter, document in enumerate(documents or [], start=1):
        metadata = document.metadata or {}
        score = metadata.get("rerank_score")
        sources.append(
            {
                "n": counter,
                "file": _file_name(metadata.get("source")),
                "section": str(metadata.get("section") or ""),
                "unit": str(metadata.get("unit") or ""),
                "score": round(float(score), 4)
                if isinstance(score, (int, float))
                else None,
                "preview": " ".join(str(document.page_content or "").split())[
                    :PREVIEW_CHARS
                ],
            }
        )
    return sources


def merge_sources(groups: list[list[dict]], limit: int = MAX_SOURCES) -> list[dict]:
    """把一轮里多次检索的引用去重合并，再重新编号。

    去重键是「文件 + 小节 + 片段 + 预览」。同一段资料被两次检索命中只留一条，
    但分数取较高的那次——两次检索的 query 不同，同一段资料的相关度本来就不一样。
    编号按首次出现顺序重排，界面上的序号才是连续的。
    """
    merged: dict[tuple, dict] = {}
    order: list[tuple] = []

    for group in groups or []:
        for item in group or []:
            if not isinstance(item, dict):
                continue
            key = (
                item.get("file"),
                item.get("section"),
                item.get("unit"),
                item.get("preview"),
            )
            existing = merged.get(key)
            if existing is None:
                merged[key] = dict(item)
                order.append(key)
                continue
            score = item.get("score")
            old = existing.get("score")
            if isinstance(score, (int, float)) and (
                not isinstance(old, (int, float)) or score > old
            ):
                existing["score"] = round(float(score), 4)

    result = []
    for position, key in enumerate(order[:limit], start=1):
        entry = dict(merged[key])
        entry["n"] = position
        result.append(entry)
    return result


def _total(items: list[dict], key: str) -> int:
    return sum(int(item.get(key) or 0) for item in items)


def _seconds_sum(items: list[dict], key: str) -> float:
    return round(sum(float(item.get(key) or 0.0) for item in items), 2)


def summarize_retrieval_stats(items: list[dict]) -> dict:
    """把一轮里多次检索的统计合成一条，供 trace span 落库。

    候选数、延迟、重排 token 用求和（一轮真的花了这么多）；``mode`` 和
    ``index_size`` 取最后一次（同一轮里不会变，变了说明语料被并发改了）；
    ``rerank_top_score`` 取最大值，代表这一轮最强的一条证据有多相关。
    """
    calls = [item for item in items or [] if isinstance(item, dict)]
    if not calls:
        return {"calls": 0}

    modes = {str(call.get("rerank")) for call in calls if call.get("rerank")}
    top_scores = [
        float(call["rerank_top_score"])
        for call in calls
        if isinstance(call.get("rerank_top_score"), (int, float))
    ]

    summary = {
        "calls": len(calls),
        "mode": calls[-1].get("mode"),
        "index_size": calls[-1].get("index_size"),
        "vector_candidates": _total(calls, "vector_candidates"),
        "bm25_candidates": _total(calls, "bm25_candidates"),
        "fused_candidates": _total(calls, "fused_candidates"),
        "overlap": _total(calls, "overlap"),
        "rerank_kept": _total(calls, "rerank_kept"),
        "rerank_usage_tokens": _total(calls, "rerank_usage_tokens"),
        "rerank": ",".join(sorted(modes)),
        "vector_latency_ms": _seconds_sum(calls, "vector_latency_ms"),
        "bm25_latency_ms": _seconds_sum(calls, "bm25_latency_ms"),
        "rerank_latency_ms": _seconds_sum(calls, "rerank_latency_ms"),
    }
    if top_scores:
        summary["rerank_top_score"] = round(max(top_scores), 4)
    return summary


__all__ = [
    "MAX_SOURCES",
    "PREVIEW_CHARS",
    "build_sources",
    "merge_sources",
    "summarize_retrieval_stats",
]
