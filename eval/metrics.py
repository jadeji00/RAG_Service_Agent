"""评测用的纯函数：文本归一化、相关性匹配、排名指标。

这里不依赖模型和向量库，方便单元测试，也方便以后复用到回答级评测。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


def normalize(text: str) -> str:
    """去掉所有空白并统一路径分隔符，避免换行/空格导致匹配失败。"""
    return "".join(str(text).replace("\\", "/").split())


def source_basename(value: str) -> str:
    """metadata 里可能是绝对路径，gold 里是文件名，统一取 basename。"""
    return Path(str(value).replace("\\", "/")).name


def is_relevant(document, item: dict) -> bool:
    """判断一条检索结果是否命中题目的 gold。

    规则：
    - 文件名必须命中 gold_sources；
    - 有 gold_terms 时以关键词为准（更精确，避免只命中章节标题）；
    - 否则看 gold_sections 是否出现在正文里。
    """
    gold_sources = {source_basename(s) for s in item.get("gold_sources", [])}
    if source_basename(document.metadata.get("source", "")) not in gold_sources:
        return False

    content = normalize(document.page_content)
    terms = item.get("gold_terms") or []
    if terms:
        return any(normalize(term) in content for term in terms)

    sections = item.get("gold_sections") or []
    return any(normalize(section) in content for section in sections)


def first_match_rank(documents: Sequence, predicate) -> int | None:
    """返回第一个满足条件的 1-based 排名；没有命中返回 None。"""
    for rank, document in enumerate(documents, start=1):
        if predicate(document):
            return rank
    return None


def first_relevant_rank(documents: Sequence, item: dict) -> int | None:
    return first_match_rank(documents, lambda doc: is_relevant(doc, item))


def first_source_rank(documents: Sequence, item: dict) -> int | None:
    """只检查文件名是否命中，用来区分「没召回文件」和「召回但没命中段落」。"""
    gold_sources = {source_basename(s) for s in item.get("gold_sources", [])}
    return first_match_rank(
        documents,
        lambda doc: source_basename(doc.metadata.get("source", "")) in gold_sources,
    )


def summarize_ranks(ranks: Iterable[int | None], ks: Sequence[int]) -> dict:
    """把每题的首次命中排名汇总成 Recall@k 与 MRR@k。

    每题只有一个 gold 答案集合，所以这里的 Recall@k 等价于 hit rate。
    """
    rank_list = list(ranks)
    total = len(rank_list)
    if total == 0:
        raise ValueError("没有可汇总的题目")

    summary = {}
    for k in ks:
        hits = [rank is not None and rank <= k for rank in rank_list]
        reciprocal = [
            1.0 / rank if rank is not None and rank <= k else 0.0
            for rank in rank_list
        ]
        summary[f"recall@{k}"] = sum(hits) / total
        summary[f"mrr@{k}"] = sum(reciprocal) / total
    return summary


def percentile(values: Sequence[float], p: float) -> float:
    """最近秩百分位，够用且不引入 numpy。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((p / 100) * (len(ordered) - 1)))))
    return ordered[index]
