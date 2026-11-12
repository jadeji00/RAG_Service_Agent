"""纯 Python BM25 索引，给混合召回补一条「字面命中」的路。

向量检索擅长语义近似，但对「E05」「尘袋」「2.4G」这类专有名词和型号
经常打不准；BM25 正好补这块。依赖里没有 jieba / rank_bm25，中文用
「单字 + 相邻双字」近似分词：「清理主刷」会切出 清/理/主/刷 与
清理/理主/主刷，既能命中完整词，也不会因为分词边界不同而漏召回。
高频字（的、机、器）靠 IDF 自动降权，不需要额外停用词表。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


def _expand_cjk_run(run: list[str]) -> list[str]:
    return [*run, *(a + b for a, b in zip(run, run[1:]))]


def tokenize(text: str) -> list[str]:
    """把文本切成 BM25 用的 token：ASCII 词 + 中文单字 + 中文双字。"""
    lowered = str(text).lower()
    # ASCII 词整体抽出，wi-fi 会得到 wi / fi，不会被中文字符循环打散。
    tokens = _ASCII_TOKEN_RE.findall(lowered)

    run: list[str] = []
    for char in lowered:
        if _CJK_CHAR_RE.match(char):
            run.append(char)
            continue
        if run:
            tokens.extend(_expand_cjk_run(run))
            run = []
    if run:
        tokens.extend(_expand_cjk_run(run))
    return tokens


@dataclass(frozen=True)
class BM25Hit:
    index: int
    score: float


class BM25Index:
    """一次性建索引、多次查询的内存倒排表；语料变化时重建即可。"""

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._lengths: list[int] = []
        self._postings: dict[str, dict[int, int]] = {}

        for doc_index, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self._lengths.append(sum(counts.values()))
            for token, freq in counts.items():
                self._postings.setdefault(token, {})[doc_index] = freq

        self._doc_count = len(self._lengths)
        self._avg_length = (
            sum(self._lengths) / self._doc_count if self._doc_count else 0.0
        )

    def __len__(self) -> int:
        return self._doc_count

    @property
    def vocab_size(self) -> int:
        return len(self._postings)

    def _idf(self, token: str) -> float:
        df = len(self._postings.get(token) or ())
        return math.log(1 + (self._doc_count - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_n: int) -> list[BM25Hit]:
        """返回分数从高到低的命中；查询词去重，避免重复词把分数抬高。"""
        if self._doc_count == 0 or top_n <= 0:
            return []

        scores: dict[int, float] = {}
        for token in set(tokenize(query)):
            posting = self._postings.get(token)
            if not posting:
                continue
            idf = self._idf(token)
            for doc_index, freq in posting.items():
                length = self._lengths[doc_index] or 1
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * length / (self._avg_length or 1.0)
                )
                contribution = idf * freq * (self.k1 + 1) / denominator
                scores[doc_index] = scores.get(doc_index, 0.0) + contribution

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            BM25Hit(index=doc_index, score=score)
            for doc_index, score in ranked[:top_n]
        ]
