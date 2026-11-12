"""混合召回：向量（语义）+ BM25（字面）双路取候选，再用 RRF 融合排序。

RRF 只看排名不看分数，天然规避了「余弦相似度和 BM25 分数不在一个量纲」
的问题；``mode: vector`` 时 BM25 那一路为空，融合结果就等于向量原序，
所以两种模式共用同一条代码路径，方便评测里做 A/B。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from time import perf_counter

from langchain_core.documents import Document

from rag.bm25 import BM25Index
from utils.logger_handler import logger


def _corpus_fingerprint(documents: list[Document]) -> str:
    joined = "\n".join(
        sorted(f"{doc.metadata.get('source', '')}|{doc.page_content}" for doc in documents)
    )
    return hashlib.md5(joined.encode("utf-8")).hexdigest()[:12]


def _doc_key(document: Document) -> tuple[str, str]:
    """同一条 chunk 在两条召回路上要能被认出来是同一个，用来源+正文当主键。"""
    return (str(document.metadata.get("source", "")), document.page_content)


@dataclass(frozen=True)
class HybridSearchResult:
    documents: list[Document]
    stats: dict = field(default_factory=dict)


class HybridRetriever:
    def __init__(
        self,
        vector_store,
        *,
        mode: str = "hybrid",
        vector_candidates: int = 20,
        bm25_candidates: int = 20,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.mode = mode
        self.vector_candidates = vector_candidates
        self.bm25_candidates = bm25_candidates
        self.rrf_k = rrf_k
        self._documents: list[Document] = []
        self._index: BM25Index | None = None
        self._fingerprint: str | None = None

    def refresh(self) -> None:
        """按当前向量库内容重建 BM25 索引；语料没变则直接复用。"""
        documents = self.vector_store.all_documents()
        fingerprint = _corpus_fingerprint(documents)
        if self._index is not None and fingerprint == self._fingerprint:
            return

        self._documents = documents
        self._index = BM25Index([doc.page_content for doc in documents])
        self._fingerprint = fingerprint
        logger.info(
            f"[检索] BM25 索引已重建：{len(documents)} chunks，"
            f"词表 {self._index.vocab_size}，指纹 {fingerprint}"
        )

    def _ensure_index(self) -> None:
        if self._index is None:
            self.refresh()
        elif self.vector_store.count() != len(self._documents):
            # 向量库被别的进程改过（新灌库/删文件），索引跟着补一次。
            self.refresh()

    def _vector_search(self, query: str, limit: int) -> list[Document]:
        return self.vector_store.get_retriever(k=limit).invoke(query)

    def _bm25_search(self, query: str, limit: int) -> list[Document]:
        if self.mode != "hybrid" or self._index is None:
            return []
        hits = self._index.search(query, limit)
        return [self._documents[hit.index] for hit in hits]

    def _fuse(self, ranked_lists: list[list[Document]]) -> tuple[list[Document], int]:
        scores: dict[tuple[str, str], float] = {}
        documents: dict[tuple[str, str], Document] = {}
        seen_in: dict[tuple[str, str], int] = {}

        for list_index, ranked in enumerate(ranked_lists):
            for rank, document in enumerate(ranked, start=1):
                key = _doc_key(document)
                documents.setdefault(key, document)
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
                seen_in[key] = seen_in.get(key, 0) | (1 << list_index)

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        both_paths = sum(1 for mask in seen_in.values() if mask == 3)
        return [documents[key] for key, _ in ordered], both_paths

    def search(self, query: str, k: int) -> HybridSearchResult:
        """召回并融合，返回前 k 条；候选数至少为 k，避免评测调大 k 时召不满。"""
        self._ensure_index()
        vector_limit = max(self.vector_candidates, k)
        bm25_limit = max(self.bm25_candidates, k)

        started = perf_counter()
        vector_docs = self._vector_search(query, vector_limit)
        vector_ms = (perf_counter() - started) * 1000

        started = perf_counter()
        bm25_docs = self._bm25_search(query, bm25_limit)
        bm25_ms = (perf_counter() - started) * 1000

        fused, overlap = self._fuse([vector_docs, bm25_docs])
        stats = {
            "mode": self.mode,
            "vector_candidates": len(vector_docs),
            "bm25_candidates": len(bm25_docs),
            "fused_candidates": len(fused),
            "overlap": overlap,
            "rrf_k": self.rrf_k,
            "index_size": len(self._documents),
            "index_fingerprint": self._fingerprint,
            "vector_latency_ms": round(vector_ms, 2),
            "bm25_latency_ms": round(bm25_ms, 2),
        }
        return HybridSearchResult(documents=fused[:k], stats=stats)
