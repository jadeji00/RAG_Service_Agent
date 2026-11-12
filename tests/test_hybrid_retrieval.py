import unittest

from langchain_core.documents import Document

from rag.bm25 import BM25Index, tokenize
from rag.hybrid_retriever import HybridRetriever, _corpus_fingerprint
from rag.reranker import RerankOutcome


def make_doc(source: str, text: str) -> Document:
    return Document(page_content=text, metadata={"source": source})


CORPUS = [
    make_doc("故障排除.txt", "E05 悬崖或轮组异常，擦净底部传感器后重启机器。"),
    make_doc("100问.txt", "Q1：拖布多久洗一次？A1：每次拖地后清洗并晾干。"),
    make_doc("选购指南.txt", "大户型看单次续航 120 分钟以上，或支持断点续扫。"),
    make_doc("选购指南.txt", "自动集尘减少倒尘盒频率，代价是基站体积和尘袋耗材。"),
]


class FakeRetriever:
    """假向量检索器：不管查什么都返回固定顺序，用来验证融合逻辑。"""

    def __init__(self, documents: list[Document]):
        self._documents = documents

    def invoke(self, query: str) -> list[Document]:
        return list(self._documents)


class FakeVectorStore:
    def __init__(self, documents: list[Document], vector_order: list[Document]):
        self._documents = documents
        self._vector_order = vector_order

    def count(self) -> int:
        return len(self._documents)

    def all_documents(self) -> list[Document]:
        return list(self._documents)

    def get_retriever(self, k: int | None = None):
        limit = k if k is not None else 3
        return FakeRetriever(self._vector_order[:limit])


class TokenizeTest(unittest.TestCase):
    def test_ascii_words_are_lowercased_and_split(self):
        tokens = tokenize("Wi-Fi 2.4G")
        self.assertIn("wi", tokens)
        self.assertIn("fi", tokens)
        self.assertIn("2", tokens)
        self.assertIn("4g", tokens)

    def test_cjk_uses_unigram_and_bigram(self):
        tokens = tokenize("清理主刷")
        for expected in ("清", "理", "主", "刷", "清理", "理主", "主刷"):
            self.assertIn(expected, tokens)

    def test_mixed_text_flushes_cjk_run(self):
        tokens = tokenize("报错E05处理")
        self.assertIn("e05", tokens)
        self.assertIn("报错", tokens)
        self.assertIn("处理", tokens)
        self.assertNotIn("错e", tokens)


class BM25IndexTest(unittest.TestCase):
    def setUp(self):
        self.index = BM25Index([doc.page_content for doc in CORPUS])

    def test_exact_term_wins(self):
        hits = self.index.search("E05 报错", top_n=4)
        self.assertTrue(hits)
        self.assertEqual(hits[0].index, 0)
        self.assertGreater(hits[0].score, 0)

    def test_rare_term_outranks_common_term(self):
        # 「尘袋」只出现在最后一条，「机器」在语料里更常见，IDF 应该拉开差距。
        hits = self.index.search("尘袋", top_n=4)
        self.assertEqual(hits[0].index, 3)

    def test_top_n_limits_result(self):
        self.assertEqual(len(self.index.search("拖布", top_n=1)), 1)
        self.assertEqual(self.index.search("拖布", top_n=0), [])

    def test_unknown_query_returns_empty(self):
        self.assertEqual(self.index.search("zzz qqq", top_n=3), [])

    def test_empty_corpus(self):
        empty = BM25Index([])
        self.assertEqual(len(empty), 0)
        self.assertEqual(empty.search("任意", top_n=3), [])


class HybridRetrieverTest(unittest.TestCase):
    def setUp(self):
        # 向量路故意把 E05 那条排在第 3 位，检验 BM25 能否把它顶上来。
        self.vector_order = [CORPUS[1], CORPUS[2], CORPUS[0], CORPUS[3]]
        self.store = FakeVectorStore(CORPUS, self.vector_order)

    def test_fusion_promotes_chunk_hit_by_both_paths(self):
        retriever = HybridRetriever(self.store, vector_candidates=4, bm25_candidates=4)
        result = retriever.search("E05 报错怎么处理", k=3)
        self.assertIs(result.documents[0], CORPUS[0])
        self.assertEqual(result.stats["overlap"], 1)
        self.assertEqual(result.stats["vector_candidates"], 4)
        self.assertEqual(result.stats["fused_candidates"], 4)

    def test_vector_mode_keeps_vector_order(self):
        retriever = HybridRetriever(
            self.store, mode="vector", vector_candidates=4, bm25_candidates=4
        )
        result = retriever.search("E05 报错怎么处理", k=4)
        self.assertEqual(result.documents, self.vector_order)
        self.assertEqual(result.stats["bm25_candidates"], 0)

    def test_candidates_never_below_k(self):
        retriever = HybridRetriever(self.store, vector_candidates=2, bm25_candidates=2)
        result = retriever.search("拖布", k=4)
        self.assertEqual(result.stats["vector_candidates"], 4)
        self.assertEqual(len(result.documents), 4)

    def test_index_rebuilds_when_corpus_changes(self):
        retriever = HybridRetriever(self.store, vector_candidates=4, bm25_candidates=4)
        first = retriever.search("尘袋", k=2)
        self.assertEqual(first.stats["index_size"], 4)

        grown = CORPUS + [make_doc("维护保养.txt", "尘袋每 1–2 个月更换一次。")]
        self.store._documents = grown
        self.store._vector_order = grown
        # k 取满 5 条：新增那条在向量路里排最后，只有索引重建后才会被 BM25 捞回来。
        second = retriever.search("尘袋", k=5)
        self.assertEqual(second.stats["index_size"], 5)
        self.assertNotEqual(
            first.stats["index_fingerprint"], second.stats["index_fingerprint"]
        )
        returned = {doc.page_content for doc in second.documents}
        self.assertIn(grown[-1].page_content, returned)

    def test_refresh_is_noop_when_corpus_unchanged(self):
        retriever = HybridRetriever(self.store)
        retriever.refresh()
        fingerprint = retriever._fingerprint
        index = retriever._index
        retriever.refresh()
        self.assertIs(retriever._index, index)
        self.assertEqual(retriever._fingerprint, fingerprint)

    def test_stats_expose_latency_and_fingerprint(self):
        retriever = HybridRetriever(self.store)
        stats = retriever.search("拖布", k=2).stats
        for key in (
            "mode",
            "vector_latency_ms",
            "bm25_latency_ms",
            "rrf_k",
            "index_fingerprint",
        ):
            self.assertIn(key, stats)
        self.assertEqual(stats["index_fingerprint"], _corpus_fingerprint(CORPUS))


class StubReranker:
    """假重排器：按给定的 (下标, 分数) 返回，或返回 None 模拟调用失败。"""

    def __init__(self, ranked=None, outcome=None):
        self._ranked = ranked
        self._outcome = outcome
        self.calls: list[tuple[str, int]] = []

    def rerank(self, query: str, documents: list[str]):
        self.calls.append((query, len(documents)))
        if self._outcome is not None:
            return self._outcome
        if self._ranked is None:
            return None
        scores = [0.0] * len(documents)
        for index, score in self._ranked:
            scores[index] = score
        order = [index for index, _ in self._ranked]
        return RerankOutcome(
            scores=scores, order=order, model="stub-rerank", latency_ms=1.5, usage_tokens=7
        )


class RerankStageTest(unittest.TestCase):
    def setUp(self):
        # 向量路的顺序故意是「答案在最后」，靠重排把它拉到首位。
        self.vector_order = [CORPUS[1], CORPUS[2], CORPUS[3], CORPUS[0]]
        self.store = FakeVectorStore(CORPUS, self.vector_order)

    def test_rerank_reorders_fused_pool(self):
        reranker = StubReranker(ranked=[(3, 0.95), (0, 0.2), (1, 0.1), (2, 0.05)])
        retriever = HybridRetriever(
            self.store, mode="vector", vector_candidates=4, reranker=reranker
        )
        result = retriever.search("E05", k=2)
        self.assertEqual(result.documents[0], CORPUS[0])
        self.assertEqual(result.stats["rerank"], "ok")
        self.assertEqual(result.stats["rerank_top_score"], 0.95)
        self.assertEqual(result.stats["rerank_usage_tokens"], 7)
        self.assertEqual(reranker.calls[0][1], 4)

    def test_min_score_drops_weak_candidates(self):
        reranker = StubReranker(ranked=[(3, 0.4), (0, 0.2), (1, 0.01), (2, 0.0)])
        retriever = HybridRetriever(
            self.store, mode="vector", vector_candidates=4, reranker=reranker, min_score=0.05
        )
        result = retriever.search("E05", k=4)
        self.assertEqual(result.documents, [CORPUS[0], CORPUS[1]])
        self.assertEqual(result.stats["rerank_kept"], 2)

    def test_all_below_threshold_returns_empty(self):
        reranker = StubReranker(ranked=[(0, 0.01), (1, 0.0)])
        retriever = HybridRetriever(
            self.store, mode="vector", vector_candidates=4, reranker=reranker, min_score=0.3
        )
        result = retriever.search("随便问点别的", k=3)
        self.assertEqual(result.documents, [])
        self.assertEqual(result.stats["rerank_kept"], 0)

    def test_rerank_failure_falls_back_to_fusion_order(self):
        retriever = HybridRetriever(
            self.store, mode="vector", vector_candidates=4, reranker=StubReranker()
        )
        result = retriever.search("E05", k=4)
        self.assertEqual(result.documents, self.vector_order)
        self.assertEqual(result.stats["rerank"], "fallback")

    def test_disabled_when_no_reranker(self):
        retriever = HybridRetriever(self.store, mode="vector", vector_candidates=4)
        result = retriever.search("E05", k=2)
        self.assertEqual(result.stats["rerank"], "disabled")
        self.assertEqual(result.documents, self.vector_order[:2])

    def test_pool_size_is_at_least_k(self):
        reranker = StubReranker(ranked=[(0, 0.9)])
        retriever = HybridRetriever(
            self.store, mode="vector", vector_candidates=4, reranker=reranker, rerank_candidates=2
        )
        retriever.search("E05", k=4)
        self.assertEqual(reranker.calls[0][1], 4)


if __name__ == "__main__":
    unittest.main()