import unittest
from types import SimpleNamespace

from rag.reranker import DashScopeReranker, _extract_results, _read


def attr_response(results, status_code=200, total_tokens=42, message=None):
    return SimpleNamespace(
        status_code=status_code,
        message=message,
        output=SimpleNamespace(
            results=[
                SimpleNamespace(index=index, relevance_score=score, document=None)
                for index, score in results
            ]
        ),
        usage=SimpleNamespace(total_tokens=total_tokens),
    )


def dict_response(results, status_code=200, total_tokens=42):
    return {
        "status_code": status_code,
        "output": {
            "results": [
                {"index": index, "relevance_score": score} for index, score in results
            ]
        },
        "usage": {"total_tokens": total_tokens},
    }


class StubReranker(DashScopeReranker):
    """把网络调用换成本地桩，只验证解析、排序和兜底逻辑。"""

    def __init__(self, response=None, error=None):
        super().__init__(model="gte-rerank-v2")
        self._response = response
        self._error = error
        self.calls: list[tuple[str, list[str]]] = []

    def _call(self, query, documents):
        self.calls.append((query, list(documents)))
        if self._error is not None:
            raise self._error
        return self._response


DOCS = ["E05 悬崖或轮组异常", "拖布多久洗一次", "自动集尘减少倒尘盒频率"]


class RerankParsingTest(unittest.TestCase):
    def test_attribute_response_is_reordered_by_score(self):
        reranker = StubReranker(attr_response([(0, 0.12), (1, 0.91), (2, 0.03)]))
        outcome = reranker.rerank("拖布", DOCS)
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.order, [1, 0, 2])
        self.assertEqual(outcome.scores, [0.12, 0.91, 0.03])
        self.assertEqual(outcome.usage_tokens, 42)
        self.assertEqual(outcome.model, "gte-rerank-v2")
        self.assertEqual(reranker.calls[0][1], DOCS)

    def test_dict_response_is_supported(self):
        reranker = StubReranker(dict_response([(2, 0.8), (0, 0.4), (1, 0.1)]))
        outcome = reranker.rerank("尘袋", DOCS)
        self.assertEqual(outcome.order, [2, 0, 1])
        self.assertEqual(outcome.scores[2], 0.8)

    def test_out_of_range_index_is_ignored(self):
        reranker = StubReranker(attr_response([(0, 0.5), (9, 0.9)]))
        outcome = reranker.rerank("E05", DOCS)
        self.assertEqual(outcome.order[0], 0)
        self.assertEqual(len(outcome.scores), len(DOCS))


class RerankFallbackTest(unittest.TestCase):
    def test_non_200_returns_none(self):
        reranker = StubReranker(attr_response([(0, 0.5)], status_code=429, message="throttled"))
        self.assertIsNone(reranker.rerank("E05", DOCS))

    def test_exception_returns_none(self):
        reranker = StubReranker(error=RuntimeError("network down"))
        self.assertIsNone(reranker.rerank("E05", DOCS))

    def test_empty_results_returns_none(self):
        reranker = StubReranker(attr_response([]))
        self.assertIsNone(reranker.rerank("E05", DOCS))

    def test_empty_documents_skips_api_call(self):
        reranker = StubReranker(attr_response([(0, 0.5)]))
        self.assertIsNone(reranker.rerank("E05", []))
        self.assertEqual(reranker.calls, [])


class ReadHelperTest(unittest.TestCase):
    def test_read_handles_object_dict_and_none(self):
        self.assertEqual(_read(SimpleNamespace(a=1), "a"), 1)
        self.assertEqual(_read({"a": 2}, "a"), 2)
        self.assertEqual(_read(None, "a", "fallback"), "fallback")
        self.assertEqual(_read(SimpleNamespace(), "missing", 0), 0)

    def test_extract_results_skips_incomplete_items(self):
        response = SimpleNamespace(
            output={"results": [{"index": 0}, {"relevance_score": 0.5}, {"index": 1, "relevance_score": 0.2}]}
        )
        self.assertEqual(_extract_results(response), [(1, 0.2)])


if __name__ == "__main__":
    unittest.main()
