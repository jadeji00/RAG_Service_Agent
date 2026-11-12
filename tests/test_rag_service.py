import unittest

from langchain_core.documents import Document

from agent.scope_router import OUT_OF_SCOPE_ANSWER
from rag.rag_service import NO_EVIDENCE_ANSWER, RagSummarizeService


class StubChain:
    """替掉真实模型链：记录调用参数，返回固定文本，不发任何网络请求。"""

    def __init__(self):
        self.calls: list[dict] = []

    def invoke(self, payload: dict) -> str:
        self.calls.append(payload)
        return "STUB_ANSWER"


class RagServiceAnswerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 只构造服务对象（打开本地 Chroma），不触发灌库和检索。
        cls.service = RagSummarizeService()

    def setUp(self):
        self.stub = StubChain()
        self._real_chain = self.service.chain
        self.service.chain = self.stub

    def tearDown(self):
        self.service.chain = self._real_chain

    def test_empty_context_returns_no_evidence_answer(self):
        answer = self.service.answer("扫地机器人能除甲醛吗", [])
        self.assertEqual(answer, NO_EVIDENCE_ANSWER)
        self.assertEqual(self.stub.calls, [])

    def test_out_of_scope_wins_over_no_evidence(self):
        answer = self.service.answer("帮我写一段 Python 爬虫代码", [])
        self.assertEqual(answer, OUT_OF_SCOPE_ANSWER)
        self.assertEqual(self.stub.calls, [])

    def test_with_context_still_calls_model(self):
        docs = [Document(page_content="滤网 1–3 个月更换", metadata={"source": "维护保养.txt"})]
        answer = self.service.answer("滤网多久换一次", docs)
        self.assertEqual(answer, "STUB_ANSWER")
        self.assertEqual(len(self.stub.calls), 1)
        self.assertIn("滤网 1–3 个月更换", self.stub.calls[0]["context"])
        self.assertIn("滤网多久换一次", self.stub.calls[0]["input"])

    def test_no_evidence_answer_trips_refusal_cue(self):
        from eval.judge import refusal_cue

        self.assertTrue(refusal_cue(NO_EVIDENCE_ANSWER))


class RetrievalConfigWiringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = RagSummarizeService()

    def test_rerank_config_is_loaded(self):
        pipeline = self.service.pipeline
        self.assertIsNotNone(pipeline.reranker)
        self.assertEqual(pipeline.reranker.model, "gte-rerank-v2")
        # 阈值来自标定（可分区间 0.1578~0.1673）；配置没接上时会退回默认 0.0。
        self.assertGreater(pipeline.min_score, 0.0)
        self.assertLess(pipeline.min_score, 0.2)
        self.assertEqual(pipeline.mode, "hybrid")
        self.assertGreaterEqual(pipeline.vector_candidates, 1)
        self.assertGreaterEqual(pipeline.bm25_candidates, 1)


if __name__ == "__main__":
    unittest.main()
