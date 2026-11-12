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


class RagServiceCitationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = RagSummarizeService()

    def setUp(self):
        self.stub = StubChain()
        self.service.chain = self.stub

    def tearDown(self):
        # 实例属性会遮住类上的真方法，必须删掉而不是赋回，否则污染后面的用例。
        self.service.__dict__.pop("chain", None)
        self.service.__dict__.pop("retrieve_with_stats", None)

    def stub_retrieval(self, documents, stats):
        self.service.retrieve_with_stats = lambda query, k=None: (documents, stats)

    def test_returns_answer_sources_and_stats(self):
        docs = [
            Document(
                page_content="E05 悬崖或轮组异常，擦净底部传感器后重启。",
                metadata={"source": r"D:\AI_RAG_Agent\data\故障排除.txt",
                          "section": "报错处理", "unit": "E05", "rerank_score": 0.8734},
            )
        ]
        stats = {"mode": "hybrid", "fused_candidates": 12, "rerank": "ok"}
        self.stub_retrieval(docs, stats)

        answer, sources, returned = self.service.answer_with_citations("E05 怎么办")

        self.assertEqual(answer, "STUB_ANSWER")
        self.assertEqual(returned, stats)
        self.assertEqual(len(sources), 1)
        # 绝对路径只留文件名，分数原样带出去给界面显示。
        self.assertEqual(sources[0]["file"], "故障排除.txt")
        self.assertEqual(sources[0]["section"], "报错处理")
        self.assertEqual(sources[0]["score"], 0.8734)
        self.assertEqual(len(self.stub.calls), 1)

    def test_out_of_scope_returns_no_sources(self):
        # 越界拒答一个字都没用到检索结果，列出「参考资料」等于暗示它有依据。
        docs = [Document(page_content="无关资料", metadata={"source": "选购指南.txt"})]
        self.stub_retrieval(docs, {"mode": "hybrid"})

        answer, sources, _ = self.service.answer_with_citations("帮我写一段 Python 爬虫代码")

        self.assertEqual(answer, OUT_OF_SCOPE_ANSWER)
        self.assertEqual(sources, [])
        self.assertEqual(self.stub.calls, [])

    def test_no_evidence_returns_no_sources(self):
        self.stub_retrieval([], {"mode": "hybrid", "rerank": "ok", "rerank_kept": 0})

        answer, sources, _ = self.service.answer_with_citations("扫地机器人能除甲醛吗")

        self.assertEqual(answer, NO_EVIDENCE_ANSWER)
        self.assertEqual(sources, [])


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
