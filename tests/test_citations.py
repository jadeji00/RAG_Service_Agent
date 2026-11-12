import unittest
from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.react_agent import ReactAgent
from rag.citations import (
    MAX_SOURCES,
    PREVIEW_CHARS,
    build_sources,
    merge_sources,
    summarize_retrieval_stats,
)


def make_doc(source: str, text: str, **extra) -> Document:
    return Document(page_content=text, metadata={"source": source, **extra})


class BuildSourcesTest(unittest.TestCase):
    def test_numbers_follow_retrieval_order(self):
        docs = [
            make_doc("故障排除.txt", "E05 处理", section="报错", unit="E05", rerank_score=0.91),
            make_doc("100问.txt", "拖布清洗", section="耗材", unit="拖布", rerank_score=0.42),
        ]
        sources = build_sources(docs)
        self.assertEqual([item["n"] for item in sources], [1, 2])
        self.assertEqual(sources[0]["file"], "故障排除.txt")
        self.assertEqual(sources[0]["section"], "报错")
        self.assertEqual(sources[0]["unit"], "E05")
        self.assertEqual(sources[0]["score"], 0.91)
        self.assertEqual(sources[0]["preview"], "E05 处理")

    def test_absolute_path_is_reduced_to_file_name(self):
        # source 元数据是服务器上的绝对路径，整条推到浏览器会泄露目录结构。
        windows = build_sources([make_doc(r"D:\AI_RAG_Agent\data\故障排除.txt", "E05")])
        posix = build_sources([make_doc("/srv/data/故障排除.txt", "E05")])
        self.assertEqual(windows[0]["file"], "故障排除.txt")
        self.assertEqual(posix[0]["file"], "故障排除.txt")

    def test_missing_rerank_score_stays_none(self):
        # 重排关闭或降级时没有分数，必须是 None 而不是 0——0 会被读成「不相关」。
        sources = build_sources([make_doc("故障排除.txt", "E05")])
        self.assertIsNone(sources[0]["score"])

    def test_preview_collapses_whitespace_and_truncates(self):
        text = "  E05\n\n  悬崖或轮组异常  " + "长" * 200
        sources = build_sources([make_doc("故障排除.txt", text)])
        self.assertEqual(len(sources[0]["preview"]), PREVIEW_CHARS)
        self.assertTrue(sources[0]["preview"].startswith("E05 悬崖或轮组异常"))
        self.assertNotIn("\n", sources[0]["preview"])

    def test_missing_metadata_fields_become_empty_strings(self):
        sources = build_sources([Document(page_content="正文", metadata={})])
        self.assertEqual(sources[0]["file"], "")
        self.assertEqual(sources[0]["section"], "")
        self.assertEqual(sources[0]["unit"], "")

    def test_empty_documents(self):
        self.assertEqual(build_sources([]), [])


class MergeSourcesTest(unittest.TestCase):
    def test_dedupes_same_chunk_across_calls_and_renumbers(self):
        first = [{"n": 1, "file": "a.txt", "section": "s", "unit": "u", "preview": "p", "score": 0.3}]
        second = [{"n": 1, "file": "a.txt", "section": "s", "unit": "u", "preview": "p", "score": 0.7}]
        merged = merge_sources([first, second])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["n"], 1)
        # 两次检索的 query 不同，同一段资料的相关度取较高的那次。
        self.assertEqual(merged[0]["score"], 0.7)

    def test_keeps_order_of_first_appearance(self):
        groups = [
            [
                {"n": 1, "file": "a.txt", "preview": "a", "score": 0.2},
                {"n": 2, "file": "b.txt", "preview": "b", "score": 0.9},
            ],
            [{"n": 1, "file": "c.txt", "preview": "c", "score": 0.5}],
        ]
        self.assertEqual([item["file"] for item in merge_sources(groups)], ["a.txt", "b.txt", "c.txt"])
        self.assertEqual([item["n"] for item in merge_sources(groups)], [1, 2, 3])

    def test_real_score_replaces_missing_one(self):
        groups = [
            [{"file": "a.txt", "preview": "a", "score": None}],
            [{"file": "a.txt", "preview": "a", "score": 0.4}],
        ]
        self.assertEqual(merge_sources(groups)[0]["score"], 0.4)

    def test_limit_caps_result(self):
        groups = [[{"file": f"f{i}.txt", "preview": str(i)} for i in range(20)]]
        self.assertEqual(len(merge_sources(groups)), MAX_SOURCES)

    def test_ignores_junk_entries(self):
        merged = merge_sources([None, ["not-a-dict"], [{"file": "a.txt"}], []])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["file"], "a.txt")

    def test_does_not_mutate_input(self):
        groups = [[{"n": 9, "file": "a.txt", "score": 0.1}]]
        merge_sources(groups)
        self.assertEqual(groups[0][0]["n"], 9)

    def test_empty_input(self):
        self.assertEqual(merge_sources([]), [])
        self.assertEqual(merge_sources(None), [])


class SummarizeRetrievalStatsTest(unittest.TestCase):
    def test_no_calls(self):
        self.assertEqual(summarize_retrieval_stats([]), {"calls": 0})
        self.assertEqual(summarize_retrieval_stats(None), {"calls": 0})

    def test_sums_candidates_and_latency(self):
        items = [
            {
                "mode": "hybrid",
                "index_size": 412,
                "vector_candidates": 20,
                "bm25_candidates": 20,
                "fused_candidates": 26,
                "overlap": 14,
                "rerank_kept": 3,
                "rerank_usage_tokens": 40,
                "rerank": "ok",
                "rerank_top_score": 0.61,
                "vector_latency_ms": 120.5,
                "bm25_latency_ms": 1.25,
                "rerank_latency_ms": 300.0,
            },
            {
                "mode": "hybrid",
                "index_size": 412,
                "vector_candidates": 20,
                "bm25_candidates": 20,
                "fused_candidates": 30,
                "overlap": 10,
                "rerank_kept": 3,
                "rerank_usage_tokens": 35,
                "rerank": "fallback",
                "vector_latency_ms": 100.0,
                "bm25_latency_ms": 1.0,
                "rerank_latency_ms": 0.0,
            },
        ]
        summary = summarize_retrieval_stats(items)
        self.assertEqual(summary["calls"], 2)
        self.assertEqual(summary["fused_candidates"], 56)
        self.assertEqual(summary["overlap"], 24)
        self.assertEqual(summary["vector_candidates"], 40)
        self.assertEqual(summary["rerank_usage_tokens"], 75)
        self.assertEqual(summary["vector_latency_ms"], 220.5)
        self.assertEqual(summary["bm25_latency_ms"], 2.25)
        self.assertEqual(summary["rerank_latency_ms"], 300.0)
        # 最强的那条证据代表这一轮的证据质量，所以取最大值而不是平均。
        self.assertEqual(summary["rerank_top_score"], 0.61)
        self.assertEqual(summary["rerank"], "fallback,ok")
        self.assertEqual(summary["index_size"], 412)

    def test_ignores_non_dict_items(self):
        summary = summarize_retrieval_stats([None, "junk", {"fused_candidates": 5}])
        self.assertEqual(summary["calls"], 1)
        self.assertEqual(summary["fused_candidates"], 5)


class FakeGraph:
    """只实现 get_state，用来离线驱动 ReactAgent.load_history。"""

    def __init__(self, messages):
        self._messages = messages

    def get_state(self, config):
        return SimpleNamespace(values={"messages": self._messages})


def make_agent(messages) -> ReactAgent:
    # 不走 __init__：真构造会打开 Chroma 并灌库，这里只需要 load_history 的逻辑。
    agent = object.__new__(ReactAgent)
    agent.agent = FakeGraph(messages)
    return agent


def tool_message(sources, retrieval=None, call_id="c1"):
    artifact = None
    if sources is not None:
        artifact = {"sources": sources, "retrieval": retrieval or {}}
    return ToolMessage(
        content="E05 是拖布未安装", tool_call_id=call_id, name="rag_summarize", artifact=artifact
    )


class LoadHistoryCitationTest(unittest.TestCase):
    def test_sources_attach_to_the_answer_that_used_them(self):
        citation = [{"n": 1, "file": "故障排除.txt", "preview": "E05"}]
        messages = [
            HumanMessage(content="E05 怎么办"),
            AIMessage(content="", tool_calls=[{"name": "rag_summarize", "args": {}, "id": "c1"}]),
            tool_message(citation),
            AIMessage(content="先检查拖布是否安装到位"),
        ]
        history = make_agent(messages).load_history("t1")
        self.assertEqual([item["role"] for item in history], ["user", "assistant"])
        self.assertEqual(history[1]["sources"], citation)
        self.assertNotIn("sources", history[0])

    def test_guardrail_message_does_not_drop_pending_sources(self):
        # 护栏提示是在工具调用之后、最终回答之前注入的，
        # 把它当成新一轮会顺手把这一轮的引用清掉。
        messages = [
            HumanMessage(content="E05 怎么办"),
            AIMessage(content="", tool_calls=[{"name": "rag_summarize", "args": {}, "id": "c1"}]),
            tool_message([{"n": 1, "file": "故障排除.txt"}]),
            HumanMessage(content="停止调用工具", additional_kwargs={"guardrail": True}),
            AIMessage(content="先检查拖布"),
        ]
        history = make_agent(messages).load_history("t1")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[1]["sources"][0]["file"], "故障排除.txt")

    def test_two_calls_in_one_turn_are_merged(self):
        messages = [
            HumanMessage(content="E05 和拖布"),
            AIMessage(content="", tool_calls=[{"name": "rag_summarize", "args": {}, "id": "c1"}]),
            tool_message([{"n": 1, "file": "故障排除.txt"}], call_id="c1"),
            AIMessage(content="", tool_calls=[{"name": "rag_summarize", "args": {}, "id": "c2"}]),
            tool_message([{"n": 1, "file": "100问.txt"}], call_id="c2"),
            AIMessage(content="两件事分别说明"),
        ]
        history = make_agent(messages).load_history("t1")
        self.assertEqual(
            [item["file"] for item in history[1]["sources"]], ["故障排除.txt", "100问.txt"]
        )

    def test_new_user_turn_clears_leftover_sources(self):
        # 上一轮工具跑完但模型没给出最终回答时，引用不能顺延到下一轮。
        messages = [
            HumanMessage(content="E05 怎么办"),
            AIMessage(content="", tool_calls=[{"name": "rag_summarize", "args": {}, "id": "c1"}]),
            tool_message([{"n": 1, "file": "故障排除.txt"}]),
            HumanMessage(content="换个问题：滤网多久换"),
            AIMessage(content="1–3 个月"),
        ]
        history = make_agent(messages).load_history("t1")
        self.assertEqual(len(history), 3)
        self.assertNotIn("sources", history[2])

    def test_history_without_artifacts_has_no_sources_key(self):
        messages = [
            HumanMessage(content="你好"),
            AIMessage(content="你好，我可以帮你解答扫地机器人问题"),
        ]
        history = make_agent(messages).load_history("t1")
        self.assertEqual(history, [{"role": "user", "content": "你好"},
                                   {"role": "assistant", "content": "你好，我可以帮你解答扫地机器人问题"}])

    def test_empty_artifact_sources_are_skipped(self):
        messages = [
            HumanMessage(content="能除甲醛吗"),
            AIMessage(content="", tool_calls=[{"name": "rag_summarize", "args": {}, "id": "c1"}]),
            tool_message([]),
            AIMessage(content="知识库里没有相关内容"),
        ]
        history = make_agent(messages).load_history("t1")
        self.assertNotIn("sources", history[1])


if __name__ == "__main__":
    unittest.main()
