"""会话历史压缩的离线测试：不调模型，摘要器用假的。

覆盖三层——折叠边界规则、对话稿构造、滚动摘要缓存（复用 / 从原文重写 / 失败降级），
外加中间件与统计聚合的接线。
"""

import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.history_compressor import (
    SUMMARY_STREAM_TAG,
    HistoryCompressor,
    build_transcript,
    find_unsourced_identifiers,
    merge_summary_into_system,
    plan_fold,
    summarize_history_stats,
)
from agent.tools.middleware import compress_history
from utils.config_handler import agent_conf

_CONF = {
    "enabled": True,
    "trigger_messages": 8,
    "keep_messages": 4,
    "max_summary_chars": 200,
    "max_source_chars": 40,
    "max_transcript_chars": 1000,
    "cache_threads": 4,
    "summarizer_model": "qwen-flash",
}


class FakeSummarizer:
    """记录每次调用的入参，返回固定摘要。

    默认摘要只用 ``_history()`` 里真实出现过的词，否则会被编造校验拦下。
    """

    def __init__(self, reply: str = "用户问了拖布怎么洗，客服答清水冲洗后晾干。"):
        self.reply = reply
        self.calls: list[str] = []
        self.configs: list[dict] = []

    def invoke(self, messages, config=None):
        body = messages[-1].content
        self.calls.append(body)
        self.configs.append(config or {})
        return AIMessage(
            content=self.reply,
            response_metadata={
                "token_usage": {
                    "input_tokens": 120,
                    "output_tokens": 30,
                    "total_tokens": 150,
                }
            },
        )


class ExplodingSummarizer:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages, config=None):
        self.calls += 1
        raise RuntimeError("dashscope down")


def _turn(index: int, question: str = "拖布怎么洗", answer: str = "清水冲洗后晾干"):
    return [HumanMessage(content=f"{question}{index}"), AIMessage(content=f"{answer}{index}")]


def _history(turns: int, tail=None):
    messages = []
    for index in range(turns):
        messages.extend(_turn(index))
    return messages + list(tail or [])


class PlanFoldTest(unittest.TestCase):
    def test_below_trigger_returns_none(self):
        plan = plan_fold(_history(2), trigger=8, keep=4)
        self.assertIsNone(plan)

    def test_boundary_keeps_last_messages_and_current_turn(self):
        messages = _history(6)  # 12 条，trigger=8 keep=4
        plan = plan_fold(messages, trigger=8, keep=4)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.before, 12)
        # 折叠边界必须落在最后一条真实用户消息之前，当前轮完整保留
        self.assertLessEqual(plan.boundary, 10)
        self.assertIsInstance(messages[plan.boundary], HumanMessage)
        # 视图里只剩保留段原文；摘要另走 system 消息，不占对话条数
        self.assertEqual(plan.after, 12 - plan.boundary)

    def test_current_turn_with_tools_is_never_folded(self):
        messages = _history(5) + [
            HumanMessage(content="现在给我查一下"),
            AIMessage(content="", tool_calls=[{"name": "rag_summarize", "args": {}, "id": "c1"}]),
            ToolMessage(content="检索结果", tool_call_id="c1", name="rag_summarize"),
        ]
        plan = plan_fold(messages, trigger=8, keep=4)

        self.assertIsNotNone(plan)
        kept = messages[plan.boundary :]
        self.assertNotIsInstance(kept[0], ToolMessage)
        # 当前轮的用户消息、工具调用、工具结果都还在原文里
        self.assertIn(messages[-3], kept)
        self.assertIn(messages[-1], kept)

    def test_kept_segment_never_starts_with_orphan_tool_message(self):
        # keep 卡在工具组中间时，边界要往后推到工具组结束
        messages = _history(4) + [
            HumanMessage(content="查一下错误码"),
            AIMessage(content="", tool_calls=[{"name": "rag_summarize", "args": {}, "id": "c1"}]),
            ToolMessage(content="E05 是拖布未安装", tool_call_id="c1", name="rag_summarize"),
            AIMessage(content="E05 表示拖布未安装到位"),
        ]
        plan = plan_fold(messages, trigger=8, keep=2)

        self.assertIsNotNone(plan)
        self.assertNotIsInstance(messages[plan.boundary], ToolMessage)

    def test_single_turn_has_nothing_to_fold(self):
        messages = [HumanMessage(content="你好"), AIMessage(content="你好，有什么可以帮你")]
        self.assertIsNone(plan_fold(messages, trigger=1, keep=1))

    def test_guardrail_message_is_not_treated_as_current_turn(self):
        messages = _history(5) + [
            HumanMessage(content="护栏提示", additional_kwargs={"guardrail": True}),
        ]
        plan = plan_fold(messages, trigger=8, keep=4)

        self.assertIsNotNone(plan)
        # 护栏注入的消息被折进摘要区，边界由最后一条真实用户消息决定
        self.assertLess(plan.boundary, len(messages) - 1)


class BuildTranscriptTest(unittest.TestCase):
    def test_labels_roles_and_keeps_tool_names_only(self):
        messages = [
            HumanMessage(content="E05 是什么"),
            AIMessage(content="", tool_calls=[{"name": "rag_summarize", "args": {}, "id": "c1"}]),
            ToolMessage(content="超长检索正文" * 50, tool_call_id="c1", name="rag_summarize"),
            AIMessage(content="E05 是拖布未安装"),
        ]
        transcript = build_transcript(messages, max_source_chars=40)

        self.assertIn("用户：E05 是什么", transcript)
        self.assertIn("[调用工具] rag_summarize", transcript)
        self.assertIn("[工具结果] rag_summarize", transcript)
        self.assertIn("客服：E05 是拖布未安装", transcript)
        # 工具正文不进稿，否则摘要模型又要读一遍最贵的内容
        self.assertNotIn("超长检索正文", transcript)

    def test_long_message_is_truncated(self):
        messages = [HumanMessage(content="啊" * 200)]
        transcript = build_transcript(messages, max_source_chars=40)
        self.assertTrue(transcript.endswith("…"))
        self.assertLessEqual(len(transcript), len("用户：") + 41)

    def test_guardrail_messages_are_skipped(self):
        messages = [
            HumanMessage(content="护栏提示", additional_kwargs={"guardrail": True}),
            HumanMessage(content="真实问题"),
        ]
        transcript = build_transcript(messages, max_source_chars=40)

        self.assertEqual(transcript, "用户：真实问题")

    def test_empty_content_lines_are_dropped(self):
        messages = [AIMessage(content=""), HumanMessage(content="  ")]
        self.assertEqual(build_transcript(messages, max_source_chars=40), "")


class CompressorViewTest(unittest.TestCase):
    def setUp(self):
        self.summarizer = FakeSummarizer()
        self.compressor = HistoryCompressor(conf=_CONF, summarizer=self.summarizer)

    def test_not_triggered_below_threshold(self):
        messages = _history(2)
        view, stats = self.compressor.view("t1", messages)

        self.assertEqual(view, messages)
        self.assertFalse(stats["triggered"])
        self.assertEqual(stats["reason"], "below_trigger")
        self.assertEqual(self.summarizer.calls, [])

    def test_disabled_config_returns_messages_untouched(self):
        compressor = HistoryCompressor(
            conf={**_CONF, "enabled": False}, summarizer=self.summarizer
        )
        messages = _history(10)
        view, stats = compressor.view("t1", messages)

        self.assertEqual(len(view), len(messages))
        self.assertEqual(stats["reason"], "disabled")
        self.assertEqual(self.summarizer.calls, [])

    def test_view_is_recent_messages_and_summary_goes_to_stats(self):
        messages = _history(6)
        view, stats = self.compressor.view("t1", messages)

        self.assertTrue(stats["triggered"])
        self.assertFalse(stats["fallback"])
        self.assertEqual(len(view), stats["after_messages"])
        self.assertLess(len(view), len(messages))
        # 摘要不混进对话消息，交给调用方并进 system 消息
        self.assertNotIn("【历史摘要】", "".join(str(m.content) for m in view))
        self.assertEqual(stats["summary_text"], self.summarizer.reply)
        # 保留段就是原消息的尾巴，对象本身没被改写
        self.assertEqual(view, messages[stats["folded_messages"] :])
        self.assertEqual(stats["kept_messages"], len(messages) - stats["folded_messages"])
        self.assertEqual(stats["kept_messages"], len(view))

    def test_usage_from_summarizer_flows_into_sink(self):
        seen = []
        _, stats = self.compressor.view("t1", _history(6), usage_sink=seen.append)

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["input_tokens"], 120)
        # 摘要开销单列，才能和「省下来的 input token」分开看
        self.assertEqual(stats["summary_input_tokens"], 120)
        self.assertEqual(stats["summary_output_tokens"], 30)
        self.assertEqual(stats["summarizer_model"], "qwen-flash")

    def test_summarizer_call_is_tagged_so_its_tokens_can_be_filtered(self):
        """摘要跑在 model 节点里，不打 tag 就会被当成回答增量推给前端。"""
        self.compressor.view("t1", _history(6))

        self.assertEqual(len(self.summarizer.configs), 1)
        self.assertIn(SUMMARY_STREAM_TAG, self.summarizer.configs[0].get("tags", []))

    def test_reused_cache_costs_no_summary_tokens(self):
        self.compressor.view("t1", _history(6))
        _, stats = self.compressor.view("t1", _history(6))

        self.assertTrue(stats["reused"])
        self.assertEqual(stats["summary_input_tokens"], 0)
        self.assertEqual(stats["summary_output_tokens"], 0)

    def test_failed_summary_still_reports_tokens_spent(self):
        class PartialSummarizer:
            """先用掉一次额度再炸，验证失败路径也记账。"""

            def __init__(self):
                self.calls = 0

            def invoke(self, messages, config=None):
                self.calls += 1
                if self.calls == 1:
                    return AIMessage(
                        content="",
                        response_metadata={
                            "token_usage": {"input_tokens": 90, "output_tokens": 0}
                        },
                    )
                raise RuntimeError("boom")

        compressor = HistoryCompressor(conf=_CONF, summarizer=PartialSummarizer())
        _, stats = compressor.view("t1", _history(6))

        self.assertEqual(stats["reason"], "empty_summary")
        self.assertEqual(stats["summary_input_tokens"], 90)

    def test_unchanged_fold_region_reuses_cache_without_second_call(self):
        messages = _history(6)
        self.compressor.view("t1", messages)
        view, stats = self.compressor.view("t1", messages)

        self.assertEqual(len(self.summarizer.calls), 1)
        self.assertTrue(stats["reused"])
        self.assertEqual(stats["summary_text"], self.summarizer.reply)
        self.assertEqual(len(view), stats["after_messages"])

    def test_longer_history_is_resummarized_from_source_not_chained(self):
        self.compressor.view("t1", _history(6))
        view, stats = self.compressor.view("t1", _history(8))

        self.assertEqual(len(self.summarizer.calls), 2)
        self.assertFalse(stats["reused"])
        second = self.summarizer.calls[1]
        # 增量合并会让弱模型编造字段，所以第二次也必须拿到最早的原文
        self.assertNotIn("【已有摘要】", second)
        self.assertNotIn("【新增对话】", second)
        self.assertIn("拖布怎么洗0", second)
        self.assertIn("拖布怎么洗5", second)
        self.assertEqual(len(view), stats["after_messages"])
        self.assertFalse(stats["transcript_truncated"])

    def test_different_threads_do_not_share_summary(self):
        self.compressor.view("t1", _history(6))
        self.compressor.view("t2", _history(6))
        self.assertEqual(len(self.summarizer.calls), 2)

    def test_cache_evicts_oldest_thread(self):
        for index in range(_CONF["cache_threads"] + 2):
            self.compressor.view(f"t{index}", _history(6))
        self.assertLessEqual(len(self.compressor._cache), _CONF["cache_threads"])

    def test_summarizer_error_falls_back_to_full_history(self):
        compressor = HistoryCompressor(conf=_CONF, summarizer=ExplodingSummarizer())
        messages = _history(6)
        view, stats = compressor.view("t1", messages)

        self.assertEqual(len(view), len(messages))
        self.assertTrue(stats["fallback"])
        self.assertEqual(stats["reason"], "summarizer_error")
        # 触发过但降级了，before/after 仍然是原样
        self.assertEqual(stats["before_messages"], len(messages))

    def test_empty_summary_falls_back(self):
        compressor = HistoryCompressor(conf=_CONF, summarizer=FakeSummarizer(reply="   "))
        messages = _history(6)
        view, stats = compressor.view("t1", messages)

        self.assertEqual(len(view), len(messages))
        self.assertEqual(stats["reason"], "empty_summary")

    def test_empty_transcript_falls_back_without_calling_summarizer(self):
        # 全是护栏消息和空内容：能折但没东西可写
        messages = [HumanMessage(content="护栏", additional_kwargs={"guardrail": True})] * 9 + [
            HumanMessage(content="当前问题")
        ]
        view, stats = self.compressor.view("t1", messages)

        self.assertEqual(len(view), len(messages))
        self.assertEqual(stats["reason"], "empty_transcript")
        self.assertEqual(self.summarizer.calls, [])

    def test_oversized_transcript_only_sends_recent_part(self):
        compressor = HistoryCompressor(
            conf={**_CONF, "max_transcript_chars": 60, "max_source_chars": 40},
            summarizer=self.summarizer,
        )
        _, stats = compressor.view("t1", _history(8))

        payload = self.summarizer.calls[0]
        self.assertLessEqual(len(payload), 60)
        self.assertTrue(stats["transcript_truncated"])
        # 截断只影响送出去的稿子，不影响完整对话稿
        self.assertNotIn("拖布怎么洗0", payload)

    def test_truncation_does_not_make_early_facts_look_unsourced(self):
        class EchoSummarizer:
            """故意在摘要里写一个来自被截断头部的标识符。"""

            def invoke(self, messages, config=None):
                return AIMessage(content="用户问过 E05 错误码。")

        compressor = HistoryCompressor(
            conf={**_CONF, "max_transcript_chars": 30, "max_source_chars": 40},
            summarizer=EchoSummarizer(),
        )
        messages = [HumanMessage(content="E05 是什么错误码")] + _history(8)
        _, stats = compressor.view("t1", messages)

        self.assertTrue(stats["transcript_truncated"])
        # 校验用的是完整对话稿，截断不会把真事实误判成编造
        self.assertFalse(stats["fallback"])

    def test_config_is_read_from_agent_conf(self):
        compressor = HistoryCompressor()
        self.assertTrue(compressor.enabled)
        self.assertGreater(compressor.trigger_messages, compressor.keep_messages)
        # 摘要必须走轻量模型，否则每轮首字延迟会被摘要拖住
        self.assertEqual(compressor.summarizer_model, "qwen-flash")
        # 长度上限要跟 config/agent.yml 一致：240 字实测会丢掉第一轮问的 E05
        self.assertEqual(compressor.max_summary_chars, agent_conf["history"]["max_summary_chars"])


class UnsourcedIdentifierTest(unittest.TestCase):
    """编造校验：摘要里的标识符必须能在对话稿里找到来源。"""

    TRANSCRIPT = "用户：E05 是什么错误码\n客服：E05 表示拖布未安装，请检查支架。"

    def test_fabricated_identifiers_are_reported(self):
        summary = "用户 ID 是 U123456，型号 XG900，错误码 E203，日期 2024-04-05。"
        unsourced = find_unsourced_identifiers(summary, self.TRANSCRIPT)

        self.assertIn("U123456", unsourced)
        self.assertIn("XG900", unsourced)
        self.assertIn("E203", unsourced)
        self.assertIn("2024-04-05", unsourced)

    def test_sourced_identifiers_pass(self):
        summary = "用户问 E05，客服答 E05 是拖布未安装。"
        self.assertEqual(find_unsourced_identifiers(summary, self.TRANSCRIPT), [])

    def test_pure_chinese_numbers_are_not_flagged(self):
        # 「3-6 个月」「2 年」这类没有字母的数字不能误杀，否则正常摘要全被拒
        summary = "边刷建议 3-6 个月更换一次，质保 2 年。"
        self.assertEqual(find_unsourced_identifiers(summary, self.TRANSCRIPT), [])

    def test_case_and_punctuation_are_normalized(self):
        summary = "错误码是 e05。"
        self.assertEqual(find_unsourced_identifiers(summary, self.TRANSCRIPT), [])

    def test_duplicates_are_reported_once(self):
        summary = "XG900 和 XG900 都出现了。"
        self.assertEqual(find_unsourced_identifiers(summary, self.TRANSCRIPT), ["XG900"])

    def test_empty_inputs_are_safe(self):
        self.assertEqual(find_unsourced_identifiers("", self.TRANSCRIPT), [])
        self.assertEqual(find_unsourced_identifiers("XG900", ""), [])

    def test_view_falls_back_when_summary_fabricates(self):
        class FabricatingSummarizer:
            def invoke(self, messages, config=None):
                return AIMessage(
                    content="用户 ID 是 U123456，型号 XG900。",
                    response_metadata={
                        "token_usage": {"input_tokens": 100, "output_tokens": 20}
                    },
                )

        compressor = HistoryCompressor(conf=_CONF, summarizer=FabricatingSummarizer())
        messages = _history(6)
        view, stats = compressor.view("t1", messages)

        self.assertEqual(len(view), len(messages))
        self.assertTrue(stats["fallback"])
        self.assertEqual(stats["reason"], "unsourced_identifiers")
        self.assertIn("U123456", stats["unsourced"])
        # 摘要已经花掉的 token 仍然要记账
        self.assertEqual(stats["summary_input_tokens"], 100)

    def test_faithful_summary_is_accepted(self):
        class FaithfulSummarizer:
            def invoke(self, messages, config=None):
                return AIMessage(content="用户问了拖布怎么洗，客服答清水冲洗后晾干。")

        compressor = HistoryCompressor(conf=_CONF, summarizer=FaithfulSummarizer())
        view, stats = compressor.view("t1", _history(6))

        self.assertFalse(stats["fallback"])
        self.assertLess(len(view), len(_history(6)))


class SummarizeHistoryStatsTest(unittest.TestCase):
    def test_empty_list_marks_no_model_call(self):
        stats = summarize_history_stats([])
        self.assertEqual(stats["model_calls"], 0)
        self.assertEqual(stats["reason"], "no_model_call")

    def test_below_trigger_keeps_reason_and_max_before(self):
        stats = summarize_history_stats(
            [
                {"triggered": False, "reason": "below_trigger", "before_messages": 6},
                {"triggered": False, "reason": "below_trigger", "before_messages": 9},
            ]
        )
        self.assertFalse(stats["triggered"])
        self.assertEqual(stats["reason"], "below_trigger")
        self.assertEqual(stats["before_messages"], 9)
        self.assertEqual(stats["model_calls"], 2)

    def test_multiple_model_calls_aggregate_to_one_line(self):
        stats = summarize_history_stats(
            [
                {
                    "triggered": True,
                    "before_messages": 18,
                    "after_messages": 11,
                    "folded_messages": 8,
                    "summary_chars": 300,
                    "reused": False,
                    "duration_ms": 900.0,
                    "summary_input_tokens": 200,
                    "summary_output_tokens": 30,
                },
                {
                    "triggered": True,
                    "before_messages": 20,
                    "after_messages": 13,
                    "folded_messages": 8,
                    "summary_chars": 300,
                    "reused": True,
                    "duration_ms": 1.5,
                    "summary_input_tokens": 200,
                    "summary_output_tokens": 60,
                },
            ]
        )
        self.assertTrue(stats["triggered"])
        self.assertEqual(stats["compressed_calls"], 2)
        self.assertEqual(stats["fallback_calls"], 0)
        self.assertEqual(stats["before_messages"], 20)
        self.assertEqual(stats["after_messages"], 11)
        self.assertTrue(stats["reused"])
        self.assertEqual(stats["duration_ms"], 901.5)
        # 复用缓存那次不花钱，只有第一次的 400/90 进账
        self.assertEqual(stats["summary_input_tokens"], 400)
        self.assertEqual(stats["summary_output_tokens"], 90)

    def test_fallback_calls_are_counted(self):
        stats = summarize_history_stats(
            [
                {"triggered": True, "fallback": True, "reason": "summarizer_error",
                 "before_messages": 18, "duration_ms": 12.0},
                {"triggered": False, "reason": "below_trigger", "before_messages": 4},
            ]
        )
        self.assertFalse(stats["triggered"])
        self.assertEqual(stats["fallback_calls"], 1)
        self.assertEqual(stats["reason"], "below_trigger")

    def test_unsourced_identifiers_are_recorded_for_diagnosis(self):
        stats = summarize_history_stats(
            [
                {
                    "triggered": True,
                    "fallback": True,
                    "reason": "unsourced_identifiers",
                    "unsourced": ["XG900", "U123456"],
                    "before_messages": 37,
                }
            ]
        )
        self.assertFalse(stats["triggered"])
        self.assertEqual(stats["reason"], "unsourced_identifiers")
        self.assertEqual(stats["unsourced"], ["U123456", "XG900"])

    def test_unsourced_is_dropped_when_compression_succeeded(self):
        stats = summarize_history_stats(
            [
                {"triggered": True, "fallback": True, "reason": "unsourced_identifiers",
                 "unsourced": ["XG900"], "before_messages": 20},
                {"triggered": True, "before_messages": 22, "after_messages": 9,
                 "folded_messages": 14, "summary_chars": 150},
            ]
        )
        self.assertTrue(stats["triggered"])
        self.assertNotIn("unsourced", stats)
        self.assertEqual(stats["fallback_calls"], 1)


class MergeSummaryIntoSystemTest(unittest.TestCase):
    def test_appends_after_existing_system_message(self):
        merged = merge_summary_into_system(SystemMessage(content="你是客服。"), "- 用户问过 E05")

        self.assertIsInstance(merged, SystemMessage)
        self.assertTrue(merged.content.startswith("你是客服。"))
        self.assertIn("- 用户问过 E05", merged.content)
        self.assertIn("【较早对话的要点清单】", merged.content)

    def test_accepts_plain_string_and_none(self):
        self.assertTrue(merge_summary_into_system("纯字符串", "摘要").content.startswith("纯字符串"))
        self.assertTrue(merge_summary_into_system(None, "摘要").content.startswith("\n\n"))

    def test_tells_model_not_to_copy_the_summary_format(self):
        merged = merge_summary_into_system(SystemMessage(content="你是客服。"), "- 用户问过 E05")
        self.assertIn("不要" "在回答里复述或照搬它的格式", merged.content)


class CompressHistoryMiddlewareTest(unittest.TestCase):
    class _Request:
        def __init__(self, context, messages, system_message=None):
            self.messages = messages
            self.system_message = system_message
            self.runtime = type("R", (), {"context": context})()

        def override(self, **kwargs):
            return CompressHistoryMiddlewareTest._Request(
                self.runtime.context,
                kwargs.get("messages", self.messages),
                kwargs.get("system_message", self.system_message),
            )

    def test_no_compressor_in_context_passes_through(self):
        seen = []

        def handler(request):
            seen.append(request.messages)
            return "resp"

        result = compress_history.wrap_model_call(
            self._Request({}, _history(20)), handler
        )
        self.assertEqual(result, "resp")
        self.assertEqual(len(seen[0]), 40)

    def test_triggered_call_injects_summary_into_system_and_trims_messages(self):
        summarizer = FakeSummarizer()
        context = {
            "thread_id": "t1",
            "history_compressor": HistoryCompressor(conf=_CONF, summarizer=summarizer),
            "history_stats": [],
        }
        seen = []

        def handler(request):
            seen.append(request)
            return "resp"

        base = SystemMessage(content="你是扫地机客服。")
        compress_history.wrap_model_call(
            self._Request(context, _history(6), base), handler
        )

        stats = context["history_stats"][0]
        self.assertEqual(len(seen[0].messages), stats["after_messages"])
        # 摘要是接在 system 末尾，不是塞一条对话消息
        self.assertIn(summarizer.reply, seen[0].system_message.content)
        self.assertTrue(seen[0].system_message.content.startswith("你是扫地机客服。"))
        self.assertIn("不要" "在回答里复述或照搬它的格式", seen[0].system_message.content)
        self.assertEqual(len(seen[0].messages), stats["kept_messages"])
        self.assertEqual(len(context["history_stats"]), 1)

    def test_base_request_is_not_mutated(self):
        context = {
            "thread_id": "t1",
            "history_compressor": HistoryCompressor(conf=_CONF, summarizer=FakeSummarizer()),
        }
        messages = _history(6)
        request = self._Request(context, messages, SystemMessage(content="原始"))
        seen = []

        compress_history.wrap_model_call(
            request, lambda req: seen.append(req) or "resp"
        )

        self.assertEqual(request.messages, messages)
        self.assertEqual(request.system_message.content, "原始")
        self.assertIsNot(seen[0], request)

    def test_missing_summary_text_passes_through(self):
        """摘要被编造校验拦下时不能 override，否则会把完整历史截掉。"""
        context = {
            "thread_id": "t1",
            "history_compressor": HistoryCompressor(
                conf=_CONF, summarizer=FakeSummarizer(reply="型号是 XG900")
            ),
        }
        messages = _history(6)
        seen = []

        compress_history.wrap_model_call(
            self._Request(context, messages, SystemMessage(content="原始")),
            lambda req: seen.append(req) or "resp",
        )

        self.assertTrue(context["history_stats"][0]["fallback"])
        self.assertEqual(len(seen[0].messages), len(messages))
        self.assertEqual(seen[0].system_message.content, "原始")

    def test_stats_are_appended_per_model_call(self):
        context = {
            "thread_id": "t1",
            "history_compressor": HistoryCompressor(conf=_CONF, summarizer=FakeSummarizer()),
        }
        request = self._Request(context, _history(3))

        compress_history.wrap_model_call(request, lambda req: "r1")
        compress_history.wrap_model_call(request, lambda req: "r2")

        self.assertEqual(len(context["history_stats"]), 2)
        self.assertFalse(context["history_stats"][0]["triggered"])

    def test_summarizer_failure_still_calls_model_with_full_history(self):
        context = {
            "thread_id": "t1",
            "history_compressor": HistoryCompressor(conf=_CONF, summarizer=ExplodingSummarizer()),
        }
        messages = _history(6)
        seen = []

        def handler(request):
            seen.append(request.messages)
            return "resp"

        result = compress_history.wrap_model_call(
            self._Request(context, messages), handler
        )

        self.assertEqual(result, "resp")
        self.assertEqual(len(seen[0]), len(messages))
        self.assertTrue(context["history_stats"][0]["fallback"])

    def test_usage_sink_receives_summary_tokens(self):
        collected = []
        context = {
            "thread_id": "t1",
            "history_compressor": HistoryCompressor(conf=_CONF, summarizer=FakeSummarizer()),
            "usage_sink": collected.append,
        }
        compress_history.wrap_model_call(
            self._Request(context, _history(6)), lambda req: "resp"
        )
        self.assertEqual(collected[0]["total_tokens"], 150)


if __name__ == "__main__":
    unittest.main()
