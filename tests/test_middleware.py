import unittest
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from agent.tools.middleware import (
    guard_before_model,
    monitor_tool,
    should_switch_report,
)
from agent.guardrails import new_guard_state


def _request(tool_name: str, context: dict, args: dict | None = None):
    return SimpleNamespace(
        tool_call={"name": tool_name, "args": args or {}, "id": "call-1"},
        runtime=SimpleNamespace(context=context),
    )


def _ok_handler(request):
    return ToolMessage(
        content="ok",
        tool_call_id=request.tool_call["id"],
        name=request.tool_call["name"],
    )


class ReportContextSwitchTest(unittest.TestCase):
    def test_fetch_switches_report_context(self):
        context = {"report": False}
        monitor_tool.wrap_tool_call(
            _request(
                "fetch_external_data",
                context,
                {"user_id": "1001", "month": "2026-10"},
            ),
            _ok_handler,
        )
        self.assertTrue(context["report"])

    def test_rag_does_not_switch_report_context(self):
        context = {"report": False}
        monitor_tool.wrap_tool_call(
            _request("rag_summarize", context, {"query": "拖布清洗"}),
            _ok_handler,
        )
        self.assertFalse(context["report"])

    def test_failed_fetch_does_not_switch_report_context(self):
        context = {"report": False}

        def failing_handler(request):
            raise RuntimeError("csv missing")

        monitor_tool.wrap_tool_call(
            _request(
                "fetch_external_data",
                context,
                {"user_id": "1001", "month": "2026-10"},
            ),
            failing_handler,
        )
        self.assertFalse(context["report"])

    def test_invalid_args_are_blocked_before_handler(self):
        context = {"report": False}
        called = []

        def handler(request):
            called.append(True)
            return _ok_handler(request)

        result = monitor_tool.wrap_tool_call(
            _request("fetch_external_data", context, {"user_id": "10"}),
            handler,
        )
        self.assertEqual(called, [])
        self.assertEqual(result.status, "error")
        self.assertTrue(result.additional_kwargs["guardrail"])
        self.assertEqual(context["guard"]["blocked"], 1)

    def test_should_switch_report_tool_set(self):
        self.assertTrue(should_switch_report("fetch_external_data"))
        self.assertFalse(should_switch_report("fill_context_for_report"))
        self.assertFalse(should_switch_report("get_user_id"))


def _artifact_handler(sources, retrieval):
    """造一个 content_and_artifact 风格的工具结果：正文给模型，附件给界面。"""

    def handler(request):
        return ToolMessage(
            content="E05 是拖布未安装到位",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            artifact={"sources": sources, "retrieval": retrieval},
        )

    return handler


CITATION = [{"n": 1, "file": "故障排除.txt", "section": "报错", "unit": "E05", "score": 0.83}]
RETRIEVAL = {"calls": 1, "mode": "hybrid", "fused_candidates": 6}


class CitationCaptureTest(unittest.TestCase):
    def test_artifact_sources_are_collected_into_context(self):
        context = {"report": False}
        monitor_tool.wrap_tool_call(
            _request("rag_summarize", context, {"query": "E05"}),
            _artifact_handler(CITATION, RETRIEVAL),
        )
        self.assertEqual(context["sources_raw"], [CITATION])
        self.assertEqual(context["retrieval_stats"], [RETRIEVAL])

    def test_two_calls_accumulate_in_call_order(self):
        context = {"report": False}
        second = [{"n": 1, "file": "100问.txt", "section": "耗材", "unit": "拖布", "score": 0.4}]
        monitor_tool.wrap_tool_call(
            _request("rag_summarize", context, {"query": "E05"}),
            _artifact_handler(CITATION, RETRIEVAL),
        )
        monitor_tool.wrap_tool_call(
            _request("rag_summarize", context, {"query": "拖布"}),
            _artifact_handler(second, {"calls": 1, "mode": "hybrid"}),
        )
        self.assertEqual(context["sources_raw"], [CITATION, second])
        self.assertEqual(len(context["retrieval_stats"]), 2)

    def test_tool_without_artifact_leaves_context_alone(self):
        context = {"report": False}
        monitor_tool.wrap_tool_call(
            _request("get_weather", context, {"city": "厦门"}), _ok_handler
        )
        self.assertNotIn("sources_raw", context)
        self.assertNotIn("retrieval_stats", context)

    def test_empty_sources_are_not_collected(self):
        # 无证据兜底时引用是空的，不该在 context 里留一条空记录，
        # 否则 trace 会显示「检索了 1 次、引用 0 条」之外的假调用数。
        context = {"report": False}
        monitor_tool.wrap_tool_call(
            _request("rag_summarize", context, {"query": "能除甲醛吗"}),
            _artifact_handler([], RETRIEVAL),
        )
        self.assertNotIn("sources_raw", context)
        self.assertEqual(context["retrieval_stats"], [RETRIEVAL])

    def test_failed_tool_does_not_capture(self):
        context = {"report": False}

        def failing_handler(request):
            raise RuntimeError("chroma down")

        monitor_tool.wrap_tool_call(
            _request("rag_summarize", context, {"query": "E05"}), failing_handler
        )
        self.assertNotIn("sources_raw", context)

    def test_blocked_call_does_not_capture(self):
        context = {"report": False}
        result = monitor_tool.wrap_tool_call(
            _request("rag_summarize", context, {"query": "   "}),
            _artifact_handler(CITATION, RETRIEVAL),
        )
        self.assertTrue(result.additional_kwargs["guardrail"])
        self.assertNotIn("sources_raw", context)

    def test_missing_context_still_returns_artifact(self):
        request = SimpleNamespace(
            tool_call={"name": "rag_summarize", "args": {"query": "E05"}, "id": "call-1"},
            runtime=SimpleNamespace(context=None),
        )
        result = monitor_tool.wrap_tool_call(
            request, _artifact_handler(CITATION, RETRIEVAL)
        )
        self.assertEqual(result.artifact["sources"], CITATION)


class GuardForceAnswerTest(unittest.TestCase):
    def _runtime(self, blocked: int):
        guard = new_guard_state()
        guard["blocked"] = blocked
        return SimpleNamespace(context={"guard": guard}), guard

    def test_no_injection_before_two_blocks(self):
        runtime, guard = self._runtime(1)
        result = guard_before_model.before_model({"messages": []}, runtime)
        self.assertIsNone(result)
        self.assertFalse(guard["force_injected"])

    def test_injects_once_after_two_blocks(self):
        runtime, guard = self._runtime(2)

        result = guard_before_model.before_model({"messages": []}, runtime)

        self.assertIsNotNone(result)
        self.assertEqual(len(result["messages"]), 1)
        injected = result["messages"][0]
        self.assertTrue(injected.additional_kwargs["guardrail"])
        self.assertIn("停止调用工具", injected.content)
        self.assertTrue(guard["force_injected"])

        again = guard_before_model.before_model({"messages": []}, runtime)
        self.assertIsNone(again)

    def test_no_guard_state_is_noop(self):
        runtime = SimpleNamespace(context={})
        self.assertIsNone(
            guard_before_model.before_model({"messages": []}, runtime)
        )


if __name__ == "__main__":
    unittest.main()
