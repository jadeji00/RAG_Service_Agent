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
