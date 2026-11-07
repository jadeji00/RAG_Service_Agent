import unittest
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from agent.tools.middleware import monitor_tool, should_switch_report


def _request(tool_name: str, context: dict):
    return SimpleNamespace(
        tool_call={"name": tool_name, "args": {}, "id": "call-1"},
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
            _request("fetch_external_data", context), _ok_handler
        )
        self.assertTrue(context["report"])

    def test_rag_does_not_switch_report_context(self):
        context = {"report": False}
        monitor_tool.wrap_tool_call(_request("rag_summarize", context), _ok_handler)
        self.assertFalse(context["report"])

    def test_failed_fetch_does_not_switch_report_context(self):
        context = {"report": False}

        def failing_handler(request):
            raise RuntimeError("csv missing")

        monitor_tool.wrap_tool_call(
            _request("fetch_external_data", context), failing_handler
        )
        self.assertFalse(context["report"])

    def test_should_switch_report_tool_set(self):
        self.assertTrue(should_switch_report("fetch_external_data"))
        self.assertFalse(should_switch_report("fill_context_for_report"))
        self.assertFalse(should_switch_report("get_user_id"))


if __name__ == "__main__":
    unittest.main()
