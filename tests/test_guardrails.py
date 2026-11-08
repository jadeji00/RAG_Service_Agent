import unittest

from agent.guardrails import (
    MAX_TOOL_CALLS,
    authorize_tool_call,
    new_guard_state,
    record_tool_result,
    validate_tool_args,
)


class ValidateArgsTest(unittest.TestCase):
    def test_fetch_args(self):
        self.assertIsNone(
            validate_tool_args(
                "fetch_external_data", {"user_id": "1001", "month": "2026-10"}
            )
        )
        self.assertIn(
            "user_id",
            validate_tool_args(
                "fetch_external_data", {"user_id": "10", "month": "2026-10"}
            ),
        )
        self.assertIn(
            "month",
            validate_tool_args(
                "fetch_external_data", {"user_id": "1001", "month": "2026-13"}
            ),
        )

    def test_rag_and_weather_args(self):
        self.assertIsNone(validate_tool_args("rag_summarize", {"query": "拖布清洗"}))
        self.assertIn("query", validate_tool_args("rag_summarize", {"query": "  "}))
        self.assertIsNone(validate_tool_args("get_weather", {"city": "厦门"}))
        self.assertIn("city", validate_tool_args("get_weather", {"city": ""}))

    def test_no_arg_tool_rejects_payload(self):
        self.assertIn(
            "不接受参数",
            validate_tool_args("get_user_id", {"unexpected": "x"}),
        )
        self.assertIsNone(validate_tool_args("get_user_id", {}))


class AuthorizeToolCallTest(unittest.TestCase):
    def test_duplicate_success_is_blocked(self):
        guard = new_guard_state()
        args = {"user_id": "1001", "month": "2026-10"}
        allowed, _, key = authorize_tool_call("fetch_external_data", args, guard)
        self.assertTrue(allowed)
        record_tool_result(guard, key, True)

        allowed, reason, _ = authorize_tool_call(
            "fetch_external_data", dict(args), guard
        )
        self.assertFalse(allowed)
        self.assertIn("重复", reason)
        self.assertEqual(guard["blocked"], 1)

    def test_one_retry_after_error(self):
        guard = new_guard_state()
        args = {"query": "拖布清洗"}
        allowed, _, key = authorize_tool_call("rag_summarize", args, guard)
        self.assertTrue(allowed)
        record_tool_result(guard, key, False)

        allowed, _, key = authorize_tool_call("rag_summarize", args, guard)
        self.assertTrue(allowed)
        record_tool_result(guard, key, False)

        allowed, reason, _ = authorize_tool_call("rag_summarize", args, guard)
        self.assertFalse(allowed)
        self.assertIn("失败多次", reason)

    def test_budget_blocks_after_limit(self):
        guard = new_guard_state()
        for index in range(MAX_TOOL_CALLS):
            allowed, _, _ = authorize_tool_call(
                "rag_summarize", {"query": f"问题{index}"}, guard
            )
            self.assertTrue(allowed)

        allowed, reason, _ = authorize_tool_call(
            "rag_summarize", {"query": "再来一次"}, guard
        )
        self.assertFalse(allowed)
        self.assertIn("上限", reason)
        self.assertEqual(guard["executed"], MAX_TOOL_CALLS)

    def test_canonical_args_ignore_key_order(self):
        guard = new_guard_state()
        first = {"user_id": "1001", "month": "2026-10"}
        second = {"month": "2026-10", "user_id": "1001"}
        allowed, _, key = authorize_tool_call("fetch_external_data", first, guard)
        self.assertTrue(allowed)
        record_tool_result(guard, key, True)
        allowed, _, _ = authorize_tool_call("fetch_external_data", second, guard)
        self.assertFalse(allowed)


if __name__ == "__main__":
    unittest.main()
