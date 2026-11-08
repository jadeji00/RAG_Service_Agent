import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from observability.pricing import estimate_cost
from observability.store import TraceStore
from observability.usage import TokenAccount, extract_usage, normalize_usage


@dataclass
class FakeMessage:
    response_metadata: dict
    usage_metadata: dict | None = None


class UsageTest(unittest.TestCase):
    def test_normalize_dashscope_fields(self):
        usage = normalize_usage(
            {
                "input_tokens": 14,
                "output_tokens": 1,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 8},
            }
        )
        self.assertEqual(
            usage,
            {
                "input_tokens": 14,
                "output_tokens": 1,
                "total_tokens": 15,
                "cached_tokens": 8,
            },
        )

    def test_extract_usage_from_response_metadata(self):
        message = FakeMessage(
            response_metadata={
                "request_id": "req-1",
                "model_name": "qwen-max",
                "token_usage": {
                    "input_tokens": 14,
                    "output_tokens": 1,
                    "total_tokens": 15,
                },
            }
        )
        usage = extract_usage(message)
        self.assertEqual(usage["request_id"], "req-1")
        self.assertEqual(usage["input_tokens"], 14)
        self.assertEqual(usage["model"], "qwen-max")

    def test_token_account_deduplicates_request_id(self):
        account = TokenAccount()
        usage = {
            "request_id": "req-1",
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "cached_tokens": 0,
        }
        self.assertTrue(account.add(usage))
        self.assertFalse(account.add(usage))
        account.add(
            {
                "request_id": "req-2",
                "input_tokens": 5,
                "output_tokens": 3,
                "total_tokens": 8,
                "cached_tokens": 0,
            }
        )
        self.assertEqual(account.calls, 2)
        self.assertEqual(account.input_tokens, 15)
        self.assertEqual(account.output_tokens, 5)
        self.assertEqual(account.total_tokens, 20)
        self.assertEqual(account.source, "provider")
        self.assertNotIn("_request_ids", account.as_dict())

    def test_cost_uses_configured_price(self):
        cost = estimate_cost("qwen-max", 1000, 1000)
        self.assertAlmostEqual(cost, 0.012, places=6)
        self.assertIsNone(estimate_cost("unknown-model", 1000, 1000))
        self.assertIsNone(estimate_cost("qwen-max", None, None))


class TraceStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = TraceStore(str(Path(self.tmp.name) / "obs.sqlite3"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_trace_span_and_usage_summary(self):
        self.store.start_trace(
            "t1", "thread-1", "拖布多久清洗一次？", "qwen-max", route="qa"
        )
        self.store.add_span(
            "t1",
            "tool:rag_summarize",
            "tool",
            duration_ms=123.4,
            detail={"query": "拖布清洗"},
        )
        self.store.finish_trace(
            "t1",
            answer="建议每 1-2 周清洗一次。",
            latency_ms=1500,
            first_token_ms=520,
            llm_ms=900,
            input_tokens=120,
            output_tokens=30,
            total_tokens=150,
            cost_cny=0.000576,
            usage_source="provider",
            tool_calls=1,
        )

        trace = self.store.get_trace("t1")
        self.assertEqual(trace["route"], "qa")
        self.assertEqual(trace["status"], "ok")
        self.assertEqual(len(trace["spans"]), 1)
        self.assertEqual(trace["spans"][0]["detail"]["query"], "拖布清洗")

        recent = self.store.list_traces(limit=5)
        self.assertEqual(recent[0]["trace_id"], "t1")
        self.assertIn("拖布", recent[0]["question"])

        summary = self.store.usage_summary(days=7)
        self.assertEqual(summary["totals"]["requests"], 1)
        self.assertEqual(summary["totals"]["total_tokens"], 150)
        self.assertEqual(summary["by_route"][0]["route"], "qa")
        self.assertEqual(summary["by_route"][0]["requests"], 1)
        self.assertEqual(len(summary["by_day"]), 1)

    def test_finish_trace_rejects_unknown_field(self):
        self.store.start_trace("t2", "thread-2", "问题", "qwen-max")
        with self.assertRaises(ValueError):
            self.store.finish_trace("t2", not_a_column=1)

    def test_missing_trace_returns_none(self):
        self.assertIsNone(self.store.get_trace("nope"))


if __name__ == "__main__":
    unittest.main()
