import tempfile
import unittest
from pathlib import Path

from observability.store import TraceStore
from server.chat_runner import ChatRunner


class FakeAgent:
    def __init__(self, events=None, error=None):
        self.events = events or []
        self.error = error
        self.seen_account = None

    def iter_chat(self, query, thread_id="default", token_account=None, owner_user_id=None):
        self.seen_account = token_account
        if token_account is not None:
            token_account.add(
                {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                    "cached_tokens": 0,
                    "request_id": "req-1",
                }
            )
        for event in self.events:
            yield event
        if self.error is not None:
            raise self.error


def _events():
    return [
        {"type": "route", "route": "qa"},
        {
            "type": "tool",
            "name": "rag_summarize",
            "status": "start",
            "id": "t1",
            "args": {"query": "滤网"},
        },
        {
            "type": "tool",
            "name": "rag_summarize",
            "status": "end",
            "id": "t1",
            "args": {"query": "滤网"},
            "ok": True,
            "duration_ms": 12.5,
            "empty": False,
        },
        {"type": "delta", "text": "先拆"},
        {"type": "delta", "text": "滤网。"},
        {"type": "result", "route": "qa", "answer": "先拆滤网。"},
    ]


class ChatRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = TraceStore(str(Path(self.tmp.name) / "obs.sqlite3"))
        self.runner = ChatRunner(
            FakeAgent(_events()), store=self.store, model_name="qwen-max"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_event_sequence_and_trace(self):
        events = list(self.runner.stream("滤网怎么拆", thread_id="th-1"))

        self.assertEqual(
            [e["event"] for e in events],
            ["meta", "tool", "tool", "delta", "delta", "done"],
        )
        self.assertEqual(events[0]["data"]["route"], "qa")
        self.assertEqual(events[1]["data"]["status"], "start")
        self.assertEqual(events[2]["data"]["status"], "end")
        self.assertTrue(events[2]["data"]["ok"])
        self.assertEqual(events[-1]["data"]["answer"], "先拆滤网。")

        done = events[-1]["data"]
        self.assertEqual(done["usage"]["total_tokens"], 120)
        self.assertIsNotNone(done["cost_cny"])
        self.assertEqual(done["tool_calls"], 1)

        trace = self.store.get_trace(done["trace_id"])
        self.assertIsNotNone(trace)
        self.assertEqual(trace["route"], "qa")
        self.assertEqual(trace["answer"], "先拆滤网。")
        self.assertEqual(trace["status"], "ok")
        self.assertEqual(trace["total_tokens"], 120)
        self.assertEqual(trace["usage_source"], "provider")
        self.assertEqual(trace["tool_calls"], 1)
        self.assertEqual(
            [span["kind"] for span in trace["spans"]], ["route", "tool"]
        )

    def test_error_yields_error_and_records_status(self):
        runner = ChatRunner(
            FakeAgent(_events()[:1], error=RuntimeError("boom")),
            store=self.store,
            model_name="qwen-max",
        )

        events = list(runner.stream("你好"))

        self.assertEqual([e["event"] for e in events], ["meta", "error"])
        self.assertEqual(events[-1]["data"]["code"], "internal_error")
        self.assertNotIn("boom", events[-1]["data"]["message"])

        trace_id = events[0]["data"]["trace_id"]
        trace = self.store.get_trace(trace_id)
        self.assertEqual(trace["status"], "error")
        self.assertIn("boom", trace["error"])

    def test_cancel_records_cancelled_status(self):
        runner = ChatRunner(
            FakeAgent(_events()[:4]), store=self.store, model_name="qwen-max"
        )
        generator = runner.stream("滤网怎么拆")
        meta = next(generator)
        tool_start = next(generator)
        self.assertEqual(meta["event"], "meta")
        self.assertEqual(tool_start["event"], "tool")

        generator.close()

        trace = self.store.get_trace(meta["data"]["trace_id"])
        self.assertEqual(trace["status"], "cancelled")

    def test_missing_usage_is_not_fabricated(self):
        class NoUsageAgent(FakeAgent):
            def iter_chat(self, query, thread_id="default", token_account=None, owner_user_id=None):
                for event in self.events:
                    yield event

        runner = ChatRunner(NoUsageAgent(_events()), store=self.store)
        events = list(runner.stream("你好"))

        done = events[-1]["data"]
        self.assertEqual(done["usage"]["source"], "missing")
        self.assertIsNone(done["cost_cny"])
        trace = self.store.get_trace(done["trace_id"])
        self.assertEqual(trace["usage_source"], "missing")
        self.assertIsNone(trace["total_tokens"])
        self.assertIsNone(trace["cost_cny"])

    def test_out_of_scope_usage_is_none_not_missing(self):
        class NoUsageAgent(FakeAgent):
            def iter_chat(self, query, thread_id="default", token_account=None, owner_user_id=None):
                for event in self.events:
                    yield event

        events = [
            {"type": "route", "route": "out_of_scope"},
            {"type": "delta", "text": "超出范围。"},
            {"type": "result", "route": "out_of_scope", "answer": "超出范围。"},
        ]
        runner = ChatRunner(NoUsageAgent(events), store=self.store)

        done = list(runner.stream("写代码"))[-1]["data"]

        self.assertEqual(done["usage"]["source"], "none")
        self.assertIsNone(done["cost_cny"])
        trace = self.store.get_trace(done["trace_id"])
        self.assertEqual(trace["usage_source"], "none")
        self.assertEqual(trace["total_tokens"], None)


if __name__ == "__main__":
    unittest.main()
