import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from observability.store import TraceStore
from server.app import create_app
from server.chat_runner import ChatRunner


class FakeAgent:
    model_name = "fake-model"

    def iter_chat(self, query, thread_id="default", token_account=None):
        if token_account is not None:
            token_account.add(
                {
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "total_tokens": 60,
                    "cached_tokens": 0,
                    "request_id": "req-api",
                }
            )
        yield {"type": "route", "route": "qa"}
        yield {
            "type": "tool",
            "name": "rag_summarize",
            "status": "start",
            "id": "t1",
            "args": {"query": query},
        }
        yield {
            "type": "tool",
            "name": "rag_summarize",
            "status": "end",
            "id": "t1",
            "args": {"query": query},
            "ok": True,
            "duration_ms": 3.0,
        }
        yield {"type": "delta", "text": "已"}
        yield {"type": "delta", "text": "收到"}
        yield {"type": "result", "route": "qa", "answer": "已收到"}

    def load_history(self, thread_id):
        return [{"role": "user", "content": f"history-{thread_id}"}]


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        name = None
        data_lines = []
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data_lines.append(line[len("data: ") :])
        events.append((name, json.loads("\n".join(data_lines))))
    return events


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = TraceStore(str(Path(self.tmp.name) / "api.sqlite3"))
        self.runner = ChatRunner(FakeAgent(), store=self.store, model_name="fake-model")
        self.app = create_app(runner=self.runner, store=self.store)
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "fake-model")

    def test_chat_stream_events(self):
        response = self.client.post(
            "/chat/stream", json={"query": "滤网怎么拆", "thread_id": "api-1"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))

        events = parse_sse(response.text)
        self.assertEqual(
            [name for name, _ in events],
            ["meta", "tool", "tool", "delta", "delta", "done"],
        )
        self.assertEqual(events[0][1]["thread_id"], "api-1")
        self.assertEqual(events[-1][1]["answer"], "已收到")
        self.assertEqual(events[-1][1]["usage"]["total_tokens"], 60)

    def test_chat_validation_rejects_empty_query(self):
        response = self.client.post("/chat/stream", json={"query": ""})
        self.assertEqual(response.status_code, 422)

    def test_history_and_trace_endpoints(self):
        response = self.client.post(
            "/chat/stream", json={"query": "你好", "thread_id": "api-2"}
        )
        trace_id = parse_sse(response.text)[-1][1]["trace_id"]

        history = self.client.get("/history/api-2")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(history.json()["messages"][0]["content"], "history-api-2")

        traces = self.client.get("/traces?limit=5")
        self.assertEqual(traces.status_code, 200)
        self.assertTrue(
            any(item["trace_id"] == trace_id for item in traces.json()["traces"])
        )

        detail = self.client.get(f"/trace/{trace_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["answer"], "已收到")

        missing = self.client.get("/trace/does-not-exist")
        self.assertEqual(missing.status_code, 404)

        usage = self.client.get("/usage?days=7")
        self.assertEqual(usage.status_code, 200)
        self.assertGreaterEqual(usage.json()["totals"]["requests"], 1)

    def test_not_ready_health_and_stream(self):
        self.app.state.runner = None
        self.app.state.startup_error = "missing key"

        health = self.client.get("/health")
        self.assertEqual(health.status_code, 503)
        self.assertIn("missing key", health.json()["error"])

        response = self.client.post("/chat/stream", json={"query": "你好"})
        events = parse_sse(response.text)
        self.assertEqual(events[0][0], "error")
        self.assertEqual(events[0][1]["code"], "not_ready")

    def test_index_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])


if __name__ == "__main__":
    unittest.main()
