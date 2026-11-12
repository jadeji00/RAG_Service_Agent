import json
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from observability.store import TraceStore
from server.app import create_app
from server.auth import AuthStore
from server.chat_runner import ChatRunner
from server.rate_limit import SlidingWindowLimiter

DEMO_USERS = [
    {
        "username": "demo",
        "display_name": "演示用户",
        "password": "demo1234",
        "business_user_id": "1001",
    },
    {
        "username": "alice",
        "display_name": "Alice",
        "password": "alice1234",
        "business_user_id": "1002",
    },
]


class FakeAgent:
    model_name = "fake-model"

    def __init__(self):
        self.owner_user_ids = []

    def iter_chat(self, query, thread_id="default", token_account=None, owner_user_id=None):
        self.owner_user_ids.append(owner_user_id)
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

    def list_sessions(self, limit=20, thread_filter=None):
        items = [
            {
                "thread_id": "a" * 32,
                "title": "拖布多久清洗一次？",
                "turns": 2,
                "updated_at": "2026-11-08T12:00:00.000+00:00",
            },
            {
                "thread_id": "b" * 32,
                "title": "滤网怎么更换？",
                "turns": 1,
                "updated_at": "2026-11-07T12:00:00.000+00:00",
            },
        ]
        if thread_filter is not None:
            items = [item for item in items if thread_filter(item["thread_id"])]
        return items[:limit]

    def delete_session(self, thread_id):
        if thread_id == "a" * 32:
            return {"checkpoints": 2, "writes": 3}
        return {"checkpoints": 0, "writes": 0}


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


class _AppTestCase(unittest.TestCase):
    """共用装配：临时 trace 库 + 临时 auth 库 + fake agent，默认已登录 demo。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = TraceStore(str(Path(self.tmp.name) / "api.sqlite3"))
        # iterations 调小：单测里要哈希很多次，20 万次会把套件拖慢好几秒。
        self.auth = AuthStore(
            str(Path(self.tmp.name) / "auth.sqlite3"),
            iterations=1000,
            demo_users=DEMO_USERS,
        )
        self.agent = FakeAgent()
        self.runner = ChatRunner(self.agent, store=self.store, model_name="fake-model")
        self.app = create_app(runner=self.runner, store=self.store, auth=self.auth)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.token = self.login("demo", "demo1234")
        self.client.headers.update({"Authorization": f"Bearer {self.token}"})

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()

    def login(self, username: str, password: str) -> str:
        response = self.client.post(
            "/auth/login", json={"username": username, "password": password}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["token"]

    def claim(self, *thread_ids: str, user: str = "demo") -> None:
        for thread_id in thread_ids:
            self.auth.claim_thread(thread_id, user)


class ApiTest(_AppTestCase):
    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "fake-model")
        self.assertTrue(response.json()["auth"])

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

    def test_sessions_endpoint_lists_and_validates_limit(self):
        self.claim("a" * 32, "b" * 32)
        response = self.client.get("/sessions?limit=1")
        self.assertEqual(response.status_code, 200)
        items = response.json()["sessions"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "拖布多久清洗一次？")
        self.assertEqual(self.client.get("/sessions?limit=0").status_code, 422)
        self.assertEqual(self.client.get("/sessions?limit=51").status_code, 422)

    def test_delete_session_removes_checkpoints_and_traces(self):
        thread_id = "a" * 32
        self.claim(thread_id)
        self.store.start_trace("trace-del", thread_id, "要删除的问题", "fake-model")
        self.store.add_span("trace-del", "tool:rag_summarize", "tool", duration_ms=2.0)

        response = self.client.delete(f"/sessions/{thread_id}")
        self.assertEqual(response.status_code, 200)
        deleted = response.json()["deleted"]
        self.assertEqual(deleted["checkpoints"], 2)
        self.assertEqual(deleted["writes"], 3)
        self.assertEqual(deleted["traces"], 1)
        self.assertEqual(deleted["spans"], 1)
        self.assertIsNone(self.store.get_trace("trace-del"))
        # 归属行也要一起清掉，否则线程 ID 复用时会继承旧主人
        self.assertIsNone(self.auth.thread_owner(thread_id))

        self.assertEqual(self.client.delete("/sessions/not-a-session").status_code, 422)
        self.assertEqual(self.client.delete(f"/sessions/{'c' * 32}").status_code, 404)

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

        sessions = self.client.get("/sessions")
        self.assertEqual(sessions.status_code, 503)

        deleted = self.client.delete(f"/sessions/{'a' * 32}")
        self.assertEqual(deleted.status_code, 503)

    def test_index_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])


class AuthApiTest(_AppTestCase):
    """登录、令牌与 thread 归属。"""

    def test_endpoints_require_login(self):
        self.client.headers.clear()
        for method, url in [
            ("get", "/sessions"),
            ("get", "/history/api-2"),
            ("get", "/usage"),
            ("get", "/traces"),
            ("delete", f"/sessions/{'a' * 32}"),
        ]:
            self.assertEqual(
                getattr(self.client, method)(url).status_code, 401, url
            )
        response = self.client.post("/chat/stream", json={"query": "你好"})
        self.assertEqual(response.status_code, 401)

        # 入口必须公开，否则永远登录不进去
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/auth/bootstrap").status_code, 200)

    def test_bare_token_and_bad_scheme_are_rejected(self):
        self.client.headers.update({"Authorization": self.token})
        self.assertEqual(self.client.get("/auth/me").status_code, 401)
        self.client.headers.update({"Authorization": f"token {self.token}"})
        self.assertEqual(self.client.get("/auth/me").status_code, 401)

    def test_bootstrap_never_leaks_credentials(self):
        self.client.headers.clear()
        payload = self.client.get("/auth/bootstrap").json()
        self.assertTrue(payload["enabled"])
        self.assertEqual(
            [item["username"] for item in payload["users"]], ["demo", "alice"]
        )
        for item in payload["users"]:
            self.assertEqual(
                sorted(item), ["display_name", "user_id", "username"]
            )
        raw = json.dumps(payload, ensure_ascii=False)
        # 口令哈希永远不出库；hint 里的演示口令是配置里写死的公开信息，
        # 目的是让访客能进得来，换部署环境时改 config/auth.yml 即可。
        self.assertNotIn("pbkdf2_sha256", raw)
        self.assertNotIn("password_hash", raw)
        self.assertEqual(payload["hint"], self.auth.login_hint)

    def test_login_rejects_wrong_credentials(self):
        self.client.headers.clear()
        bad_password = self.client.post(
            "/auth/login", json={"username": "demo", "password": "wrong"}
        )
        self.assertEqual(bad_password.status_code, 401)
        unknown = self.client.post(
            "/auth/login", json={"username": "nobody", "password": "demo1234"}
        )
        self.assertEqual(unknown.status_code, 401)
        self.assertEqual(unknown.headers.get("WWW-Authenticate"), None)

    def test_me_and_logout(self):
        me = self.client.get("/auth/me").json()
        self.assertEqual(me["user"]["username"], "demo")
        self.assertEqual(me["user"]["user_id"], "1001")

        self.assertEqual(self.client.post("/auth/logout").status_code, 200)
        self.assertEqual(self.client.get("/auth/me").status_code, 401)

    def test_token_outlives_process_and_expiry_is_enforced(self):
        # 令牌在库里，换一个 client（相当于重启进程）仍然有效
        second = TestClient(self.app)
        second.__enter__()
        self.addCleanup(second.__exit__, None, None, None)
        second.headers.update({"Authorization": f"Bearer {self.token}"})
        self.assertEqual(second.get("/auth/me").status_code, 200)

        self.auth.token_ttl_hours = -1
        expired = self.login("demo", "demo1234")
        response = self.client.get(
            "/auth/me", headers={"Authorization": f"Bearer {expired}"}
        )
        self.assertEqual(response.status_code, 401)

    def test_thread_ownership_blocks_other_users(self):
        mine = "1" * 32
        self.assertEqual(
            self.client.post(
                "/chat/stream", json={"query": "你好", "thread_id": mine}
            ).status_code,
            200,
        )
        self.assertEqual(self.auth.thread_owner(mine), "demo")
        # 登录账号绑定的业务 ID 要透传给报告链路
        self.assertEqual(self.agent.owner_user_ids[-1], "1001")

        alice = {"Authorization": f"Bearer {self.login('alice', 'alice1234')}"}
        self.assertEqual(
            self.client.post(
                "/chat/stream", json={"query": "你好", "thread_id": mine}, headers=alice
            ).status_code,
            403,
        )
        self.assertEqual(self.client.get(f"/history/{mine}", headers=alice).status_code, 403)
        self.assertEqual(
            self.client.delete(f"/sessions/{mine}", headers=alice).status_code, 403
        )
        # 越权请求不应该跑到 agent 里
        self.assertNotIn("1002", self.agent.owner_user_ids)
        # 自己的线程照常可用
        self.assertEqual(self.client.get(f"/history/{mine}").status_code, 200)

    def test_sessions_are_scoped_to_owner(self):
        self.claim("a" * 32, user="demo")
        self.claim("b" * 32, user="alice")

        mine = self.client.get("/sessions?limit=10").json()["sessions"]
        self.assertEqual([item["thread_id"] for item in mine], ["a" * 32])

        alice = {"Authorization": f"Bearer {self.login('alice', 'alice1234')}"}
        theirs = self.client.get("/sessions?limit=10", headers=alice).json()["sessions"]
        self.assertEqual([item["thread_id"] for item in theirs], ["b" * 32])

    def test_missing_thread_id_gets_a_fresh_one(self):
        response = self.client.post("/chat/stream", json={"query": "你好"})
        events = parse_sse(response.text)
        thread_id = events[0][1]["thread_id"]
        self.assertNotEqual(thread_id, "default")
        self.assertEqual(len(thread_id), 32)
        self.assertEqual(self.auth.thread_owner(thread_id), "demo")

        # 第二个账号不会因为都用过 default 而被挡在门外
        alice = {"Authorization": f"Bearer {self.login('alice', 'alice1234')}"}
        other = parse_sse(
            self.client.post("/chat/stream", json={"query": "你好"}, headers=alice).text
        )
        self.assertEqual(self.auth.thread_owner(other[0][1]["thread_id"]), "alice")

    def test_auth_disabled_keeps_old_behaviour(self):
        auth = AuthStore(
            str(Path(self.tmp.name) / "off.sqlite3"),
            enabled=False,
            iterations=1000,
            demo_users=DEMO_USERS,
        )
        app = create_app(runner=self.runner, store=self.store, auth=auth)
        client = TestClient(app)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)

        self.assertFalse(client.get("/auth/bootstrap").json()["enabled"])
        self.assertEqual(
            client.post(
                "/auth/login", json={"username": "demo", "password": "demo1234"}
            ).status_code,
            400,
        )
        # 不带令牌也能用，会话不做归属过滤
        self.assertEqual(client.get("/sessions").status_code, 200)
        self.assertEqual(len(client.get("/sessions").json()["sessions"]), 2)
        thread_id = "c" * 32
        self.assertEqual(
            client.post(
                "/chat/stream", json={"query": "你好", "thread_id": thread_id}
            ).status_code,
            200,
        )
        self.assertIsNone(auth.thread_owner(thread_id))
        self.assertIsNone(self.agent.owner_user_ids[-1])


class RateLimitApiTest(_AppTestCase):
    """``/chat/stream`` 的按用户限流：超额返回 429 + ``Retry-After``。"""

    def use_limiter(self, **kwargs) -> None:
        """换上受控的限流器：默认是 20 次/分钟，单测里既等不起也不想等。"""
        self.client.__exit__(None, None, None)
        self.app = create_app(
            runner=self.runner,
            store=self.store,
            auth=self.auth,
            limiter=SlidingWindowLimiter(**kwargs),
        )
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.client.headers.update({"Authorization": f"Bearer {self.token}"})

    def ask(self, thread_id: str, token: str | None = None):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return self.client.post(
            "/chat/stream", json={"query": "滤网怎么拆", "thread_id": thread_id},
            headers=headers,
        )

    def test_third_request_is_throttled(self):
        self.use_limiter(enabled=True, window_seconds=60, max_requests=2)

        self.assertEqual(self.ask("rl-1").status_code, 200)
        self.assertEqual(self.ask("rl-1").status_code, 200)
        blocked = self.ask("rl-1")

        self.assertEqual(blocked.status_code, 429)
        self.assertEqual(blocked.headers["Retry-After"], "60")
        self.assertIn("请求太频繁", blocked.json()["detail"])

    def test_throttled_request_leaves_no_trace(self):
        self.use_limiter(enabled=True, window_seconds=60, max_requests=1)

        self.ask("rl-owned")
        blocked = self.ask("rl-fresh")

        self.assertEqual(blocked.status_code, 429)
        # 认领线程是写操作，被限流的请求不该在归属表里留下痕迹。
        self.assertIsNone(self.auth.thread_owner("rl-fresh"))
        self.assertEqual(self.agent.owner_user_ids, ["1001"])
        # 也不该产生 trace：请求根本没走到 runner。
        self.assertEqual(
            [t["thread_id"] for t in self.store.list_traces(limit=20)], ["rl-owned"]
        )

    def test_budget_is_per_account(self):
        self.use_limiter(enabled=True, window_seconds=60, max_requests=1)
        alice_token = self.login("alice", "alice1234")

        self.assertEqual(self.ask("rl-demo").status_code, 200)
        self.assertEqual(self.ask("rl-demo").status_code, 429)
        self.assertEqual(self.ask("rl-alice", token=alice_token).status_code, 200)

    def test_read_only_endpoints_are_not_throttled(self):
        self.use_limiter(enabled=True, window_seconds=60, max_requests=1)
        self.claim("d" * 32, "demo")

        self.assertEqual(self.ask("rl-1").status_code, 200)
        self.assertEqual(self.ask("rl-1").status_code, 429)
        # 只有会调模型的 /chat/stream 计数；查询类接口被限流会让页面直接不可用。
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/sessions").status_code, 200)
        self.assertEqual(self.client.get(f"/history/{'d' * 32}").status_code, 200)
        self.assertEqual(self.client.get("/usage").status_code, 200)

    def test_disabled_limiter_never_throttles(self):
        self.use_limiter(enabled=False, window_seconds=60, max_requests=1)

        for _ in range(10):
            self.assertEqual(self.ask("rl-off").status_code, 200)

    def test_window_slides_in_real_time(self):
        # 用最短的窗口跑一次真实计时，确认接的是 monotonic 而不是某个不会走的假时钟。
        self.use_limiter(enabled=True, window_seconds=0.2, max_requests=1)

        self.assertEqual(self.ask("rl-slide").status_code, 200)
        self.assertEqual(self.ask("rl-slide").status_code, 429)
        time.sleep(0.25)
        self.assertEqual(self.ask("rl-slide").status_code, 200)


if __name__ == "__main__":
    unittest.main()
