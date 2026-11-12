"""身份模块单测：口令哈希、令牌生命周期、会话归属。

不调模型、不联网；每个用例一个临时库。``iterations`` 一律调小到 1000，
否则 20 万次 PBKDF2 会把整个套件拖慢好几秒。
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from server.auth import ANONYMOUS_USER, AuthStore, hash_password, verify_password

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


class PasswordHashTest(unittest.TestCase):
    def test_hash_is_salted_and_verifiable(self):
        first = hash_password("demo1234", iterations=1000)
        second = hash_password("demo1234", iterations=1000)
        self.assertNotEqual(first, second, "同一个口令两次哈希必须不同（盐随机）")
        self.assertNotIn("demo1234", first)
        self.assertTrue(verify_password("demo1234", first))
        self.assertTrue(verify_password("demo1234", second))
        self.assertFalse(verify_password("demo1235", first))

    def test_same_salt_gives_same_hash(self):
        salt = b"0123456789abcdef"
        self.assertEqual(
            hash_password("x", iterations=1000, salt=salt),
            hash_password("x", iterations=1000, salt=salt),
        )

    def test_malformed_stores_are_rejected_not_raised(self):
        for stored in ["", "plain-text", "pbkdf2_sha256$abc", "bcrypt$1$x$y", None]:
            self.assertFalse(verify_password("demo1234", stored), stored)


class AuthStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "auth.sqlite3")
        self.auth = AuthStore(
            self.db_path, iterations=1000, demo_users=DEMO_USERS, legacy_owner="demo"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_seeds_demo_users_once(self):
        self.assertEqual(
            [item["username"] for item in self.auth.demo_users()], ["demo", "alice"]
        )
        # 再建一次同一个库：不重复写、不覆盖已有账号
        again = AuthStore(self.db_path, iterations=1000, demo_users=DEMO_USERS)
        self.assertEqual(len(again.demo_users()), 2)
        self.assertIsNotNone(again.login("demo", "demo1234"))

    def test_demo_users_payload_has_no_password(self):
        payload = self.auth.demo_users()
        self.assertEqual(payload[0]["display_name"], "演示用户")
        self.assertEqual(payload[0]["user_id"], "1001")
        self.assertNotIn("password", str(payload))
        self.assertNotIn("demo1234", str(payload))

    def test_env_override_wins_over_config_password(self):
        auth = AuthStore(
            str(Path(self.tmp.name) / "env.sqlite3"),
            iterations=1000,
            demo_users=DEMO_USERS,
        )
        with patch.dict(os.environ, {"AUTH_DEMO_PASSWORD": "rotated-pass"}):
            rotated = AuthStore(
                str(Path(self.tmp.name) / "env2.sqlite3"),
                iterations=1000,
                demo_users=DEMO_USERS,
            )
        self.assertIsNotNone(rotated.login("demo", "rotated-pass"))
        self.assertIsNone(rotated.login("demo", "demo1234"))
        self.assertIsNotNone(auth.login("demo", "demo1234"))

    def test_login_rejects_unknown_user_and_wrong_password(self):
        self.assertIsNone(self.auth.login("demo", "wrong"))
        self.assertIsNone(self.auth.login("nobody", "demo1234"))
        self.assertIsNone(self.auth.login("", ""))
        self.assertIsNone(self.auth.login("  demo  ", "wrong"))

    def test_login_returns_user_with_business_id(self):
        result = self.auth.login("alice", "alice1234")
        self.assertIsNotNone(result)
        user, token = result
        self.assertEqual(user.username, "alice")
        self.assertEqual(user.business_user_id, "1002")
        self.assertEqual(user.as_dict()["user_id"], "1002")
        self.assertGreaterEqual(len(token), 32)
        self.assertEqual(self.auth.user_by_token(token), user)

    def test_trimmed_username_still_logs_in(self):
        self.assertIsNotNone(self.auth.login("  demo", "demo1234"))

    def test_logout_revokes_only_that_token(self):
        _, first = self.auth.login("demo", "demo1234")
        _, second = self.auth.login("demo", "demo1234")
        self.auth.logout(first)
        self.assertIsNone(self.auth.user_by_token(first))
        self.assertIsNotNone(self.auth.user_by_token(second))
        self.auth.logout("")  # 空令牌不报错

    def test_expired_token_is_rejected(self):
        self.auth.token_ttl_hours = -1
        _, token = self.auth.login("demo", "demo1234")
        self.assertIsNone(self.auth.user_by_token(token))

    def test_token_lookup_tolerates_junk(self):
        self.assertIsNone(self.auth.user_by_token(None))
        self.assertIsNone(self.auth.user_by_token(""))
        self.assertIsNone(self.auth.user_by_token("not-a-token"))

    def test_claim_thread_is_first_writer_wins(self):
        thread = "a" * 32
        self.assertEqual(self.auth.claim_thread(thread, "demo"), "demo")
        # 第二个人抢不走，只能拿到真正的归属者
        self.assertEqual(self.auth.claim_thread(thread, "alice"), "demo")
        self.assertEqual(self.auth.thread_owner(thread), "demo")
        self.assertIsNone(self.auth.claim_thread("   ", "demo"))
        self.assertIsNone(self.auth.thread_owner("未认领的线程"))

    def test_owned_thread_ids_and_filter(self):
        self.auth.claim_thread("a" * 32, "demo")
        self.auth.claim_thread("b" * 32, "demo")
        self.auth.claim_thread("c" * 32, "alice")

        self.assertEqual(self.auth.owned_thread_ids("demo"), {"a" * 32, "b" * 32})
        predicate = self.auth.owner_filter("demo")
        self.assertTrue(predicate("a" * 32))
        self.assertFalse(predicate("c" * 32))
        self.assertEqual(self.auth.owned_thread_ids("nobody"), set())

    def test_release_thread_clears_ownership(self):
        self.auth.claim_thread("a" * 32, "demo")
        self.auth.release_thread("a" * 32)
        self.assertIsNone(self.auth.thread_owner("a" * 32))
        self.assertEqual(self.auth.claim_thread("a" * 32, "alice"), "alice")

    def test_adopt_threads_only_touches_unowned_rows(self):
        self.auth.claim_thread("a" * 32, "alice")
        adopted = self.auth.adopt_threads(["a" * 32, "b" * 32, "", None], "demo")
        self.assertEqual(adopted, 1)
        self.assertEqual(self.auth.thread_owner("a" * 32), "alice")
        self.assertEqual(self.auth.thread_owner("b" * 32), "demo")
        # 再跑一次不会重复认领
        self.assertEqual(self.auth.adopt_threads(["b" * 32], "demo"), 0)
        self.assertEqual(self.auth.adopt_threads(["c" * 32], "  "), 0)

    def test_disabled_store_still_answers_queries(self):
        auth = AuthStore(
            str(Path(self.tmp.name) / "off.sqlite3"),
            enabled=False,
            iterations=1000,
            demo_users=DEMO_USERS,
        )
        self.assertFalse(auth.enabled)
        self.assertIsNotNone(auth.login("demo", "demo1234"))
        self.assertEqual(ANONYMOUS_USER.username, "anonymous")
        self.assertIsNone(ANONYMOUS_USER.business_user_id)


if __name__ == "__main__":
    unittest.main()
