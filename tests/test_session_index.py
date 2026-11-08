import sqlite3
import tempfile
import unittest
from pathlib import Path

from agent.session_index import (
    checkpoint_time_ms,
    count_turns,
    is_session_thread,
    iso_from_ms,
    list_checkpoint_threads,
    make_title,
)


class SessionIndexTest(unittest.TestCase):
    def test_only_frontend_thread_ids_are_sessions(self):
        self.assertTrue(is_session_thread("0b0910c6b1464d79aef08830b34364db"))
        self.assertFalse(is_session_thread("answer-eval-202611062147"))
        self.assertFalse(is_session_thread("smoke-qa-1"))
        self.assertFalse(is_session_thread("35cc1d11-de26-406d-b3d7-bb6ffe46e560"))
        self.assertFalse(is_session_thread(""))

    def test_checkpoint_time_parses_uuid6(self):
        value = checkpoint_time_ms("1f1db6be-536e-6d5b-8010-e10b0261c129")
        self.assertEqual(value, 1794139187876)
        self.assertEqual(iso_from_ms(value), "2026-11-08T11:59:47.876+00:00")

    def test_checkpoint_time_parses_uuid7_and_rejects_unknown(self):
        raw = "0196a1b2c3d4"
        value = checkpoint_time_ms(f"{raw[:8]}-{raw[8:]}-7000-8000-000000000000")
        self.assertEqual(value, int(raw, 16))
        self.assertIsNone(checkpoint_time_ms("not-a-checkpoint"))
        self.assertIsNone(checkpoint_time_ms(None))

    def test_list_checkpoint_threads_skips_eval_and_orders(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "checkpoints.sqlite3"
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint_id TEXT)")
            conn.executemany(
                "INSERT INTO checkpoints (thread_id, checkpoint_id) VALUES (?, ?)",
                [
                    ("answer-eval-abc", "eval-1"),
                    ("a" * 32, "id-1"),
                    ("b" * 32, "id-2"),
                    ("a" * 32, "id-3"),
                ],
            )
            conn.commit()
            conn.close()

            items = list_checkpoint_threads(str(db))

        self.assertEqual([item["thread_id"] for item in items], ["a" * 32, "b" * 32])
        self.assertEqual(items[0]["last_checkpoint"], "id-3")
        self.assertEqual(items[0]["checkpoints"], 2)

    def test_title_uses_first_user_message_and_truncates(self):
        messages = [
            {"role": "assistant", "content": "你好"},
            {"role": "user", "content": " 拖布\n多久  清洗一次？ "},
            {"role": "user", "content": "第二个问题"},
        ]
        self.assertEqual(make_title(messages), "拖布 多久 清洗一次？")
        self.assertEqual(make_title(messages, limit=4), "拖布 多…")
        self.assertEqual(make_title([{"role": "assistant", "content": "x"}]), "")
        self.assertEqual(count_turns(messages), 2)


if __name__ == "__main__":
    unittest.main()
