import json
import unittest
from pathlib import Path

from agent.scope_router import (
    OUT_OF_SCOPE_ANSWER,
    is_out_of_scope_query,
    route_intent,
)

ROOT = Path(__file__).resolve().parents[1]


def load_rows(name: str) -> list[dict]:
    path = ROOT / "eval" / name
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class ScopeRouterTest(unittest.TestCase):
    def test_easy_negatives_are_routed_out_of_scope(self):
        rows = load_rows("dataset.jsonl")
        for row in rows:
            if row["type"] != "negative":
                continue
            self.assertEqual(
                route_intent(row["question"]),
                "out_of_scope",
                msg=f"{row['id']} 未命中越界路由：{row['question']}",
            )

    def test_answerable_questions_are_not_refused(self):
        for name in ("dataset.jsonl", "dataset_hard.jsonl"):
            for row in load_rows(name):
                if row["type"] != "answerable":
                    continue
                self.assertFalse(
                    is_out_of_scope_query(row["question"]),
                    msg=f"{row['id']} 被误判越界：{row['question']}",
                )

    def test_washing_machine_usage_is_still_answerable(self):
        # 关键反例：出现「洗衣机」不代表越界，问的是拖布怎么洗。
        self.assertFalse(is_out_of_scope_query("拖布能用洗衣机洗吗？"))
        self.assertTrue(is_out_of_scope_query("你们能上门修洗衣机吗？"))
        self.assertTrue(is_out_of_scope_query("洗衣机报 E01 错误怎么处理？"))

    def test_robot_adjacent_questions_are_not_refused(self):
        # 出现「手机」「感冒」不等于越界：问的仍是扫地机器人的使用。
        self.assertFalse(
            is_out_of_scope_query("扫地机器人连不上手机 App，是手机故障吗？")
        )
        self.assertFalse(
            is_out_of_scope_query("家里有人感冒，扫地机器人使用要注意什么？")
        )
        self.assertTrue(is_out_of_scope_query("感冒发烧应该吃什么药？"))
        self.assertTrue(is_out_of_scope_query("手机黑屏了怎么修？"))

    def test_report_wins_over_writing_rule(self):
        self.assertEqual(route_intent("帮我写一份 1001 在 2026-10 的使用报告"), "report")

    def test_normal_robot_question_is_qa(self):
        self.assertEqual(route_intent("拖布多久清洗一次？"), "qa")
        self.assertIn("服务范围", OUT_OF_SCOPE_ANSWER)


if __name__ == "__main__":
    unittest.main()
