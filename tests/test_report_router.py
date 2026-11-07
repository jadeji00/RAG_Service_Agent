import json
import unittest
from pathlib import Path

from agent.report_router import (
    extract_report_request,
    is_report_query,
    parse_month,
    parse_user_id,
)

ROOT = Path(__file__).resolve().parents[1]


class ReportRouterTest(unittest.TestCase):
    def test_parse_user_id_excludes_year(self):
        self.assertEqual(parse_user_id("用户 1001 在 2026-10 的报告"), "1001")
        self.assertIsNone(parse_user_id("2026-10 的报告"))
        self.assertIsNone(parse_user_id("没有编号"))

    def test_parse_month_formats(self):
        self.assertEqual(parse_month("2026-10"), "2026-10")
        self.assertEqual(parse_month("2026/9"), "2026-09")
        self.assertEqual(parse_month("2026年7月"), "2026-07")
        self.assertIsNone(parse_month("2026-13"))
        self.assertIsNone(parse_month("没有月份"))

    def test_report_query_with_explicit_params(self):
        question = "用户 1002 在 2026-09 的清洁效率和耗材情况怎么样？"
        self.assertTrue(is_report_query(question))
        request = extract_report_request(question)
        self.assertEqual((request.user_id, request.month), ("1002", "2026-09"))

    def test_report_query_with_explicit_word(self):
        self.assertTrue(is_report_query("帮我生成我的使用报告"))

    def test_knowledge_question_is_not_report(self):
        self.assertFalse(is_report_query("拖布多久清洗一次？"))
        self.assertFalse(is_report_query("耗材多久换一次？"))
        self.assertFalse(is_report_query("2026-10 的天气怎么样？"))

    def test_dataset_routing_split(self):
        rows = [
            json.loads(line)
            for line in (ROOT / "eval" / "dataset.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        for row in rows:
            expected = row["type"] == "report"
            self.assertEqual(
                is_report_query(row["question"]),
                expected,
                msg=f"{row['id']} 路由结果与类型不一致",
            )


if __name__ == "__main__":
    unittest.main()
