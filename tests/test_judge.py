import unittest
from pathlib import Path

from eval.judge import (
    compact,
    extract_json,
    load_records,
    refusal_cue,
    score_report_facts,
)

ROOT = Path(__file__).resolve().parents[1]

RECORD = {
    "特征": "65㎡公寓 | 单身 | 木地板",
    "清洁效率": "覆盖率:87%\\n日均清扫:47㎡\\n漏扫区域:茶几底部（高度不足）",
    "耗材": "主刷寿命:剩余28天\\n滤网状态:正常",
    "对比": "覆盖率-1%、日均清扫-1㎡；本月出差4天出勤减少",
}
FIELDS = ["特征", "清洁效率", "耗材", "对比"]

GOOD_ANSWER = """用户 1001，65㎡公寓，单身，木地板。
清洁效率：覆盖率 87%，日均清扫 47㎡，漏扫区域为茶几底部（高度不足）。
耗材：主刷剩余寿命约 28 天，滤网状态正常。
对比：覆盖率下降 1 个百分点，日均清扫面积减少 1 平方米；本月出差 4 天，出勤减少。
建议：提前准备主刷，并针对茶几底部设置定点清扫。"""


class JsonParseTest(unittest.TestCase):
    def test_extract_json_from_code_block(self):
        text = '```json\n{"correctness": 2, "faithfulness": 1}\n```'
        self.assertEqual(
            extract_json(text), {"correctness": 2, "faithfulness": 1}
        )

    def test_extract_json_from_prose(self):
        text = '判分结果如下：{"refusal": 1, "reason": "明确拒答"} 完毕'
        self.assertEqual(
            extract_json(text), {"refusal": 1, "reason": "明确拒答"}
        )

    def test_extract_json_invalid(self):
        self.assertIsNone(extract_json("not a json"))
        self.assertIsNone(extract_json("{"))
        self.assertIsNone(extract_json("[1, 2]"))


class CompactTest(unittest.TestCase):
    def test_compact_removes_markdown_and_unifies_units(self):
        self.assertEqual(compact("**覆盖率**: 87%"), "覆盖率87%")
        self.assertEqual(compact("下降 1 个百分点"), "下降1%")
        self.assertEqual(compact("减少 1 平方米"), "减少1㎡")


class ReportFactsTest(unittest.TestCase):
    def test_complete_answer_covers_all_fields(self):
        result = score_report_facts(GOOD_ANSWER, RECORD, FIELDS)
        self.assertTrue(result["complete"])
        self.assertEqual(result["field_coverage"], 1.0)
        self.assertEqual(result["fact_coverage"], 1.0)

    def test_paraphrased_units_still_match(self):
        """报告写成「1 个百分点」「主刷剩余寿命」不应被判缺失。"""
        result = score_report_facts(GOOD_ANSWER, RECORD, FIELDS)
        self.assertTrue(result["fields"]["对比"]["covered"])
        self.assertTrue(result["fields"]["耗材"]["covered"])

    def test_missing_comparison_is_detected(self):
        answer = """用户 1001，65㎡公寓，单身，木地板。
清洁效率：覆盖率 87%，日均清扫 47㎡，漏扫区域为茶几底部。
耗材：主刷剩余寿命约 28 天，滤网状态正常。"""
        result = score_report_facts(answer, RECORD, FIELDS)
        self.assertFalse(result["complete"])
        self.assertFalse(result["fields"]["对比"]["covered"])
        self.assertEqual(result["field_coverage"], 0.75)

    def test_positive_change_without_plus_sign(self):
        """CSV 里的 +1% 应能匹配报告里的「提升 1%」。"""
        record = dict(RECORD)
        record["对比"] = "覆盖率+1%、日均清扫+1㎡；胶刷临近更换"
        answer = "覆盖率提升 1%，日均清扫面积增加 1 平方米；胶刷快该换了。"
        result = score_report_facts(answer, record, ["对比"])
        self.assertTrue(result["complete"])

    def test_baseline_month_without_comparison(self):
        record = dict(RECORD)
        record["对比"] = "基线数据，无对比"
        answer = "本月为基线数据，无对比；覆盖率 87%。"
        result = score_report_facts(answer, record, ["对比"])
        self.assertTrue(result["complete"])


class RefusalCueTest(unittest.TestCase):
    def test_refusal_phrases(self):
        self.assertTrue(refusal_cue("参考资料中没有提到上门维修洗衣机的服务。"))
        self.assertTrue(refusal_cue("我无法回答这个问题。"))
        self.assertTrue(refusal_cue("这不在我们的服务范围内。"))

    def test_normal_answer_is_not_refusal(self):
        self.assertFalse(refusal_cue("扫地机器人主要用于清扫地面的干垃圾。"))
        self.assertFalse(refusal_cue("可以的，我们提供上门维修服务。"))


class RecordsTest(unittest.TestCase):
    def test_load_records_indexes_user_and_month(self):
        records = load_records(ROOT / "data" / "external" / "records.csv")
        record = records[("1001", "2026-10")]
        self.assertIn("65㎡公寓", record["特征"])
        self.assertIn("87%", record["清洁效率"])


if __name__ == "__main__":
    unittest.main()
