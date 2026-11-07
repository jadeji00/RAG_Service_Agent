import unittest
from unittest.mock import patch

from agent import report_pipeline

RECORD = (
    '{"特征": "65㎡公寓 | 单身 | 木地板", '
    '"效率": "覆盖率:87%", '
    '"耗材": "主刷寿命:剩余28天", '
    '"对比": "覆盖率-1%"}'
)


class ReportPipelineTest(unittest.TestCase):
    def test_generate_report_injects_record_and_fetches_once(self):
        with patch.object(
            report_pipeline, "_fetch_record", return_value=RECORD
        ) as fetch, patch.object(
            report_pipeline, "_write_report", return_value="# 报告"
        ) as writer:
            result = report_pipeline.generate_report(
                "帮我看下 1001 在 2026-10 的使用报告",
                user_id="1001",
                month="2026-10",
            )

        fetch.assert_called_once_with("1001", "2026-10")
        self.assertEqual(result["answer"], "# 报告")
        self.assertEqual(
            result["tool_calls"],
            [
                {
                    "name": "fetch_external_data",
                    "args": {"user_id": "1001", "month": "2026-10"},
                }
            ],
        )
        self.assertTrue(result["fetch_args_ok"])
        self.assertEqual(result["record"]["特征"], "65㎡公寓 | 单身 | 木地板")

        system_prompt = writer.call_args.args[0]
        self.assertIn("【本次使用记录】", system_prompt)
        self.assertIn("覆盖率:87%", system_prompt)
        self.assertNotIn("fill_context_for_report", system_prompt)

    def test_empty_record_skips_model_write(self):
        with patch.object(
            report_pipeline, "_fetch_record", return_value=""
        ), patch.object(report_pipeline, "_write_report") as writer:
            result = report_pipeline.generate_report(
                "用户 1001 在 2026-12 的使用报告",
                user_id="1001",
                month="2026-12",
            )

        writer.assert_not_called()
        self.assertIn("未找到用户 1001 在 2026-12 的使用记录", result["answer"])
        self.assertEqual(result["record"], {})

    def test_missing_params_fall_back_to_mock_tools(self):
        with patch.object(report_pipeline, "_mock_user_id", return_value="1008"), patch.object(
            report_pipeline, "_mock_month", return_value="2026-11"
        ):
            request = report_pipeline.resolve_report_request("帮我生成使用报告")

        self.assertEqual((request.user_id, request.month), ("1008", "2026-11"))

    def test_explicit_month_is_normalized(self):
        request = report_pipeline.resolve_report_request(
            "报告", user_id="1001", month="2026/9"
        )
        self.assertEqual((request.user_id, request.month), ("1001", "2026-09"))


if __name__ == "__main__":
    unittest.main()
