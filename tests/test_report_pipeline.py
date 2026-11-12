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

    def test_owner_denial_skips_fetch_and_model(self):
        with patch.object(report_pipeline, "_fetch_record") as fetch, patch.object(
            report_pipeline, "_write_report"
        ) as writer:
            result = report_pipeline.generate_report(
                "帮我看下 1002 在 2026-10 的使用报告", owner_user_id="1001"
            )

        fetch.assert_not_called()
        writer.assert_not_called()
        self.assertTrue(result["owner_denied"])
        self.assertTrue(result["no_model_call"])
        self.assertEqual(result["tool_calls"], [])
        self.assertIn("1002", result["answer"])
        self.assertIn("1001", result["answer"])

    def test_owner_check_allows_self_and_missing_owner(self):
        request = report_pipeline.ReportRequest(user_id="1002", month="2026-10")
        # 没有登录态（评测脚本 / CLI）不校验，保持旧行为
        self.assertIsNone(report_pipeline.check_report_owner(request, None))
        self.assertIsNone(report_pipeline.check_report_owner(request, ""))
        self.assertIsNone(
            report_pipeline.check_report_owner(
                report_pipeline.ReportRequest(user_id="1001", month="2026-10"), "1001"
            )
        )
        self.assertIsNotNone(report_pipeline.check_report_owner(request, "1001"))

    def test_stream_report_denial_yields_fixed_answer(self):
        with patch.object(report_pipeline, "_fetch_record") as fetch, patch.object(
            report_pipeline, "_mock_month", return_value="2026-10"
        ):
            events = list(
                report_pipeline.stream_report("1002 的报告", owner_user_id="1001")
            )

        fetch.assert_not_called()
        self.assertEqual([event["type"] for event in events], ["meta", "delta", "result"])
        self.assertTrue(events[-1]["owner_denied"])
        self.assertTrue(events[-1]["no_model_call"])
        self.assertEqual(events[-1]["tool_calls"], [])

    def test_empty_record_marks_no_model_call(self):
        with patch.object(report_pipeline, "_fetch_record", return_value=""):
            result = report_pipeline.generate_report(
                "用户 1001 在 2026-12 的使用报告", user_id="1001", month="2026-12"
            )

        # 空记录也不调模型，账本上要记成「本来就没有用量」而不是「供应商漏报」
        self.assertTrue(result["no_model_call"])

    def test_owner_id_fills_in_when_query_has_no_user_id(self):
        # 登录后问「我的报告」不带 ID：要用登录绑定号，不能让 mock 抽一个随机号
        with patch.object(report_pipeline, "_mock_month", return_value="2026-10"), patch.object(
            report_pipeline, "_mock_user_id"
        ) as mock_id:
            request = report_pipeline.resolve_report_request(
                "帮我看下我的使用报告", owner_user_id="1001"
            )

        mock_id.assert_not_called()
        self.assertEqual(request.user_id, "1001")

    def test_owner_id_does_not_override_explicit_request(self):
        with patch.object(report_pipeline, "_mock_month", return_value="2026-10"):
            request = report_pipeline.resolve_report_request(
                "帮我看下 1002 的使用报告", owner_user_id="1001"
            )
        self.assertEqual(request.user_id, "1002")
        self.assertIsNotNone(report_pipeline.check_report_owner(request, "1001"))

    def test_own_report_is_not_denied(self):
        with patch.object(
            report_pipeline, "_fetch_record", return_value=RECORD
        ) as fetch, patch.object(report_pipeline, "_write_report", return_value="# 我的报告"):
            result = report_pipeline.generate_report(
                "帮我看下我 2026-10 的使用报告", owner_user_id="1001"
            )

        fetch.assert_called_once_with("1001", "2026-10")
        self.assertNotIn("owner_denied", result)
        self.assertFalse(result["no_model_call"])
        self.assertEqual(result["answer"], "# 我的报告")


if __name__ == "__main__":
    unittest.main()
