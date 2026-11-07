import unittest

from eval.run_answer_eval import _attach_runs, _passk_stats


class PassKTest(unittest.TestCase):
    def test_strict_and_at_least_once(self):
        rows = [
            {"id": "a", "pass_count": 3},
            {"id": "b", "pass_count": 2},
            {"id": "c", "pass_count": 0},
        ]
        stats = _passk_stats(rows, repeat=3)

        self.assertAlmostEqual(stats["strict_pass_rate"], 1 / 3)
        self.assertAlmostEqual(stats["at_least_once_rate"], 2 / 3)
        self.assertAlmostEqual(stats["mean_run_pass_rate"], 5 / 9)
        self.assertEqual(stats["flaky_count"], 1)
        self.assertEqual(stats["flaky_ids"], ["b"])

    def test_repeat_one_returns_none(self):
        self.assertIsNone(_passk_stats([{"id": "a", "pass_count": 1}], repeat=1))

    def test_attach_runs_keeps_first_run_fields(self):
        runs = [
            {
                "passed": True,
                "answer": "first",
                "verdict": {"correctness": 2, "faithfulness": 2},
                "gen_latency_ms": 10,
                "error": "",
            },
            {
                "passed": False,
                "answer": "second",
                "verdict": {"correctness": 1, "faithfulness": 1},
                "gen_latency_ms": 11,
                "error": "",
            },
        ]
        merged = _attach_runs(runs[0], runs, repeat=2)

        self.assertEqual(merged["answer"], "first")
        self.assertEqual(merged["pass_count"], 1)
        self.assertFalse(merged["passed_all"])
        self.assertTrue(merged["passed_any"])
        self.assertEqual(len(merged["runs"]), 2)


if __name__ == "__main__":
    unittest.main()
