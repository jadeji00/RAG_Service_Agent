import unittest
from collections import Counter
from pathlib import Path

from eval.run_eval import load_rows, validate_rows

ROOT = Path(__file__).resolve().parents[1]


class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_rows(ROOT / "eval" / "dataset.jsonl")

    def test_dataset_size_and_distribution(self):
        self.assertEqual(len(self.rows), 80)
        self.assertEqual(
            Counter(row["type"] for row in self.rows),
            {"answerable": 55, "negative": 15, "report": 10},
        )

    def test_gold_labels_match_current_data(self):
        errors = validate_rows(
            self.rows,
            ROOT / "data",
            ROOT / "data" / "external" / "records.csv",
        )
        self.assertEqual(errors, [])

    def test_all_ids_unique(self):
        ids = [row["id"] for row in self.rows]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
