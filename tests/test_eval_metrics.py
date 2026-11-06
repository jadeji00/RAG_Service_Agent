import unittest

from eval.metrics import (
    first_relevant_rank,
    first_source_rank,
    normalize,
    percentile,
    source_basename,
    summarize_ranks,
)


class FakeDocument:
    def __init__(self, content: str, source: str):
        self.page_content = content
        self.metadata = {"source": source}


class MetricsTest(unittest.TestCase):
    def test_normalize_removes_whitespace(self):
        self.assertEqual(normalize("【E01 碰撞\n传感器】"), normalize("【E01碰撞传感器】"))

    def test_source_basename_handles_windows_path(self):
        self.assertEqual(
            source_basename(r"D:\AI_RAG_Agent\data\故障排除.txt"),
            "故障排除.txt",
        )

    def test_term_match_requires_gold_source(self):
        item = {
            "gold_sources": ["选购指南.txt"],
            "gold_sections": ["一、先明确需求"],
            "gold_terms": ["60㎡以下", "性价比"],
        }
        self.assertTrue(
            is_relevant_doc(FakeDocument("60㎡以下看性价比", "D:/data/选购指南.txt"), item)
        )
        self.assertFalse(
            is_relevant_doc(FakeDocument("60㎡以下看性价比", "D:/data/其他.txt"), item)
        )

    def test_section_fallback_when_no_terms(self):
        item = {
            "gold_sources": ["故障排除.txt"],
            "gold_sections": ["【回充失败】"],
            "gold_terms": [],
        }
        self.assertTrue(
            is_relevant_doc(FakeDocument("【回充失败】检查充电座", "data/故障排除.txt"), item)
        )
        self.assertFalse(
            is_relevant_doc(FakeDocument("【频繁掉线】检查 Wi-Fi", "data/故障排除.txt"), item)
        )

    def test_rank_is_one_based(self):
        item = {"gold_sources": ["a.txt"], "gold_sections": ["Q1"], "gold_terms": []}
        docs = [
            FakeDocument("Q9", "a.txt"),
            FakeDocument("Q1", "a.txt"),
        ]
        self.assertEqual(first_relevant_rank(docs, item), 2)
        self.assertEqual(first_source_rank(docs, item), 1)

    def test_summarize_ranks(self):
        summary = summarize_ranks([1, 2, None, 4], [1, 3, 5])
        self.assertAlmostEqual(summary["recall@1"], 0.25)
        self.assertAlmostEqual(summary["recall@3"], 0.5)
        self.assertAlmostEqual(summary["recall@5"], 0.75)
        self.assertAlmostEqual(summary["mrr@1"], 0.25)
        self.assertAlmostEqual(summary["mrr@3"], 0.375)
        self.assertAlmostEqual(summary["mrr@5"], 0.4375)

    def test_percentile(self):
        self.assertEqual(percentile([10, 20, 30, 40], 50), 30)
        self.assertEqual(percentile([], 95), 0.0)


def is_relevant_doc(document, item):
    from eval.metrics import is_relevant

    return is_relevant(document, item)


if __name__ == "__main__":
    unittest.main()
