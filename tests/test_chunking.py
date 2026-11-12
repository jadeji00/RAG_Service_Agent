import glob
import unittest
from pathlib import Path

from langchain_core.documents import Document

from rag.chunking import split_documents, split_signature, split_text

QA_TEXT = """扫地机器人常见问题（使用与故障篇）

【日常使用】
Q1：第一次使用需要充多久？
A1：建议先充满电再建图，通常需要 3–4 小时。

Q2：可以一直放在充电座上吗？
A2：可以，机器充满后会停止充电。

【拖地与水箱】
Q1：拖布多久洗一次？
A1：每次拖地后清洗；自动洗拖布的机型也要定期取出彻底清洗。
"""

FAULT_TEXT = """扫地机器人故障排查手册

说明：不同品牌的错误码含义可能不同。

一、错误代码
【E05 悬崖或轮组异常】
原因：下视传感器被灰尘遮挡、机器悬空、轮组卡死。
处理：擦净底部传感器；把机器放到平整地面重启。

二、常见现象
【吸力变弱】
原因：尘盒满、滤网堵塞、主刷缠绕。
处理：清空尘盒；清洗或更换滤网。

三、排查顺序建议
1. 先看 App 报错和说明书错误码。
2. 再断电清理刷子、尘盒、滤网和传感器。
"""

LIST_TEXT = """扫地机器人维护保养手册

一、每日维护
1. 清空尘盒：每次清扫后倒掉垃圾。
2. 清理主刷：拔下主刷，割断缠绕的毛发。
3. 擦拭传感器：用干软布擦回充传感器。
4. 检查拖布：拖地后取下拖布清洗。

二、每周维护
1. 清洗滤网：必须完全晾干再装回。
"""


class ChunkingTest(unittest.TestCase):
    def test_qa_pair_stays_in_one_chunk_with_section(self):
        chunks = split_text(QA_TEXT)
        qa_chunks = [c for c in chunks if c.mode == "qa"]

        self.assertEqual(len(qa_chunks), 3)
        first = qa_chunks[0]
        self.assertIn("【日常使用】", first.text)
        self.assertIn("Q1：第一次使用需要充多久？", first.text)
        self.assertIn("A1：建议先充满电再建图", first.text)
        self.assertEqual(first.section, "日常使用")
        self.assertEqual(first.unit, "Q1")

        # 第二个小节的问题不会串到第一个小节里。
        last = qa_chunks[-1]
        self.assertEqual(last.section, "拖地与水箱")
        self.assertEqual(last.unit, "Q1")
        self.assertNotIn("充电座", last.text)

    def test_error_code_block_keeps_reason_and_action(self):
        chunks = split_text(FAULT_TEXT)
        e05 = next(c for c in chunks if c.unit == "E05 悬崖或轮组异常")

        self.assertEqual(e05.mode, "block")
        self.assertEqual(e05.section, "错误代码")
        self.assertIn("一、错误代码", e05.text)
        self.assertIn("【E05 悬崖或轮组异常】", e05.text)
        self.assertIn("原因：", e05.text)
        self.assertIn("处理：", e05.text)
        self.assertNotIn("吸力变弱", e05.text)

    def test_numbered_list_groups_and_repeats_heading(self):
        chunks = [c for c in split_text(LIST_TEXT, max_unit_chars=60) if c.mode == "list"]

        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertTrue(chunk.text.startswith("一、每日维护") or chunk.text.startswith("二、每周维护"))
        self.assertEqual(chunks[0].unit, "1-2")
        self.assertEqual(chunks[0].section, "每日维护")
        self.assertIn("1. 清空尘盒", chunks[0].text)
        self.assertIn("2. 清理主刷", chunks[0].text)
        self.assertNotIn("3. 擦拭传感器", chunks[0].text)

    def test_whole_section_kept_when_under_limit(self):
        chunks = [c for c in split_text(LIST_TEXT, max_unit_chars=600) if c.mode == "list"]
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].unit, "1-4")
        self.assertEqual(chunks[1].unit, "1")
        self.assertEqual(chunks[1].section, "每周维护")

    def test_preamble_becomes_first_chunk(self):
        chunks = split_text(FAULT_TEXT)
        self.assertEqual(chunks[0].mode, "preamble")
        self.assertIn("说明：不同品牌的错误码含义可能不同。", chunks[0].text)
        self.assertEqual(chunks[0].unit, "前言")

    def test_split_documents_attaches_metadata(self):
        document = Document(page_content=QA_TEXT, metadata={"source": "data/qa.txt"})
        chunks = split_documents([document], max_unit_chars=600)

        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertEqual(chunk.metadata["source"], "data/qa.txt")
            self.assertIn("mode", chunk.metadata)
            self.assertIn("unit", chunk.metadata)

    def test_signature_changes_with_params(self):
        base = split_signature("qa_aware", 600, 200, 20, ["\n\n"])
        self.assertEqual(base, split_signature("qa_aware", 600, 200, 20, ["\n\n"]))
        self.assertNotEqual(base, split_signature("qa_aware", 500, 200, 20, ["\n\n"]))
        self.assertNotEqual(base, split_signature("recursive", 600, 200, 20, ["\n\n"]))
        self.assertNotEqual(base, split_signature("qa_aware", 600, 200, 20, ["\n"]))

    def test_real_data_files_lose_no_line(self):
        paths = sorted(glob.glob("data/*.txt"))
        self.assertTrue(paths, "data 目录缺少 txt，测试无法验证覆盖率")

        for path in paths:
            text = Path(path).read_text(encoding="utf-8")
            chunks = split_text(text, max_unit_chars=600)
            joined = "".join("".join(chunk.text.split()) for chunk in chunks)
            missing = [
                line
                for line in text.splitlines()
                if line.strip() and "".join(line.split()) not in joined
            ]
            self.assertEqual(missing, [], f"{path} 有内容没进任何 chunk")


if __name__ == "__main__":
    unittest.main()
