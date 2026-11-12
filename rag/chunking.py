"""知识库切分：按问答对 / 条目块 / 编号清单切，不按固定字数硬切。

现有 `data/*.txt` 有三种结构，硬切会把一个问答或一条错误码劈成两半，
检索时命中了片段却拿不到完整答案：

- 100 问：``【小节】`` + ``Q1：`` / ``A1：`` 成对出现；
- 故障排除：``一、错误代码`` + ``【E01 碰撞传感器异常】`` + 原因/处理；
- 选购/保养：``一、先明确需求`` + ``1. …`` 编号清单。

切分结果带 ``section`` / ``unit`` / ``mode`` 元数据，既给评测的
gold_sections 对齐，也给前端引用溯源用。切分参数会算进 ``split_signature``，
参数一变就强制重建向量库，避免「配置改了、库里还是旧 chunk」。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from langchain_core.documents import Document

# 改切分逻辑时手动 +1，让旧向量库自动失效。
CHUNKING_VERSION = "2"

_HEADING_RE = re.compile(r"^([一二三四五六七八九十百]+)、\s*(.+?)\s*$")
_BLOCK_RE = re.compile(r"^【(.+?)】\s*$")
_QUESTION_RE = re.compile(r"^Q(\d+)\s*[：:]\s*(.*)$")
_ANSWER_RE = re.compile(r"^A(\d+)\s*[：:]\s*(.*)$")
_ITEM_RE = re.compile(r"^\d+\s*[\.、]\s*")


@dataclass(frozen=True)
class Chunk:
    text: str
    section: str
    unit: str
    mode: str


def split_signature(
    mode: str,
    max_unit_chars: int,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> str:
    """把切分参数压成一个短指纹，写进每个 chunk 的元数据。"""
    payload = "|".join(
        [
            CHUNKING_VERSION,
            str(mode),
            str(max_unit_chars),
            str(chunk_size),
            str(chunk_overlap),
            ",".join(separators),
        ]
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def _flush_qa(
    chunks: list[Chunk],
    header: str,
    section: str,
    question_no: str | None,
    question_line: str,
    answer_lines: list[str],
) -> None:
    if question_no is None:
        return
    body = "\n".join([question_line, *answer_lines]).strip()
    if not body:
        return
    # 正文里保留「【小节】」原样，评测的 gold_sections 才可能整串命中。
    text = f"{header}\n{body}" if header else body
    chunks.append(Chunk(text=text, section=section, unit=f"Q{question_no}", mode="qa"))


def _pack_items(
    chunks: list[Chunk],
    heading: str,
    section: str,
    items: list[str],
    max_unit_chars: int,
) -> None:
    """把编号清单按字数上限打包；每个 chunk 都重复标题，保证单独可读。"""
    if not items:
        return
    group: list[str] = []
    first_no = 1
    start_no = 1
    for index, item in enumerate(items, start=1):
        candidate = group + [item]
        text = "\n".join([heading, *candidate]) if heading else "\n".join(candidate)
        if group and len(text) > max_unit_chars:
            chunks.append(
                Chunk(
                    text="\n".join([heading, *group]) if heading else "\n".join(group),
                    section=section,
                    unit=f"{start_no}-{index - 1}",
                    mode="list",
                )
            )
            group = [item]
            start_no = index
        else:
            group = candidate
        first_no = index

    if group:
        unit = str(start_no) if start_no == first_no else f"{start_no}-{first_no}"
        chunks.append(
            Chunk(
                text="\n".join([heading, *group]) if heading else "\n".join(group),
                section=section,
                unit=unit,
                mode="list",
            )
        )


def _pack_paragraphs(
    chunks: list[Chunk],
    heading: str,
    section: str,
    unit: str,
    paragraphs: list[str],
    max_unit_chars: int,
) -> None:
    """非清单正文（错误码的原因/处理）按空行分段打包，标题重复。"""
    group: list[str] = []
    for index, paragraph in enumerate(paragraphs, start=1):
        candidate = group + [paragraph]
        text = "\n".join([heading, *candidate]) if heading else "\n".join(candidate)
        if group and len(text) > max_unit_chars:
            chunks.append(
                Chunk(
                    text="\n".join([heading, *group]) if heading else "\n".join(group),
                    section=section,
                    unit=f"{unit}-{index - 1}",
                    mode="block",
                )
            )
            group = [paragraph]
        else:
            group = candidate
    if group:
        chunks.append(
            Chunk(
                text="\n".join([heading, *group]) if heading else "\n".join(group),
                section=section,
                unit=unit,
                mode="block",
            )
        )


def split_text(text: str, max_unit_chars: int = 600) -> list[Chunk]:
    """把一份知识库文本切成结构完整的 chunk。"""
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    chunks: list[Chunk] = []

    preamble: list[str] = []
    heading = ""  # 最近的「一、xxx」标题行
    heading_section = ""
    block_header = ""  # 最近的「【xxx】」标题行
    block_section = ""

    question_no: str | None = None
    question_line = ""
    answer_lines: list[str] = []
    items: list[str] = []
    paragraphs: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            paragraphs.append("\n".join(paragraph).strip())
            paragraph = []

    def flush_qa() -> None:
        nonlocal question_no, question_line, answer_lines
        _flush_qa(
            chunks,
            block_header,
            block_section,
            question_no,
            question_line,
            answer_lines,
        )
        question_no, question_line, answer_lines = None, "", []

    def flush_items() -> None:
        nonlocal items
        if items:
            _pack_items(chunks, heading, heading_section, items, max_unit_chars)
            items = []

    def flush_block() -> None:
        nonlocal paragraphs
        flush_paragraph()
        if paragraphs:
            title = "\n".join(part for part in (heading, block_header) if part)
            section = heading_section or block_section
            _pack_paragraphs(
                chunks,
                title,
                section,
                block_section or heading_section or "正文",
                paragraphs,
                max_unit_chars,
            )
            paragraphs = []

    def flush_all() -> None:
        flush_qa()
        flush_items()
        flush_block()

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        heading_match = _HEADING_RE.match(stripped)
        block_match = _BLOCK_RE.match(stripped)
        question_match = _QUESTION_RE.match(stripped)

        if heading_match or block_match:
            flush_all()
            if heading_match:
                heading = stripped
                heading_section = heading_match.group(2)
                block_header = ""
                block_section = ""
            else:
                block_header = stripped
                block_section = block_match.group(1)
            continue

        if question_match:
            # 问答文件里，问题行就是一个新单元的边界。
            flush_qa()
            flush_items()
            flush_block()
            question_no = question_match.group(1)
            question_line = stripped
            answer_lines = []
            continue

        if question_no is not None:
            if _ANSWER_RE.match(stripped):
                if stripped:
                    answer_lines.append(stripped)
                continue
            if not stripped:
                flush_qa()
                continue
            # 问题单元里的其它行（多行答案）继续归属当前问答。
            answer_lines.append(stripped)
            continue

        if not stripped:
            if items:
                continue  # 清单内的空行只是排版
            flush_paragraph()
            continue

        if block_header:
            paragraph.append(stripped)
            continue

        if heading:
            if _ITEM_RE.match(stripped):
                items.append(stripped)
            else:
                flush_items()
                paragraph.append(stripped)
            continue

        preamble.append(stripped)

    flush_all()

    if any(part.strip() for part in preamble):
        chunks.insert(
            0,
            Chunk(
                text="\n".join(part for part in preamble if part.strip()),
                section="",
                unit="前言",
                mode="preamble",
            ),
        )
    return chunks


def split_documents(
    documents: list[Document],
    max_unit_chars: int = 600,
) -> list[Document]:
    """把 loader 产出的 Document 切成带元数据的 chunk Document。"""
    result: list[Document] = []
    for document in documents:
        metadata = dict(document.metadata or {})
        for chunk in split_text(document.page_content, max_unit_chars):
            result.append(
                Document(
                    page_content=chunk.text,
                    metadata={
                        **metadata,
                        "section": chunk.section,
                        "unit": chunk.unit,
                        "mode": chunk.mode,
                    },
                )
            )
    return result


__all__ = [
    "CHUNKING_VERSION",
    "Chunk",
    "split_documents",
    "split_signature",
    "split_text",
]
