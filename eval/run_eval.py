"""RAG 检索评测：校验评测集，跑 Recall@k / MRR，输出 JSON 与 Markdown。

用法：
    python eval/run_eval.py --validate-only
    python eval/run_eval.py --limit 10
    python eval/run_eval.py --ks 1,3,5,8
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.metrics import (  # noqa: E402
    first_relevant_rank,
    first_source_rank,
    normalize,
    percentile,
    source_basename,
    summarize_ranks,
)


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no} JSON 解析失败：{exc}") from exc
    return rows


def validate_rows(rows: list[dict], data_dir: Path, records_path: Path) -> list[str]:
    """校验 schema 与 gold 标签是否还能在当前数据里找到。"""
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_questions: dict[str, str] = {}
    text_cache: dict[str, str] = {}
    valid_types = {"answerable", "negative", "report"}

    def read_source(name: str) -> str | None:
        if name not in text_cache:
            path = data_dir / name
            if not path.exists():
                return None
            text_cache[name] = path.read_text(encoding="utf-8")
        return text_cache[name]

    with records_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        record_fields = set(reader.fieldnames or [])
        record_index = {(r["用户ID"], r["时间"]) for r in reader}

    for row in rows:
        item_id = row.get("id", "<missing-id>")
        item_type = row.get("type")
        question = row.get("question", "")

        if item_id in seen_ids:
            errors.append(f"{item_id}: id 重复")
        seen_ids.add(item_id)

        question_key = normalize(question)
        if question_key in seen_questions:
            errors.append(f"{item_id}: 问题与 {seen_questions[question_key]} 重复")
        seen_questions[question_key] = item_id

        if item_type not in valid_types:
            errors.append(f"{item_id}: type 非法：{item_type!r}")
            continue
        if not question.strip():
            errors.append(f"{item_id}: question 为空")
        if not row.get("gold_answer", "").strip():
            errors.append(f"{item_id}: gold_answer 为空")

        if item_type == "answerable":
            sources = row.get("gold_sources") or []
            sections = row.get("gold_sections") or []
            terms = row.get("gold_terms") or []
            if not sources:
                errors.append(f"{item_id}: gold_sources 为空")
            if not sections and not terms:
                errors.append(f"{item_id}: gold_sections 与 gold_terms 至少填一个")

            source_texts = []
            for source in sources:
                text = read_source(source)
                if text is None:
                    errors.append(f"{item_id}: 找不到数据文件 {source}")
                    continue
                source_texts.append(normalize(text))

            # 多来源题只要求「至少一个 gold 来源」包含该章节/关键词。
            for section in sections:
                if not any(normalize(section) in text for text in source_texts):
                    errors.append(f"{item_id}: 章节 {section!r} 不在任何 gold 来源中")
            for term in terms:
                if not any(normalize(term) in text for text in source_texts):
                    errors.append(f"{item_id}: 关键词 {term!r} 不在任何 gold 来源中")

        elif item_type == "negative":
            if row.get("expected") != "refuse":
                errors.append(f"{item_id}: negative 题的 expected 必须是 refuse")

        elif item_type == "report":
            user_id = row.get("user_id")
            month = row.get("month")
            if (user_id, month) not in record_index:
                errors.append(f"{item_id}: records.csv 中找不到 {user_id}/{month}")
            for field in row.get("expected_fields") or []:
                if field not in record_fields:
                    errors.append(f"{item_id}: records.csv 没有字段 {field!r}")

    return errors


def run_retrieval(service, rows: list[dict], ks: list[int], limit: int | None) -> dict:
    items = [row for row in rows if row.get("type") == "answerable"]
    if limit is not None:
        items = items[:limit]
    if not items:
        raise SystemExit("没有可评测的 answerable 题目")

    max_k = max(ks)
    per_question = []
    latencies = []

    for item in items:
        started = time.perf_counter()
        documents = service.retrieve(item["question"], k=max_k)
        latencies.append((time.perf_counter() - started) * 1000)

        span_rank = first_relevant_rank(documents, item)
        source_rank = first_source_rank(documents, item)
        per_question.append(
            {
                "id": item["id"],
                "question": item["question"],
                "span_rank": span_rank,
                "source_rank": source_rank,
                "top_sources": [
                    source_basename(doc.metadata.get("source", "")) for doc in documents
                ],
            }
        )

    span_metrics = summarize_ranks([row["span_rank"] for row in per_question], ks)
    source_metrics = summarize_ranks([row["source_rank"] for row in per_question], ks)

    return {
        "evaluated": len(items),
        "ks": ks,
        "metrics": {"span": span_metrics, "source": source_metrics},
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2),
            "mean": round(mean(latencies), 2),
        },
        "per_question": per_question,
    }


def to_markdown(result: dict, rows: list[dict], meta: dict) -> str:
    counts = Counter(row["type"] for row in rows)
    ks = result["ks"]
    header = "| 指标 | " + " | ".join(f"@{k}" for k in ks) + " |"
    divider = "|---|" + "---|" * len(ks)
    lines = [
        "# 检索评测基线",
        "",
        f"- 生成时间：{meta['generated_at']}",
        f"- 向量库：`{meta['collection']}` / {meta['chunk_count']} chunks",
        f"- Embedding：`{meta['embedding_model']}`，Chat：`{meta['chat_model']}`",
        f"- 数据集：{len(rows)} 题（answerable {counts['answerable']} / negative {counts['negative']} / report {counts['report']}）",
        f"- 本次评测：{result['evaluated']} 道 answerable，检索上限 k={max(ks)}",
        "",
        "## 指标",
        "",
        header,
        divider,
    ]

    for label, key in [
        ("Span Recall@k（命中答案段落）", "recall"),
        ("Span MRR@k", "mrr"),
    ]:
        values = [f"{result['metrics']['span'][f'{key}@{k}']:.2%}" for k in ks]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    values = [f"{result['metrics']['source'][f'recall@{k}']:.2%}" for k in ks]
    lines.append("| Source Recall@k（只命中文件） | " + " | ".join(values) + " |")

    lines += [
        "",
        f"- 检索延迟：P50 {result['latency_ms']['p50']} ms / "
        f"P95 {result['latency_ms']['p95']} ms / 均值 {result['latency_ms']['mean']} ms",
        "",
        "> 每题只有一个 gold 答案集合，因此 Recall@k 等价于 hit rate；"
        "Span 指标要求文件名和答案关键词同时命中，Source 指标只要求文件名命中。",
        "",
        "## 未命中或非首位命中的题",
        "",
        "| ID | 问题 | 首次命中排名 | Top 来源 |",
        "|---|---|---|---|",
    ]
    misses = [
        row for row in result["per_question"]
        if row["span_rank"] is None or row["span_rank"] > 1
    ]
    for row in misses[:15]:
        rank = "未命中" if row["span_rank"] is None else row["span_rank"]
        sources = "<br>".join(row["top_sources"])
        lines.append(f"| {row['id']} | {row['question']} | {rank} | {sources} |")
    if not misses:
        lines.append("| - | 全部在首位命中 | - | - |")
    if len(misses) > 15:
        lines.append(f"| ... | 其余 {len(misses) - 15} 题见 JSON | - | - |")

    lines += [
        "",
        "## 复现",
        "",
        "```bash",
        "python eval/run_eval.py --ks 1,3,5,8",
        "```",
        "",
        "> 注意：当前题目多来自知识库原始问句，属于 easy split；"
        "hard split（改写问法、跨节组合）尚未加入。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 检索评测")
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--ks", default="1,3,5,8")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 道 answerable")
    parser.add_argument("--out", default="eval/results/baseline.json")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    dataset_path = (ROOT / args.dataset).resolve()
    out_path = (ROOT / args.out).resolve()
    data_dir = ROOT / "data"
    records_path = ROOT / "data" / "external" / "records.csv"

    rows = load_rows(dataset_path)
    errors = validate_rows(rows, data_dir, records_path)
    if errors:
        print(f"VALIDATION_FAILED count={len(errors)}")
        for error in errors[:30]:
            print(f"- {error}")
        return 1
    print(f"VALIDATION_OK rows={len(rows)}")
    if args.validate_only:
        return 0

    ks = sorted({int(value) for value in args.ks.split(",") if value.strip()})
    if not ks or min(ks) < 1:
        raise SystemExit("--ks 必须是正整数，例如 1,3,5,8")

    from rag.rag_service import RagSummarizeService
    from utils.config_handler import chroma_conf, rag_conf

    service = RagSummarizeService()
    service.ensure_knowledge_base()
    chunk_count = service.vector_store.count()

    result = run_retrieval(service, rows, ks, args.limit)
    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "collection": chroma_conf["collection_name"],
        "chunk_count": chunk_count,
        "embedding_model": rag_conf["embedding_model_name"],
        "chat_model": rag_conf["chat_model_name"],
    }
    payload = {
        "meta": meta,
        "dataset": str(dataset_path.relative_to(ROOT)),
        "counts": dict(Counter(row["type"] for row in rows)),
        **result,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = out_path.with_suffix(".md")
    markdown_path.write_text(to_markdown(result, rows, meta), encoding="utf-8")

    span = result["metrics"]["span"]
    print(
        " ".join(
            f"span_recall@{k}={span[f'recall@{k}']:.4f}" for k in ks
        )
    )
    print(f"span_mrr@1={span['mrr@1']:.4f} evaluated={result['evaluated']}")
    print(f"wrote {out_path.relative_to(ROOT)} and {markdown_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
