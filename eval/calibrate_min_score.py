"""标定「无证据兜底」用的重排分数阈值 min_score。

阈值不能拍脑袋定：定高了会把正常问题的证据一起砍掉，定低了等于没兜底。
这里把数据集里的问题全部过一遍真实检索管线，按 answerable / negative / report
分组统计每题的最高重排分，再在所有观测到的分数里挑一个能把两组分开的阈值。

用法：
    python eval/calibrate_min_score.py
    python eval/calibrate_min_score.py --datasets eval/dataset_hard.jsonl --out eval/results/x.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.metrics import percentile  # noqa: E402
from agent.scope_router import is_out_of_scope_query  # noqa: E402
from rag.rag_service import RagSummarizeService  # noqa: E402

DEFAULT_DATASETS = ["eval/dataset.jsonl", "eval/dataset_hard.jsonl"]
GROUPS = ("answerable", "negative", "report")


def load_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item["_dataset"] = Path(path).name
                rows.append(item)
    return rows


def group_of(item: dict) -> str:
    kind = item.get("type") or item.get("kind") or "answerable"
    return kind if kind in GROUPS else "answerable"


def collect(service: RagSummarizeService, rows: list[dict], k: int) -> list[dict]:
    records = []
    for item in rows:
        question = item["question"]
        _, stats = service.retrieve_with_stats(question, k=k)
        records.append(
            {
                "id": item.get("id"),
                "dataset": item["_dataset"],
                "group": group_of(item),
                "question": question,
                "top_score": stats.get("rerank_top_score", 0.0),
                "kept": stats.get("rerank_kept", 0),
                "pool": stats.get("rerank_pool", 0),
                "rerank": stats.get("rerank"),
                "latency_ms": stats.get("rerank_latency_ms", 0.0),
                # 已经被服务范围路由拦下的 negative 根本走不到「无证据兜底」，
                # 不能拿它们的分数来定阈值，否则会把阈值抬高误伤正常问题。
                "out_of_scope": is_out_of_scope_query(question),
            }
        )
        print(
            f"[{records[-1]['group']:>10}] {records[-1]['id']} "
            f"top={records[-1]['top_score']:.4f} kept={records[-1]['kept']} "
            f"scope_blocked={records[-1]['out_of_scope']}"
        )
    return records


def group_scores(records: list[dict]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {name: [] for name in GROUPS}
    for record in records:
        grouped.setdefault(record["group"], []).append(record["top_score"])
    return grouped


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 4),
        "p5": round(percentile(ordered, 5), 4),
        "p50": round(percentile(ordered, 50), 4),
        "max": round(ordered[-1], 4),
    }


def pick_threshold(records: list[dict], keep_ratio: float = 1.0) -> dict:
    """在所有观测分数里挑阈值：先要求 answerable 全保住，再看能拒掉多少 negative。

    只用「没被服务范围路由拦下」的 negative 参与计算：被拦下的那些根本走不到
    无证据兜底，把它们的高分算进来只会把阈值抬到误伤正常问题的位置。
    """
    answerable = [r["top_score"] for r in records if r["group"] == "answerable"]
    negatives = [r for r in records if r["group"] == "negative"]
    negative = [r["top_score"] for r in negatives if not r["out_of_scope"]]
    blocked = len(negatives) - len(negative)
    if not answerable or not negative:
        return {
            "threshold": None,
            "reason": "缺少 answerable 或未被路由拦下的 negative 样本",
            "scope_blocked_negatives": blocked,
        }

    candidates = sorted({round(score, 4) for score in [*answerable, *negative]})
    best = None
    for threshold in candidates:
        kept = sum(1 for score in answerable if score >= threshold) / len(answerable)
        if kept < keep_ratio:
            continue
        refused = sum(1 for score in negative if score < threshold) / len(negative)
        if best is None or refused > best["negative_refused"]:
            best = {
                "threshold": threshold,
                "answerable_kept": round(kept, 4),
                "negative_refused": round(refused, 4),
            }
    if best is None:
        return {
            "threshold": None,
            "reason": f"没有阈值能在保住 {keep_ratio:.0%} answerable 的前提下拒掉 negative",
            "answerable_min": round(min(answerable), 4),
            "negative_max": round(max(negative), 4),
            "scope_blocked_negatives": blocked,
        }

    answerable_min = round(min(answerable), 4)
    negative_max = round(max(negative), 4)
    best["answerable_min"] = answerable_min
    best["negative_max"] = negative_max
    best["negative_samples"] = len(negative)
    best["scope_blocked_negatives"] = blocked
    best["separable"] = negative_max < answerable_min
    # 两组能分开时不贴着 answerable 的最低分取值，改用间隔中点，
    # 给没进评测集的正常问题留余量，避免阈值一抖动就把证据砍掉。
    suggested = (
        round((negative_max + answerable_min) / 2, 4)
        if best["separable"]
        else best["threshold"]
    )
    best["suggested"] = suggested
    best["answerable_kept_at_suggested"] = round(
        sum(1 for score in answerable if score >= suggested) / len(answerable), 4
    )
    best["negative_refused_at_suggested"] = round(
        sum(1 for score in negative if score < suggested) / len(negative), 4
    )
    return best


def render_markdown(payload: dict) -> str:
    lines = [
        "# min_score 阈值标定",
        "",
        f"- 生成时间：{payload['meta']['generated_at']}",
        f"- 数据集：{', '.join(payload['meta']['datasets'])}",
        f"- 重排模型：`{payload['meta']['rerank_model']}`，检索 k={payload['meta']['k']}",
        f"- 样本数：{payload['meta']['rows']}",
        "",
        "## 各组最高重排分分布",
        "",
        "| 分组 | 样本 | 最小 | P5 | P50 | 最大 |",
        "|---|---|---|---|---|---|",
    ]
    for group, stats in payload["groups"].items():
        if not stats.get("count"):
            continue
        lines.append(
            f"| {group} | {stats['count']} | {stats['min']} | {stats['p5']} | "
            f"{stats['p50']} | {stats['max']} |"
        )

    suggestion = payload["suggestion"]
    lines += ["", "## 阈值建议", ""]
    if suggestion.get("threshold") is None:
        lines.append(f"- 无法自动分离：{suggestion.get('reason')}")
    else:
        lines += [
            f"- 建议 `min_score = {suggestion['suggested']}`（分离边界 {suggestion['threshold']}，两组可分时取间隔中点留余量）",
            f"- answerable 保留率：{suggestion['answerable_kept_at_suggested']:.2%}",
            f"- 未被路由拦下的 negative 拒答率：{suggestion['negative_refused_at_suggested']:.2%}"
            f"（样本 {suggestion['negative_samples']} 条）",
            f"- 已被服务范围路由拦下、不参与阈值计算的 negative：{suggestion['scope_blocked_negatives']} 条",
            f"- 两组是否完全可分：{'是' if suggestion.get('separable') else '否（有重叠，阈值只是折中）'}",
            f"- answerable 最低分 {suggestion['answerable_min']} / negative 最高分 {suggestion['negative_max']}",
        ]

    lines += ["", "## 边界样本（分数最低的 answerable 与分数最高的 negative）", ""]
    lines.append("| 分组 | ID | 最高重排分 | 路由已拦 | 问题 |")
    lines.append("|---|---|---|---|---|")
    for record in payload["edge_cases"]:
        blocked = "是" if record.get("out_of_scope") else "否"
        lines.append(
            f"| {record['group']} | {record['id']} | {record['top_score']:.4f} | "
            f"{blocked} | {record['question']} |"
        )

    lines += [
        "",
        "## 复现",
        "",
        "```bash",
        "python eval/calibrate_min_score.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="标定重排 min_score 阈值")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", default="eval/results/min_score_calibration.json")
    args = parser.parse_args()

    rows = load_rows(args.datasets)
    service = RagSummarizeService()
    service.ensure_knowledge_base()

    records = collect(service, rows, args.k)
    grouped = group_scores(records)
    suggestion = pick_threshold(records)

    answerable_sorted = sorted(
        (r for r in records if r["group"] == "answerable"), key=lambda r: r["top_score"]
    )[:5]
    negatives = [r for r in records if r["group"] == "negative"]
    # 阈值只对「路由没拦下」的 negative 生效，所以边界样本也按这批来看；
    # 另附分数最高的被拦样本，说明为什么它们不该参与计算。
    in_scope_negative = sorted(
        (r for r in negatives if not r["out_of_scope"]), key=lambda r: -r["top_score"]
    )[:5]
    blocked_negative = sorted(
        (r for r in negatives if r["out_of_scope"]), key=lambda r: -r["top_score"]
    )[:3]

    payload = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "datasets": args.datasets,
            "rerank_model": service.pipeline.reranker.model
            if service.pipeline.reranker
            else None,
            "k": args.k,
            "rows": len(records),
            "chunks": service.vector_store.count(),
        },
        "groups": {name: summarize(values) for name, values in grouped.items()},
        "suggestion": suggestion,
        "edge_cases": [
            {
                k: r[k]
                for k in ("group", "id", "question", "top_score", "kept", "out_of_scope")
            }
            for r in [*answerable_sorted, *in_scope_negative, *blocked_negative]
        ],
        "records": records,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    out_path.with_suffix(".md").write_text(render_markdown(payload), encoding="utf-8")

    print(f"suggestion={json.dumps(suggestion, ensure_ascii=False)}")
    for name, stats in payload["groups"].items():
        print(f"group={name} {json.dumps(stats, ensure_ascii=False)}")
    print(f"wrote {out_path} and {out_path.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
