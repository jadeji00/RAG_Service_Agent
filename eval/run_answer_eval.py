"""回答级评测：给 80 题生成回答并判分，输出 JSON 与 Markdown 基线。

三条链路分开跑，避免用同一种「满意」口径糊在一起：
- answerable：RAG 检索 + 生成，评 correctness / faithfulness；
- negative：超范围问题，评是否拒答；
- report：确定性报告子流程（路由 → fetch_external_data → 一次写作），
  评字段完整性、数值忠实度与建议质量。

``--repeat k`` 会为每道题独立生成并判分 k 次，额外输出严格 pass^k、
至少一次通过和波动题；默认 k=1，行为与旧版一致。

用法：
    python eval/run_answer_eval.py --validate-only
    python eval/run_answer_eval.py --types answerable,negative --limit 2
    python eval/run_answer_eval.py --types report --limit 1
    python eval/run_answer_eval.py --repeat 3
    python eval/run_answer_eval.py
"""

from __future__ import annotations

import argparse
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

from eval.judge import (  # noqa: E402
    JudgeModel,
    load_records,
    refusal_cue,
    score_report_facts,
)
from eval.metrics import normalize, percentile, source_basename  # noqa: E402
from eval.run_eval import load_rows, validate_rows  # noqa: E402


def _mean(values) -> float | None:
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def _rate(values, predicate) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(1 for value in values if predicate(value)) / len(values)


def _distribution(values) -> dict[str, int]:
    return {str(score): sum(1 for value in values if value == score) for score in (2, 1, 0)}


def _latency_summary(values) -> dict:
    values = [value for value in values if value is not None]
    if not values:
        return {"p50": None, "p95": None, "mean": None}
    return {
        "p50": round(percentile(values, 50), 2),
        "p95": round(percentile(values, 95), 2),
        "mean": round(mean(values), 2),
    }


def _run_answerable_once(service, judge: JudgeModel, item: dict, k: int) -> dict:
    started = time.perf_counter()
    try:
        documents = service.retrieve(item["question"], k=k)
        answer = (service.answer(item["question"], documents) or "").strip()
        gen_ms = (time.perf_counter() - started) * 1000
    except Exception as exc:
        return {
            "id": item["id"],
            "type": "answerable",
            "question": item["question"],
            "answer": "",
            "sources": [],
            "verdict": {},
            "passed": False,
            "gen_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "judge_latency_ms": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    context = service.build_context(documents)
    sources = [source_basename(doc.metadata.get("source", "")) for doc in documents]

    if answer:
        judge_started = time.perf_counter()
        verdict = judge.judge_answerable(
            question=item["question"],
            gold_answer=item.get("gold_answer", ""),
            gold_terms=item.get("gold_terms") or [],
            context=context,
            answer=answer,
        )
        judge_ms = (time.perf_counter() - judge_started) * 1000
    else:
        verdict = {
            "correctness": 0,
            "faithfulness": 0,
            "reason": "empty answer",
            "error": "empty_answer",
        }
        judge_ms = None

    terms = item.get("gold_terms") or []
    answer_normalized = normalize(answer)
    term_rate = None
    if terms:
        term_rate = sum(
            1 for term in terms if normalize(term) in answer_normalized
        ) / len(terms)

    correctness = verdict.get("correctness")
    faithfulness = verdict.get("faithfulness")
    passed = bool(
        correctness is not None
        and correctness >= 1
        and faithfulness is not None
        and faithfulness >= 1
    )

    return {
        "id": item["id"],
        "type": "answerable",
        "question": item["question"],
        "answer": answer,
        "sources": sources,
        "verdict": verdict,
        "term_hit_rate": term_rate,
        "passed": passed,
        "gen_latency_ms": round(gen_ms, 2),
        "judge_latency_ms": round(judge_ms, 2) if judge_ms is not None else None,
        "error": "",
    }


def _run_summary(run: dict) -> dict:
    verdict = run.get("verdict", {})
    return {
        "passed": bool(run.get("passed")),
        "correctness": verdict.get("correctness"),
        "faithfulness": verdict.get("faithfulness"),
        "gen_latency_ms": run.get("gen_latency_ms"),
        "error": run.get("error", ""),
    }


def _attach_runs(first: dict, runs: list[dict], repeat: int) -> dict:
    if repeat <= 1:
        return first
    result = dict(first)
    result["runs"] = [
        {**_run_summary(run), "answer": run.get("answer", "")} for run in runs
    ]
    result["pass_count"] = sum(1 for run in runs if run.get("passed"))
    result["passed_all"] = result["pass_count"] == repeat
    result["passed_any"] = result["pass_count"] > 0
    return result


def run_answerable(
    service,
    judge: JudgeModel,
    rows: list[dict],
    k: int,
    limit: int | None,
    repeat: int = 1,
) -> list[dict]:
    items = [row for row in rows if row.get("type") == "answerable"]
    if limit is not None:
        items = items[:limit]

    details = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        runs = [_run_answerable_once(service, judge, item, k) for _ in range(repeat)]
        first = _attach_runs(runs[0], runs, repeat)
        details.append(first)
        verdict = first.get("verdict", {})
        suffix = f" runs={first.get('pass_count')}/{repeat}" if repeat > 1 else ""
        print(
            f"[{index}/{total}] {item['id']} "
            f"corr={verdict.get('correctness')} "
            f"faith={verdict.get('faithfulness')} "
            f"pass={int(bool(first.get('passed')))}{suffix}"
        )

    return details


def _run_negative_once(service, judge: JudgeModel, item: dict, k: int) -> dict:
    started = time.perf_counter()
    try:
        documents = service.retrieve(item["question"], k=k)
        answer = (service.answer(item["question"], documents) or "").strip()
        gen_ms = (time.perf_counter() - started) * 1000
    except Exception as exc:
        return {
            "id": item["id"],
            "type": "negative",
            "question": item["question"],
            "answer": "",
            "sources": [],
            "verdict": {},
            "cue": False,
            "passed": False,
            "gen_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "judge_latency_ms": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    sources = [source_basename(doc.metadata.get("source", "")) for doc in documents]
    if answer:
        judge_started = time.perf_counter()
        verdict = judge.judge_negative(item["question"], answer)
        judge_ms = (time.perf_counter() - judge_started) * 1000
    else:
        verdict = {"refusal": 0, "reason": "empty answer", "error": "empty_answer"}
        judge_ms = None

    refusal = verdict.get("refusal")
    return {
        "id": item["id"],
        "type": "negative",
        "question": item["question"],
        "answer": answer,
        "sources": sources,
        "verdict": verdict,
        "cue": refusal_cue(answer),
        "passed": refusal == 1,
        "gen_latency_ms": round(gen_ms, 2),
        "judge_latency_ms": round(judge_ms, 2) if judge_ms is not None else None,
        "error": "",
    }


def run_negative(
    service,
    judge: JudgeModel,
    rows: list[dict],
    k: int,
    limit: int | None,
    repeat: int = 1,
) -> list[dict]:
    items = [row for row in rows if row.get("type") == "negative"]
    if limit is not None:
        items = items[:limit]

    details = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        runs = [_run_negative_once(service, judge, item, k) for _ in range(repeat)]
        first = _attach_runs(runs[0], runs, repeat)
        details.append(first)
        suffix = f" runs={first.get('pass_count')}/{repeat}" if repeat > 1 else ""
        print(
            f"[{index}/{total}] {item['id']} "
            f"refusal={first.get('verdict', {}).get('refusal')} "
            f"pass={int(bool(first.get('passed')))}{suffix}"
        )

    return details


def _final_answer(messages) -> str:
    for message in reversed(messages):
        if getattr(message, "type", "") != "ai":
            continue
        if getattr(message, "tool_calls", None):
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _messages_tool_calls(messages) -> list[dict]:
    tool_calls = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            tool_calls.append(
                {
                    "name": call.get("name", ""),
                    "args": call.get("args", {}),
                }
            )
    return tool_calls


def _run_report_once(
    agent,
    judge: JudgeModel,
    item: dict,
    record: dict,
    run_id: str,
    run_index: int,
) -> dict:
    from agent.report_pipeline import is_report_query

    user_id = str(item.get("user_id", ""))
    month = str(item.get("month", ""))
    expected_fields = item.get("expected_fields") or []
    started = time.perf_counter()
    routed = is_report_query(item["question"])

    try:
        if routed:
            result = agent.generate_report(
                item["question"], user_id=user_id, month=month
            )
            answer = str(result.get("answer", "")).strip()
            tool_calls = result.get("tool_calls", [])
            route = result.get("route", "report_pipeline")
            pipeline_ok = True
            fetch_args_ok = bool(result.get("fetch_args_ok"))
        else:
            thread_id = f"answer-eval-{run_id}-{item['id']}-r{run_index}"
            result = agent.agent.invoke(
                {"messages": [{"role": "user", "content": item["question"]}]},
                config={
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": 20,
                },
                context={"report": False},
            )
            messages = result.get("messages", []) if isinstance(result, dict) else []
            answer = _final_answer(messages)
            tool_calls = _messages_tool_calls(messages)
            route = "react_fallback"
            pipeline_ok = False
            fetch_args_ok = any(
                call["name"] == "fetch_external_data"
                and str(call["args"].get("user_id", "")).strip() == user_id
                and str(call["args"].get("month", "")).strip() == month
                for call in tool_calls
            )
        gen_ms = (time.perf_counter() - started) * 1000
    except Exception as exc:
        return {
            "id": item["id"],
            "type": "report",
            "question": item["question"],
            "user_id": user_id,
            "month": month,
            "expected_fields": expected_fields,
            "answer": "",
            "tool_calls": [],
            "route": "report_pipeline" if routed else "react_fallback",
            "pipeline_ok": False,
            "fetch_args_ok": False,
            "facts": {},
            "verdict": {},
            "passed": False,
            "gen_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "judge_latency_ms": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    facts = score_report_facts(answer, record, expected_fields)
    if answer:
        judge_started = time.perf_counter()
        verdict = judge.judge_report(item["question"], record, answer)
        judge_ms = (time.perf_counter() - judge_started) * 1000
    else:
        verdict = {
            "faithfulness": 0,
            "advice": 0,
            "reason": "empty answer",
            "error": "empty_answer",
        }
        judge_ms = None

    faithfulness = verdict.get("faithfulness")
    passed = bool(
        answer
        and facts.get("complete")
        and faithfulness is not None
        and faithfulness >= 1
    )
    return {
        "id": item["id"],
        "type": "report",
        "question": item["question"],
        "user_id": user_id,
        "month": month,
        "expected_fields": expected_fields,
        "answer": answer,
        "tool_calls": tool_calls,
        "route": route,
        "pipeline_ok": pipeline_ok,
        "fetch_args_ok": fetch_args_ok,
        "facts": facts,
        "verdict": verdict,
        "passed": passed,
        "gen_latency_ms": round(gen_ms, 2),
        "judge_latency_ms": round(judge_ms, 2) if judge_ms is not None else None,
        "error": "",
    }


def run_report(
    agent,
    judge: JudgeModel,
    rows: list[dict],
    records: dict,
    limit: int | None,
    run_id: str,
    repeat: int = 1,
) -> list[dict]:
    items = [row for row in rows if row.get("type") == "report"]
    if limit is not None:
        items = items[:limit]

    details = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        user_id = str(item.get("user_id", ""))
        month = str(item.get("month", ""))
        record = records.get((user_id, month), {})
        runs = [
            _run_report_once(agent, judge, item, record, run_id, run_index)
            for run_index in range(1, repeat + 1)
        ]
        first = _attach_runs(runs[0], runs, repeat)
        details.append(first)

        facts = first.get("facts", {})
        covered = sum(
            1 for info in facts.get("fields", {}).values() if info.get("covered")
        )
        suffix = f" runs={first.get('pass_count')}/{repeat}" if repeat > 1 else ""
        print(
            f"[{index}/{total}] {item['id']} fields={covered}/{len(first.get('expected_fields') or [])} "
            f"faith={first.get('verdict', {}).get('faithfulness')} "
            f"route={first.get('route')} "
            f"fetch_args={int(bool(first.get('fetch_args_ok')))} "
            f"pass={int(bool(first.get('passed')))}{suffix}"
        )

    return details


def _passk_stats(rows: list[dict], repeat: int) -> dict | None:
    if repeat <= 1 or not rows:
        return None
    pass_counts = [int(row.get("pass_count", 0)) for row in rows]
    total = len(rows)
    return {
        "questions": total,
        "strict_pass_rate": sum(1 for count in pass_counts if count == repeat) / total,
        "at_least_once_rate": sum(1 for count in pass_counts if count > 0) / total,
        "mean_run_pass_rate": sum(pass_counts) / (total * repeat),
        "flaky_count": sum(1 for count in pass_counts if 0 < count < repeat),
        "flaky_ids": [
            row["id"]
            for row in rows
            if 0 < int(row.get("pass_count", 0)) < repeat
        ],
    }


def _passk_block(
    answerable: list[dict],
    negative: list[dict],
    report: list[dict],
    repeat: int,
) -> dict | None:
    if repeat <= 1:
        return None
    types = {
        "answerable": answerable,
        "negative": negative,
        "report": report,
        "overall": answerable + negative + report,
    }
    return {
        "k": repeat,
        "types": {
            label: _passk_stats(rows, repeat)
            for label, rows in types.items()
            if rows
        },
    }


def summarize(
    answerable: list[dict],
    negative: list[dict],
    report: list[dict],
    repeat: int = 1,
) -> dict:
    correctness = [
        row["verdict"].get("correctness")
        for row in answerable
        if row.get("verdict", {}).get("correctness") is not None
    ]
    answer_faithfulness = [
        row["verdict"].get("faithfulness")
        for row in answerable
        if row.get("verdict", {}).get("faithfulness") is not None
    ]
    term_rates = [
        row["term_hit_rate"]
        for row in answerable
        if row.get("term_hit_rate") is not None
    ]

    answerable_summary = {
        "evaluated": len(answerable),
        "scored": len(correctness),
        "correct_rate": _rate(correctness, lambda value: value == 2),
        "acceptable_rate": _rate(correctness, lambda value: value >= 1),
        "mean_correctness": _mean(correctness),
        "correctness_distribution": _distribution(correctness),
        "grounded_rate": _rate(answer_faithfulness, lambda value: value == 2),
        "faithful_rate": _rate(answer_faithfulness, lambda value: value >= 1),
        "hallucination_rate": _rate(answer_faithfulness, lambda value: value == 0),
        "faithfulness_distribution": _distribution(answer_faithfulness),
        "mean_term_hit_rate": _mean(term_rates),
        "pass_rate": _rate([row["passed"] for row in answerable], bool),
        "gen_latency_ms": _latency_summary([row.get("gen_latency_ms") for row in answerable]),
        "judge_latency_ms": _latency_summary([row.get("judge_latency_ms") for row in answerable]),
        "errors": sum(1 for row in answerable if row.get("error")),
        "judge_errors": sum(
            1 for row in answerable if row.get("verdict", {}).get("error")
        ),
    }

    refusals = [
        row["verdict"].get("refusal")
        for row in negative
        if row.get("verdict", {}).get("refusal") is not None
    ]
    negative_summary = {
        "evaluated": len(negative),
        "scored": len(refusals),
        "refusal_accuracy": _rate(refusals, lambda value: value == 1),
        "cue_rate": _rate([row.get("cue") for row in negative], bool),
        "pass_rate": _rate([row["passed"] for row in negative], bool),
        "gen_latency_ms": _latency_summary([row.get("gen_latency_ms") for row in negative]),
        "judge_latency_ms": _latency_summary([row.get("judge_latency_ms") for row in negative]),
        "errors": sum(1 for row in negative if row.get("error")),
        "judge_errors": sum(
            1 for row in negative if row.get("verdict", {}).get("error")
        ),
    }

    field_names: list[str] = []
    for row in report:
        for field in row.get("expected_fields") or []:
            if field not in field_names:
                field_names.append(field)

    field_coverage = {}
    for field in field_names:
        covered = sum(
            1
            for row in report
            if row.get("facts", {}).get("fields", {}).get(field, {}).get("covered")
        )
        field_coverage[field] = {
            "covered": covered,
            "total": len(report),
            "rate": covered / len(report) if report else None,
        }

    report_faithfulness = [
        row["verdict"].get("faithfulness")
        for row in report
        if row.get("verdict", {}).get("faithfulness") is not None
    ]
    report_advice = [
        row["verdict"].get("advice")
        for row in report
        if row.get("verdict", {}).get("advice") is not None
    ]
    report_summary = {
        "evaluated": len(report),
        "scored": len(report_faithfulness),
        "complete_rate": _rate(
            [row.get("facts", {}).get("complete") for row in report], bool
        ),
        "field_coverage": field_coverage,
        "mean_field_coverage": _mean(
            [row.get("facts", {}).get("field_coverage") for row in report]
        ),
        "mean_fact_coverage": _mean(
            [row.get("facts", {}).get("fact_coverage") for row in report]
        ),
        "faithful_rate": _rate(report_faithfulness, lambda value: value >= 1),
        "grounded_rate": _rate(report_faithfulness, lambda value: value == 2),
        "advice_rate": _rate(report_advice, lambda value: value == 1),
        "pipeline_ok_rate": _rate([row.get("pipeline_ok") for row in report], bool),
        "fetch_args_ok_rate": _rate([row.get("fetch_args_ok") for row in report], bool),
        "pass_rate": _rate([row["passed"] for row in report], bool),
        "gen_latency_ms": _latency_summary([row.get("gen_latency_ms") for row in report]),
        "judge_latency_ms": _latency_summary([row.get("judge_latency_ms") for row in report]),
        "errors": sum(1 for row in report if row.get("error")),
        "judge_errors": sum(1 for row in report if row.get("verdict", {}).get("error")),
    }

    all_rows = answerable + negative + report
    overall = {
        "evaluated": len(all_rows),
        "passed": sum(1 for row in all_rows if row.get("passed")),
        "pass_rate": _rate([row["passed"] for row in all_rows], bool),
        "errors": sum(1 for row in all_rows if row.get("error")),
        "judge_errors": sum(1 for row in all_rows if row.get("verdict", {}).get("error")),
    }
    return {
        "answerable": answerable_summary,
        "negative": negative_summary,
        "report": report_summary,
        "overall": overall,
        "pass_k": _passk_block(answerable, negative, report, repeat),
    }


def _pct(value) -> str:
    return "-" if value is None else f"{value:.2%}"


def _ms(value) -> str:
    return "-" if value is None else f"{value:.0f}"


def _cell(value) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def _failure_rows(payload: dict) -> list[tuple[str, str, str, str, str]]:
    rows = []
    for item in payload["details"]["answerable"]:
        if item["passed"]:
            continue
        verdict = item.get("verdict", {})
        score = f"correctness={verdict.get('correctness')}, faithfulness={verdict.get('faithfulness')}"
        reason = verdict.get("reason") or item.get("error") or item.get("answer", "")[:120]
        rows.append((item["id"], "answerable", item["question"], score, reason))

    for item in payload["details"]["negative"]:
        if item["passed"]:
            continue
        verdict = item.get("verdict", {})
        score = f"refusal={verdict.get('refusal')}"
        reason = verdict.get("reason") or item.get("error") or item.get("answer", "")[:120]
        rows.append((item["id"], "negative", item["question"], score, reason))

    for item in payload["details"]["report"]:
        if item["passed"]:
            continue
        fields = item.get("facts", {}).get("fields", {})
        missing = [field for field, info in fields.items() if not info.get("covered")]
        verdict = item.get("verdict", {})
        score = (
            f"missing={','.join(missing) or '-'}, "
            f"faithfulness={verdict.get('faithfulness')}"
        )
        reason = verdict.get("reason") or item.get("error") or item.get("answer", "")[:120]
        rows.append((item["id"], "report", item["question"], score, reason))
    return rows


def to_markdown(payload: dict) -> str:
    meta = payload["meta"]
    summary = payload["summary"]
    answerable = summary["answerable"]
    negative = summary["negative"]
    report = summary["report"]
    overall = summary["overall"]

    counts = meta["counts"]
    lines = [
        "# 回答级评测基线",
        "",
        f"- 生成时间：{meta['generated_at']}",
        f"- 数据集：`{meta['dataset']}`（answerable {counts.get('answerable', 0)} / "
        f"negative {counts.get('negative', 0)} / report {counts.get('report', 0)}）",
        f"- 生成模型：`{meta['chat_model']}`；判分模型：`{meta['judge_model']}`",
        f"- RAG 检索：Top-{meta['k']}；向量库：`{meta['collection']}` / "
        f"{meta['chunk_count'] if meta['chunk_count'] is not None else '-'} chunks",
        "- 报告链路：确定性报告子流程（路由 → `fetch_external_data` → 一次写作；"
        "未命中路由时回退 ReAct）",
        f"- 重复次数：{meta.get('repeat', 1)}（>1 时计算 pass^k）",
        "",
        "## 总览",
        "",
        "| 类型 | 样本 | 通过率 | 关键指标 |",
        "|---|---|---|---|",
        f"| answerable | {answerable['evaluated']} | {_pct(answerable['pass_rate'])} | "
        f"完全正确 {_pct(answerable['correct_rate'])}；完全有据 {_pct(answerable['grounded_rate'])}；"
        f"幻觉 {_pct(answerable['hallucination_rate'])} |",
        f"| negative | {negative['evaluated']} | {_pct(negative['pass_rate'])} | "
        f"拒答正确率 {_pct(negative['refusal_accuracy'])} |",
        f"| report | {report['evaluated']} | {_pct(report['pass_rate'])} | "
        f"字段完整 {_pct(report['complete_rate'])}；忠实 {_pct(report['faithful_rate'])} |",
        f"| **合计** | **{overall['evaluated']}** | **{_pct(overall['pass_rate'])}** | "
        f"- |",
        "",
        "> 通过口径：answerable = correctness ≥ 1 且 faithfulness ≥ 1；"
        "negative = 明确拒答；report = 期望字段全部覆盖且 faithfulness ≥ 1。",
        "",
        "## 知识问答",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 参与判分 | {answerable['scored']} / {answerable['evaluated']} |",
        f"| correctness=2（完全正确） | {answerable['correctness_distribution']['2']} 题 |",
        f"| correctness=1（部分正确） | {answerable['correctness_distribution']['1']} 题 |",
        f"| correctness=0（错误） | {answerable['correctness_distribution']['0']} 题 |",
        f"| 平均 correctness | "
        f"{answerable['mean_correctness']:.2f} |"
        if answerable["mean_correctness"] is not None
        else "| 平均 correctness | - |",
        f"| faithfulness=2（全部有据） | {answerable['faithfulness_distribution']['2']} 题 |",
        f"| faithfulness=1（少量推断） | {answerable['faithfulness_distribution']['1']} 题 |",
        f"| faithfulness=0（编造） | {answerable['faithfulness_distribution']['0']} 题 |",
        f"| 关键词命中率（诊断，不计分） | {_pct(answerable['mean_term_hit_rate'])} |",
        "",
    ]

    lines += [
        "## 拒答",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 参与判分 | {negative['scored']} / {negative['evaluated']} |",
        f"| 拒答正确率（LLM judge） | {_pct(negative['refusal_accuracy'])} |",
        f"| 规则拒答信号命中率（诊断） | {_pct(negative['cue_rate'])} |",
        "",
        "## 报告",
        "",
        "| 字段 | 覆盖题数 | 覆盖率 |",
        "|---|---|---|",
    ]
    for field, info in report["field_coverage"].items():
        lines.append(
            f"| {field} | {info['covered']} / {info['total']} | {_pct(info['rate'])} |"
        )

    lines += [
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 字段完整率 | {_pct(report['complete_rate'])} |",
        f"| 平均字段覆盖率 | {_pct(report['mean_field_coverage'])} |",
        f"| 平均事实片段覆盖率 | {_pct(report['mean_fact_coverage'])} |",
        f"| 忠实率（faithfulness ≥ 1） | {_pct(report['faithful_rate'])} |",
        f"| 完全有据率（faithfulness = 2） | {_pct(report['grounded_rate'])} |",
        f"| 可执行建议率 | {_pct(report['advice_rate'])} |",
        f"| 报告子流程执行率（路由 → fetch） | {_pct(report['pipeline_ok_rate'])} |",
        f"| fetch 参数正确率 | {_pct(report['fetch_args_ok_rate'])} |",
        "",
    ]

    lines += [
        "## 延迟",
        "",
        "| 链路 | 生成 P50 | 生成 P95 | 判分 P50 | 判分 P95 |",
        "|---|---|---|---|---|",
    ]
    for label, info in [
        ("answerable", answerable),
        ("negative", negative),
        ("report", report),
    ]:
        lines.append(
            f"| {label} | {_ms(info['gen_latency_ms']['p50'])} ms | "
            f"{_ms(info['gen_latency_ms']['p95'])} ms | "
            f"{_ms(info['judge_latency_ms']['p50'])} ms | "
            f"{_ms(info['judge_latency_ms']['p95'])} ms |"
        )

    pass_k = summary.get("pass_k")
    if pass_k:
        k = pass_k["k"]
        lines += [
            "",
            f"## 稳定性（pass^{k}）",
            "",
            f"同一题独立跑 {k} 次，严格 pass^{k} 要求 {k} 次全部通过；"
            "`至少一次` 只要求有一次通过，用来区分「不稳定」和「完全不会」。",
            "",
            f"| 类型 | 题数 | 严格 pass^{k} | 至少一次通过 | 平均单次通过 | 波动题 |",
            "|---|---|---|---|---|---|",
        ]
        for label in ("answerable", "negative", "report", "overall"):
            stats = pass_k["types"].get(label)
            if not stats:
                continue
            lines.append(
                f"| {label} | {stats['questions']} | {_pct(stats['strict_pass_rate'])} | "
                f"{_pct(stats['at_least_once_rate'])} | "
                f"{_pct(stats['mean_run_pass_rate'])} | {stats['flaky_count']} |"
            )
        flaky_ids = [
            f"{label}: {', '.join(stats['flaky_ids'])}"
            for label, stats in pass_k["types"].items()
            if label != "overall" and stats["flaky_ids"]
        ]
        if flaky_ids:
            lines += ["", "波动题（至少一次通过、至少一次失败）：", ""]
            lines += [f"- {item}" for item in flaky_ids]
        lines += [""]

    lines += [
        "",
        "## 失败样例",
        "",
        "| ID | 类型 | 问题 | 判分 | 原因 / 回答摘要 |",
        "|---|---|---|---|---|",
    ]
    failures = _failure_rows(payload)
    for item_id, item_type, question, score, reason in failures[:20]:
        lines.append(
            f"| {item_id} | {item_type} | {_cell(question)} | {_cell(score)} | {_cell(reason)[:160]} |"
        )
    if not failures:
        lines.append("| - | - | 全部通过 | - | - |")
    if len(failures) > 20:
        lines.append(f"| ... | | 其余 {len(failures) - 20} 条见 JSON | | |")

    lines += [
        "",
        "## 复现",
        "",
        "```bash",
        "python eval/run_answer_eval.py --types answerable,negative --limit 2",
        "python eval/run_answer_eval.py --types report --limit 1",
        "python eval/run_answer_eval.py",
        "python -m unittest discover -s tests",
        "```",
        "",
        "## 限制",
        "",
        "- 生成与判分默认同为 `qwen-max`，存在自偏好；已用 gold answer、明确 rubric 和"
        "低 temperature 降低，但未做人工双标与一致性系数。",
        f"- correctness / faithfulness 是 0/1/2 粗粒度 rubric，同一条回答在不同运行间"
        f"可能有 1 分波动；本次 repeat={meta.get('repeat', 1)}，"
        f"{'已计算 pass^k。' if meta.get('repeat', 1) > 1 else '未做 pass^k。'}",
        "- gold answer 是单一参考答案，语义正确但表述不同的回答可能被判为 partial。",
        "- 报告题都显式给出 user_id / month；`get_user_id` / `get_current_month` 的随机 mock"
        "分支未被覆盖。",
        "- 报告字段覆盖是词面匹配（标签 + 数值），不是语义等价；语义忠实度由 LLM judge 判断。",
        "- 报告请求正常走确定性子流程，工具序列由代码保证；"
        "`pipeline_ok_rate` 下降只可能来自路由未命中或子流程异常回退。",
        "- 报告生成耗时主要来自最终输出长度；当前只统计总耗时，未拆首 token 与流式延迟。",
        "- answerable / negative 走 `RagSummarizeService` 的 Top-3 生成链路；检索基线用 k=8，"
        "两者口径不同，不能直接相减。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG / Agent 回答级评测")
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument(
        "--types",
        default="answerable,negative,report",
        help="逗号分隔：answerable,negative,report",
    )
    parser.add_argument("--k", type=int, default=3, help="RAG 生成时检索的片段数")
    parser.add_argument("--limit", type=int, default=None, help="每类只跑前 N 题")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="每道题独立跑 k 次，k>1 时输出 pass^k；默认 1",
    )
    parser.add_argument("--judge-model", default=None, help="判分模型，默认与生成模型相同")
    parser.add_argument("--out", default="eval/results/answer_baseline.json")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    types = [value.strip() for value in args.types.split(",") if value.strip()]
    valid_types = {"answerable", "negative", "report"}
    unknown = [value for value in types if value not in valid_types]
    if unknown or not types:
        raise SystemExit(f"--types 非法：{unknown or args.types}")
    if args.repeat < 1:
        raise SystemExit("--repeat 必须是正整数")

    dataset_path = (ROOT / args.dataset).resolve()
    out_path = (ROOT / args.out).resolve()
    records_path = ROOT / "data" / "external" / "records.csv"

    rows = load_rows(dataset_path)
    errors = validate_rows(rows, ROOT / "data", records_path)
    if errors:
        print(f"VALIDATION_FAILED count={len(errors)}")
        for error in errors[:30]:
            print(f"- {error}")
        return 1
    print(f"VALIDATION_OK rows={len(rows)}")
    if args.validate_only:
        return 0

    from utils.config_handler import chroma_conf, rag_conf

    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    judge = JudgeModel(args.judge_model)

    service = None
    chunk_count = None
    if "answerable" in types or "negative" in types:
        from rag.rag_service import RagSummarizeService

        service = RagSummarizeService()
        service.ensure_knowledge_base()
        chunk_count = service.vector_store.count()

    answerable = []
    negative = []
    report = []

    if "answerable" in types:
        answerable = run_answerable(
            service, judge, rows, args.k, args.limit, args.repeat
        )
    if "negative" in types:
        negative = run_negative(
            service, judge, rows, args.k, args.limit, args.repeat
        )
    if "report" in types:
        from agent.react_agent import ReactAgent

        agent = ReactAgent()
        report = run_report(
            agent,
            judge,
            rows,
            load_records(records_path),
            args.limit,
            run_id,
            args.repeat,
        )

    summary = summarize(answerable, negative, report, args.repeat)
    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "dataset": str(dataset_path.relative_to(ROOT)),
        "types": types,
        "counts": dict(Counter(row["type"] for row in rows)),
        "collection": chroma_conf["collection_name"],
        "chunk_count": chunk_count,
        "embedding_model": rag_conf["embedding_model_name"],
        "chat_model": rag_conf["chat_model_name"],
        "judge_model": judge.model_name,
        "judge_calls": judge.calls,
        "judge_errors": judge.errors,
        "k": args.k,
        "repeat": args.repeat,
    }
    payload = {
        "meta": meta,
        "summary": summary,
        "details": {
            "answerable": answerable,
            "negative": negative,
            "report": report,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path = out_path.with_suffix(".md")
    markdown_path.write_text(to_markdown(payload), encoding="utf-8")

    print(
        f"answerable pass={_pct(summary['answerable']['pass_rate'])} "
        f"correct={_pct(summary['answerable']['correct_rate'])} "
        f"grounded={_pct(summary['answerable']['grounded_rate'])}"
    )
    print(
        f"negative pass={_pct(summary['negative']['pass_rate'])} "
        f"refusal={_pct(summary['negative']['refusal_accuracy'])}"
    )
    print(
        f"report pass={_pct(summary['report']['pass_rate'])} "
        f"complete={_pct(summary['report']['complete_rate'])} "
        f"faithful={_pct(summary['report']['faithful_rate'])}"
    )
    print(
        f"overall pass={summary['overall']['passed']}/{summary['overall']['evaluated']} "
        f"({_pct(summary['overall']['pass_rate'])})"
    )
    pass_k = summary.get("pass_k")
    if pass_k:
        overall_pk = pass_k["types"].get("overall", {})
        print(
            f"pass^{pass_k['k']} strict={_pct(overall_pk.get('strict_pass_rate'))} "
            f"at_least_once={_pct(overall_pk.get('at_least_once_rate'))} "
            f"flaky={overall_pk.get('flaky_count')}"
        )
    print(
        f"errors={summary['overall']['errors']} "
        f"judge_errors={summary['overall']['judge_errors']}"
    )
    print(f"wrote {out_path.relative_to(ROOT)} and {markdown_path.relative_to(ROOT)}")

    return 2 if summary["overall"]["errors"] or summary["overall"]["judge_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
