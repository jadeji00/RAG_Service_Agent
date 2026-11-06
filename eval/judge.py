"""回答级评测：LLM rubric 判分 + 报告事实的确定性核对。

分工：
- 语义判断（答得对不对、有没有依据、是否拒答）交给 LLM judge，强制 JSON；
- 报告是否覆盖字段、数字是否出现在 CSV 里，用确定性规则核对，不靠模型；
- 纯函数不依赖模型和网络，``tests/test_judge.py`` 可以直接覆盖。

注意：这是「参考标准答案 + rubric」的自动判分，不是人工标注的绝对真值。
"""

from __future__ import annotations

import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ANSWERABLE_JUDGE_PROMPT = """你是严格的 RAG 回答评测员。请依据「标准答案」和「检索上下文」给「模型回答」打分。

# 问题
{question}

# 标准答案（事实基准）
{gold_answer}

# 参考答案关键词（辅助提示，不要求逐字出现）
{terms}

# 检索上下文（模型实际看到的内容）
{context}

# 模型回答
{answer}

# 评分标准
correctness:
2 = 核心事实与标准答案一致，关键要点完整；
1 = 部分正确，但遗漏关键点或表述含糊；
0 = 与标准答案冲突、答非所问，或声称不知道。

faithfulness:
2 = 回答中的事实都能在检索上下文中找到依据；
1 = 大部分有依据，但包含少量上下文没有的推断；
0 = 包含上下文没有的具体事实（编造型号、数字、结论等）。

只评事实，不因语气、格式、长度扣分；回答比标准答案更详细不扣分。
只输出 JSON 对象，不要 Markdown 代码块：
{{"correctness": 0, "faithfulness": 0, "reason": "不超过60字"}}
"""


NEGATIVE_JUDGE_PROMPT = """你是客服 Agent 的拒答评测员。下面是知识库范围外的请求，正确行为是明确说明无法回答、资料不足或不在服务范围，不编造具体信息。

# 用户问题
{question}

# 模型回答
{answer}

# 判定
refusal:
1 = 明确说明无法回答、资料不足、不在服务范围，或明确说明本产品不具备该能力并把话题拉回扫地/扫拖机器人业务；
0 = 尝试完成超范围任务（如写代码、翻译、做饭等），或给出了编造的具体信息。

只输出 JSON 对象，不要 Markdown 代码块：
{{"refusal": 0, "reason": "不超过60字"}}
"""


REPORT_JUDGE_PROMPT = """你是使用报告的质量评测员。请对照 CSV 原始记录，判断模型报告是否忠实、是否给出可执行建议。

# 用户问题
{question}

# CSV 原始记录（JSON）
{record}

# 模型报告
{answer}

# 评分标准
faithfulness:
2 = 报告中的数字、状态、对比变化都能在原始记录中找到；
1 = 大部分来自记录，但存在轻微推断或表述偏差；
0 = 出现原始记录中没有的具体数字或事实。

advice:
1 = 至少给出一条针对该用户数据的可执行建议；
0 = 没有建议，或建议与数据无关。

只输出 JSON 对象，不要 Markdown 代码块：
{{"faithfulness": 0, "advice": 0, "reason": "不超过60字"}}
"""


_MARKUP_RE = re.compile(r"[\s\*#`>_~:：,，。;；、|｜+\-—()（）\[\]【】]+")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?\s*(?:%|㎡|天|次|个月|年|小时|分钟)?")
_PAREN_RE = re.compile(r"[（(].*?[）)]")
_APPROX_RE = re.compile(r"(?:大约|约|左右)")


def extract_json(text: str) -> dict | None:
    """从模型输出里提取第一个合法 JSON 对象，兼容代码块和前后解释。"""
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def compact(text: Any) -> str:
    """去掉空白与 Markdown 标点，统一常见单位，避免格式差异导致漏判。"""
    value = str(text)
    value = value.replace("个百分点", "%").replace("百分比", "%")
    value = value.replace("平方米", "㎡").replace("平米", "㎡")
    value = _APPROX_RE.sub("", value)
    return _MARKUP_RE.sub("", value).lower()


def split_report_segments(field: str, raw: Any) -> list[str]:
    """把 CSV 单元格拆成可独立核对的事实片段。"""
    value = str(raw or "").replace("\\n", "\n").strip()
    if not value:
        return []
    if field == "特征":
        parts = re.split(r"[|｜]", value)
    else:
        parts = re.split(r"[\n；;、]", value)
    return [part.strip() for part in parts if part.strip()]


def fact_anchors(segment: str) -> list[str]:
    """提取一个事实片段的核对锚点。

    - ``覆盖率:87%`` -> ``["覆盖率", "87%"]``，标签和数值都要出现；
    - ``覆盖率-1%`` -> ``["覆盖率", "-1%"]``；
    - ``65㎡公寓`` -> ``["65㎡"]``，没有标签时只核对数值；
    - ``单身`` -> ``["单身"]``，没有数值时核对整个短语。
    """
    if ":" in segment or "：" in segment:
        label, value = re.split(r"[:：]", segment, maxsplit=1)
        value = _PAREN_RE.sub("", value).strip()
        return [part for part in (label.strip(), value) if part]

    match = _NUMBER_RE.search(segment)
    if match:
        prefix = segment[: match.start()].strip()
        return [part for part in (prefix, match.group(0).strip()) if part]
    return [segment.strip()]


def _label_matches(label: str, answer_compact: str) -> bool:
    """标签匹配：短标签要求原样出现；4 字以上允许省略一个二字片段。

    ``主刷寿命`` 可以匹配 ``主刷剩余寿命``，``日均清扫`` 可以匹配
    ``日均清扫面积``；这样不会因为报告换了说法就直接判缺失。
    """
    label_compact = compact(label)
    if not label_compact:
        return True
    if label_compact in answer_compact:
        return True
    if len(label_compact) >= 4:
        bigrams = [
            label_compact[index: index + 2]
            for index in range(len(label_compact) - 1)
        ]
        hits = sum(1 for bigram in bigrams if bigram in answer_compact)
        return hits >= len(bigrams) - 1
    return False


def _value_matches(value: str, answer_compact: str) -> bool:
    """数值优先匹配：``剩余28天`` 只要回答里有 ``28天`` 即可。"""
    value_compact = compact(value)
    if not value_compact:
        return True
    if value_compact in answer_compact:
        return True
    number = _NUMBER_RE.search(value)
    if number:
        return compact(number.group(0)) in answer_compact
    return False


def _segment_matches(segment: str, answer_compact: str) -> bool:
    if ":" in segment or "：" in segment:
        label, value = re.split(r"[:：]", segment, maxsplit=1)
        value = _PAREN_RE.sub("", value).strip()
        return _label_matches(label, answer_compact) and _value_matches(
            value, answer_compact
        )

    match = _NUMBER_RE.search(segment)
    if match:
        prefix = segment[: match.start()].strip()
        if prefix and not _label_matches(prefix, answer_compact):
            return False
        return _value_matches(match.group(0), answer_compact)

    return compact(segment) in answer_compact


def score_report_facts(
    answer: str,
    record: Mapping[str, Any],
    expected_fields: Sequence[str],
) -> dict:
    """确定性核对报告字段覆盖情况。

    字段覆盖口径：该字段至少命中 1 条事实，且命中比例不低于 50%。
    这是「有没有写到」的下界，不判断语义等价；语义忠实度由 LLM judge 负责。
    """
    answer_compact = compact(answer)
    fields: dict[str, dict] = {}
    total_facts = 0
    total_matched = 0

    for field in expected_fields:
        segments = split_report_segments(field, record.get(field, ""))
        facts = []
        for segment in segments:
            anchors = fact_anchors(segment)
            matched = _segment_matches(segment, answer_compact)
            facts.append(
                {
                    "fact": segment,
                    "anchors": anchors,
                    "matched": matched,
                }
            )

        matched_count = sum(1 for fact in facts if fact["matched"])
        total = len(facts)
        threshold = max(1, math.ceil(total * 0.5)) if total else 0
        covered = total > 0 and matched_count >= threshold
        fields[field] = {
            "matched": matched_count,
            "total": total,
            "covered": covered,
            "facts": facts,
        }
        total_facts += total
        total_matched += matched_count

    covered_fields = sum(1 for info in fields.values() if info["covered"])
    return {
        "fields": fields,
        "field_coverage": covered_fields / len(expected_fields) if expected_fields else 0.0,
        "fact_coverage": total_matched / total_facts if total_facts else 0.0,
        "complete": bool(expected_fields) and covered_fields == len(expected_fields),
    }


def refusal_cue(answer: str) -> bool:
    """规则版拒答信号，仅作诊断；最终以 LLM judge 的 refusal 为准。"""
    text = str(answer or "")
    patterns = (
        r"不知道",
        r"无法(?:回答|提供|处理|帮助|维修)",
        r"未能(?:找到|回答)",
        r"(?:资料|知识库|上下文|记录).{0,8}(?:没有|未|无法)",
        r"没有(?:提到|相关|找到)",
        r"不在.{0,8}(?:范围|服务)",
        r"建议.{0,12}(?:联系|咨询).{0,12}(?:售后|客服|品牌)",
        r"抱歉.{0,20}(?:无法|不能|仅能|只能)",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _safe_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(part if isinstance(part, str) else str(part) for part in value)
    return str(value)


def _score(value: Any, upper: int) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        match = re.match(r"\s*([0-2])", value)
        if match:
            value = match.group(1)
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if 0 <= number <= upper else None


class JudgeModel:
    """qwen-max 判分器；temperature=0，失败自动重试一次。"""

    def __init__(self, model_name: str | None = None, temperature: float = 0.0):
        from langchain_community.chat_models.tongyi import ChatTongyi
        from utils.config_handler import rag_conf

        self.model_name = model_name or rag_conf["chat_model_name"]
        self.model = ChatTongyi(model=self.model_name, temperature=temperature)
        self.calls = 0
        self.errors = 0

    def _ask_json(self, prompt: str) -> dict:
        from langchain_core.messages import HumanMessage

        last_raw = ""
        for attempt in range(2):
            try:
                response = self.model.invoke([HumanMessage(content=prompt)])
                last_raw = _safe_text(response.content)
            except Exception as exc:  # 网络/限流等，重试一次
                if attempt == 0:
                    time.sleep(1)
                    continue
                self.errors += 1
                return {"_error": f"{type(exc).__name__}: {exc}", "_raw": last_raw}

            self.calls += 1
            parsed = extract_json(last_raw)
            if parsed is not None:
                return parsed
            prompt = prompt + "\n\n上次输出不是合法 JSON。只输出 JSON 对象，不要任何解释。"

        self.errors += 1
        return {"_error": "judge_json_parse_failed", "_raw": last_raw[:600]}

    def judge_answerable(
        self,
        question: str,
        gold_answer: str,
        gold_terms: Sequence[str],
        context: str,
        answer: str,
    ) -> dict:
        prompt = ANSWERABLE_JUDGE_PROMPT.format(
            question=question,
            gold_answer=gold_answer,
            terms="、".join(gold_terms) if gold_terms else "无",
            context=context,
            answer=answer,
        )
        data = self._ask_json(prompt)
        if "_error" in data:
            return {
                "correctness": None,
                "faithfulness": None,
                "reason": data.get("_raw", ""),
                "error": data["_error"],
            }

        correctness = _score(data.get("correctness"), 2)
        faithfulness = _score(data.get("faithfulness"), 2)
        error = ""
        if correctness is None or faithfulness is None:
            error = "judge_score_invalid"
        return {
            "correctness": correctness,
            "faithfulness": faithfulness,
            "reason": str(data.get("reason", ""))[:300],
            "error": error,
        }

    def judge_negative(self, question: str, answer: str) -> dict:
        prompt = NEGATIVE_JUDGE_PROMPT.format(question=question, answer=answer)
        data = self._ask_json(prompt)
        if "_error" in data:
            return {
                "refusal": None,
                "reason": data.get("_raw", ""),
                "error": data["_error"],
            }

        refusal = _score(data.get("refusal"), 1)
        return {
            "refusal": refusal,
            "reason": str(data.get("reason", ""))[:300],
            "error": "" if refusal is not None else "judge_score_invalid",
        }

    def judge_report(self, question: str, record: Mapping[str, Any], answer: str) -> dict:
        prompt = REPORT_JUDGE_PROMPT.format(
            question=question,
            record=json.dumps(record, ensure_ascii=False, indent=1),
            answer=answer,
        )
        data = self._ask_json(prompt)
        if "_error" in data:
            return {
                "faithfulness": None,
                "advice": None,
                "reason": data.get("_raw", ""),
                "error": data["_error"],
            }

        faithfulness = _score(data.get("faithfulness"), 2)
        advice = _score(data.get("advice"), 1)
        error = ""
        if faithfulness is None or advice is None:
            error = "judge_score_invalid"
        return {
            "faithfulness": faithfulness,
            "advice": advice,
            "reason": str(data.get("reason", ""))[:300],
            "error": error,
        }


def load_records(path: str | Path) -> dict[tuple[str, str], dict]:
    """读取报告用的 CSV，按 (用户ID, 月份) 建索引。"""
    records: dict[tuple[str, str], dict] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            user_id = (row.get("用户ID") or "").strip()
            month = (row.get("时间") or "").strip()
            if user_id and month:
                records[(user_id, month)] = row
    return records
