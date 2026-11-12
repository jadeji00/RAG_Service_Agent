"""会话历史压缩：给模型「摘要 + 最近若干条原文」的视图，checkpoint 原文不动。

长会话每轮都把全部历史回灌给模型，input token 随轮数线性增长。这里在模型调用
前把**较早的完整轮次**折成一段滚动摘要，最近的消息保持原文。

关键取舍：用 ``wrap_model_call`` + ``ModelRequest.override`` 只改本次请求的视图，
不用 ``before_model`` 返回 ``RemoveMessage`` 改写 state。后者会把摘要写进
checkpoint，界面历史就只剩摘要 + 最近几条，用户翻不回早期对话；而且官方
``SummarizationMiddleware`` 正是这么做的，所以这里没有直接复用它。

摘要按 thread 缓存：折叠区一字未变就直接复用，不再调模型。折叠区变长时**从原文
重写**而不是「旧摘要 + 新增对话」增量合并——实测轻量模型连着合并几轮会挤掉早期
事实、还会编造对话里没有的型号 / 用户 ID，所以每轮都从对话稿重新摘。

摘要出来后还要过一道编造校验（``find_unsourced_identifiers``）：摘要里出现对话稿
中找不到的标识符就整轮放弃压缩。主模型会把摘要当成用户说过的事实复述，这类错
比多花几千 token 严重得多。
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from observability.usage import extract_usage
from utils.config_handler import agent_conf
from utils.logger_handler import logger

_DEFAULTS = {
    "enabled": True,
    "trigger_messages": 16,
    "keep_messages": 8,
    "max_summary_chars": 600,
    "max_source_chars": 400,
    "max_transcript_chars": 12000,
    "cache_threads": 64,
    # 摘要用轻量模型而不是主模型：实测同一段对话稿 qwen-max 要 5s、qwen-flash
    # 只要 1.5s，而摘要质量对「只抄事实」这种任务差别不大。压不下来就直接关掉
    # 压缩（enabled: false），不要为了省钱牺牲主回答质量。
    "summarizer_model": "qwen-flash",
}

_INSTRUCTION = (
    "把下面这段扫地/扫拖机器人客服对话压成要点清单，作为背景交给客服模型。\n"
    "要求：\n"
    "1. 用「- 」开头，一轮问答只写一条，形如「问X → 答Y」，每条不超过 30 字；\n"
    "2. 不要罗列原因、步骤、注意事项，只留最关键的那一个结论；\n"
    "3. 按时间先后排列，用户问过的每一轮都要留一条，不要只保留最近几轮；\n"
    "4. 只写对话里**确实出现过**的信息；对话里没有的型号、用户 ID、错误码、"
    "日期、版本号一律不要写，更不要编造示例值；\n"
    "5. 对话里出现过的型号、错误码、数字、日期必须原样照抄，不要改写或补全；\n"
    "6. 不要分段标题、不要开场白和结尾；总长不超过 {max_chars} 字。"
)
# 第 1、2 条是实测调出来的：早期版本只说「一条一个事实」，qwen-flash 会把一个
# 错误码的每种成因都写成一条，13 轮对话能产出 49 条 / 1150 字，摘要耗时 5.7s
# 而且完全无视字数上限。改成「一轮问答一条 + 不罗列原因」后是 7 条 / 268 字 /
# 1.6s，第一轮的 E05 仍然留着（见 log/probe_cap.py）。

# 摘要注入到 system 消息里，而不是塞一条 HumanMessage。
# 实测塞 HumanMessage 时模型会把摘要的排版当成回答模板，每个回答开头都先复述
# 一遍「用户诉求 / 已给出的结论」，所以这里明确写清它是背景、不要照抄格式。
SUMMARY_SYSTEM_TEMPLATE = (
    "\n\n【较早对话的要点清单】\n"
    "{summary}\n"
    "（以上是你和这位用户早前对话的背景要点，仅用于理解上下文。"
    "不要在回答里复述或照搬它的格式，直接回答用户最新的问题。）"
)

# 摘要模型是在 model 节点内部调用的，它的 token 会和主模型的 token 一起被
# stream_mode="messages" 推出来（metadata 里 langgraph_node 同样是 "model"）。
# 打上这个 tag，外层就能把摘要的增量丢掉——否则用户会先看到一整段要点清单，
# 然后才是真回答。
SUMMARY_STREAM_TAG = "history_summary"

# 摘要里出现、但对话稿里找不到来源的「标识符」就是编造。只查同时含字母和数字
# 的短 token（E05 / XG900 / U123456 / 2.4G）和 ISO 日期，纯中文数字如
# 「3-6 个月」不算，否则正常摘要会被误杀。
# 必须加 re.ASCII：\w 在 Python3 默认匹配中文，「E05为悬崖异常」会被当成一个
# 标识符，导致忠实摘要被误判为编造。
_IDENTIFIER_PATTERN = re.compile(
    r"[A-Za-z]+[A-Za-z0-9_.\-]*\d[A-Za-z0-9_.\-]*|\d{4}-\d{2}-\d{2}",
    re.ASCII,
)


@dataclass(frozen=True)
class FoldPlan:
    """折叠方案：``messages[:boundary]`` 进摘要，``messages[boundary:]`` 留原文。"""

    boundary: int
    before: int

    @property
    def after(self) -> int:
        # 视图里只剩保留段；摘要单独并进 system 消息，不算对话消息。
        return self.before - self.boundary


def message_text(content: Any) -> str:
    """把 str / 多模态 block 统一成纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def _is_flagged(message: BaseMessage, flag: str) -> bool:
    return bool((getattr(message, "additional_kwargs", None) or {}).get(flag))


def plan_fold(
    messages: Sequence[BaseMessage],
    *,
    trigger: int,
    keep: int,
) -> FoldPlan | None:
    """决定折叠边界；不该压或压不动时返回 ``None``。

    两条硬规则：
    - **当前轮不折**：边界不会越过最后一条真实用户消息，否则模型只能从摘要里
      读到自己刚被问的问题；
    - **不拆工具组**：保留段不能以 ``ToolMessage`` 开头，它对应的
      ``AIMessage(tool_calls)`` 一旦被折走，模型会收到没有调用记录的孤儿结果。
    """
    total = len(messages)
    if total <= max(1, int(trigger)):
        return None

    last_human = -1
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage) and not _is_flagged(message, "guardrail"):
            last_human = index
    if last_human <= 0:
        # 只有当前这一轮，没有可折的历史
        return None

    boundary = min(total - max(1, int(keep)), last_human)
    while boundary < total and isinstance(messages[boundary], ToolMessage):
        boundary += 1
    if boundary <= 0 or boundary >= total:
        return None
    return FoldPlan(boundary=boundary, before=total)


def build_transcript(
    messages: Sequence[BaseMessage],
    *,
    max_source_chars: int,
) -> str:
    """把要折叠的消息转成给摘要模型看的对话稿。

    工具结果（尤其是检索片段）体积最大、时效最短，只留工具名不留正文；护栏注入
    的收口提示不是用户说的话，也不进稿。
    """
    lines: list[str] = []
    for message in messages:
        if _is_flagged(message, "guardrail"):
            continue
        if isinstance(message, ToolMessage):
            lines.append(f"[工具结果] {getattr(message, 'name', '') or 'tool'}")
            continue
        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                names = ", ".join(str(call.get("name", "")) for call in tool_calls)
                lines.append(f"[调用工具] {names}")
                continue
            label = "客服"
        elif isinstance(message, HumanMessage):
            label = "用户"
        else:
            continue

        text = message_text(getattr(message, "content", "")).strip()
        if not text:
            continue
        if len(text) > max_source_chars:
            text = text[:max_source_chars] + "…"
        lines.append(f"{label}：{text}")
    return "\n".join(lines)


def merge_summary_into_system(
    system_message: BaseMessage | str | None,
    summary: str,
) -> SystemMessage:
    """把摘要接到 system 消息末尾。

    ``dynamic_prompt`` 这类中间件是整体替换 ``system_message`` 的，所以本函数必须
    排在那之后执行，否则追加的摘要会被覆盖掉。
    """
    base = ""
    if isinstance(system_message, str):
        base = system_message
    elif system_message is not None:
        base = message_text(getattr(system_message, "content", ""))
    return SystemMessage(content=base + SUMMARY_SYSTEM_TEMPLATE.format(summary=summary))


def find_unsourced_identifiers(summary: str, transcript: str) -> list[str]:
    """找出摘要里凭空出现的标识符（型号 / 错误码 / 用户 ID / 日期）。

    弱模型做「摘要 of 摘要」时会自己补全字段：对话里从没出现过 XG900、E203、
    U123456，它也会写进去，主模型再把这些当成用户说过的事实复述出来。这里做一
    次纯字符串校验，任何没有来源的标识符都让本轮放弃压缩，宁可不省 token。
    """
    if not summary or not transcript:
        return []
    haystack = transcript.upper()
    unsourced = []
    for token in _IDENTIFIER_PATTERN.findall(summary):
        candidate = token.strip(".,;:、，。；").upper()
        if len(candidate) < 2 or candidate in haystack:
            continue
        if candidate not in unsourced:
            unsourced.append(candidate)
    return unsourced


def summarize_history_stats(stats_list: Sequence[dict] | None) -> dict:
    """把一轮里多次模型调用的压缩统计合成一条，供 trace span 与前端展示。

    一轮 ReAct 可能调好几次模型（工具循环），每次都产出一条 stats；这里只关心
    「有没有真压下去」和「压了多少」，所以取触发成功的调用做聚合。
    """
    items = [s for s in (stats_list or []) if s]
    if not items:
        return {"model_calls": 0, "triggered": False, "reason": "no_model_call"}

    compressed = [s for s in items if s.get("triggered") and not s.get("fallback")]
    summary: dict[str, Any] = {
        "model_calls": len(items),
        "triggered": bool(compressed),
        "compressed_calls": len(compressed),
        "fallback_calls": len([s for s in items if s.get("triggered") and s.get("fallback")]),
        "duration_ms": round(sum(float(s.get("duration_ms") or 0) for s in items), 2),
    }
    if not compressed:
        summary["reason"] = next(
            (s["reason"] for s in reversed(items) if s.get("reason")), "below_trigger"
        )
        summary["before_messages"] = max(int(s.get("before_messages") or 0) for s in items)
        # 被编造校验拦下时把 token 记进 trace，否则线上只会看到「没压」却不知道为啥。
        unsourced = [t for s in items for t in (s.get("unsourced") or [])]
        if unsourced:
            summary["unsourced"] = sorted(set(unsourced))[:10]
        return summary

    summary.update(
        {
            "before_messages": max(int(s.get("before_messages") or 0) for s in compressed),
            "after_messages": min(int(s.get("after_messages") or 0) for s in compressed),
            "folded_messages": max(int(s.get("folded_messages") or 0) for s in compressed),
            "summary_chars": int(compressed[-1].get("summary_chars") or 0),
            "reused": any(bool(s.get("reused")) for s in compressed),
            # 摘要本身也要花钱，单列出来才能判断「压缩到底划不划算」。
            "summary_input_tokens": sum(int(s.get("summary_input_tokens") or 0) for s in items),
            "summary_output_tokens": sum(int(s.get("summary_output_tokens") or 0) for s in items),
        }
    )
    return summary


class HistoryCompressor:
    """按 thread 维护滚动摘要，产出「摘要 + 最近原文」的消息视图。"""

    def __init__(
        self,
        conf: dict | None = None,
        summarizer: Any | None = None,
    ):
        raw = (
            conf
            if conf is not None
            else ((agent_conf or {}).get("history") or {})
        )
        cfg = {**_DEFAULTS, **(raw or {})}
        self.enabled = bool(cfg["enabled"])
        self.trigger_messages = int(cfg["trigger_messages"])
        self.keep_messages = int(cfg["keep_messages"])
        self.max_summary_chars = int(cfg["max_summary_chars"])
        self.max_source_chars = int(cfg["max_source_chars"])
        self.max_transcript_chars = int(cfg["max_transcript_chars"])
        self.cache_threads = int(cfg["cache_threads"])
        self.summarizer_model = str(cfg["summarizer_model"] or "").strip()
        # 测试缝：注入假 summarizer 就能完全离线验证触发、复用与增量逻辑。
        self._summarizer = summarizer
        self._model = None
        # thread_id -> (折叠区对话稿, 摘要, 对话稿指纹)
        self._cache: dict[str, tuple[str, str, str]] = {}

    # ------------------------------------------------------------------ 视图

    def view(
        self,
        thread_id: str,
        messages: Sequence[BaseMessage],
        usage_sink: Callable[[dict], None] | None = None,
    ) -> tuple[list[BaseMessage], dict]:
        """返回 ``(给模型的消息视图, 统计)``。

        压缩成功时视图里只剩**保留段的原文**，摘要放在 ``stats["summary_text"]``，
        由调用方并进 system 消息（``merge_summary_into_system``）；不触发或需要
        退回时原样返回全部消息，``stats["fallback"]`` 为真。
        """
        if not self.enabled:
            return list(messages), {"triggered": False, "reason": "disabled"}

        plan = plan_fold(
            messages,
            trigger=self.trigger_messages,
            keep=self.keep_messages,
        )
        if plan is None:
            return list(messages), {
                "triggered": False,
                "reason": "below_trigger",
                "before_messages": len(messages),
            }

        started = time.perf_counter()
        transcript = build_transcript(
            messages[: plan.boundary], max_source_chars=self.max_source_chars
        )
        stats: dict[str, Any] = {
            "triggered": True,
            "before_messages": plan.before,
            "after_messages": plan.after,
            "folded_messages": plan.boundary,
            "kept_messages": plan.before - plan.boundary,
            "transcript_chars": len(transcript),
            "summarizer_model": self.summarizer_model,
        }

        # 摘要的用量要同时进总账（真实花费）和本次统计（压缩开销），
        # 否则「省了多少 input token」和「摘要自己花了多少」会混成一个数。
        summary_usage: list[dict] = []

        def sink(usage: dict) -> None:
            summary_usage.append(usage)
            if usage_sink is not None:
                usage_sink(usage)

        if not transcript.strip():
            stats.update({"fallback": True, "reason": "empty_transcript"})
            stats["duration_ms"] = self._ms(started)
            return list(messages), stats

        try:
            summary, reused, truncated = self._summary_for(
                str(thread_id or ""), transcript, sink
            )
        except Exception as exc:
            # 摘要失败不该打断对话：本轮退回完整历史，只记一条警告。
            logger.warning(
                f"[history] 摘要生成失败，本轮不压缩：{type(exc).__name__}: {exc}"
            )
            stats.update({"fallback": True, "reason": "summarizer_error"})
            self._add_usage(stats, summary_usage)
            stats["duration_ms"] = self._ms(started)
            return list(messages), stats

        stats["transcript_truncated"] = truncated

        if not summary.strip():
            stats.update({"fallback": True, "reason": "empty_summary"})
            self._add_usage(stats, summary_usage)
            stats["duration_ms"] = self._ms(started)
            return list(messages), stats

        # 编造校验：摘要里出现对话稿里没有的型号/错误码/ID/日期就整轮放弃压缩。
        # 主模型会把摘要当成用户说过的事实复述，这种错比多花几千 token 严重得多。
        unsourced = find_unsourced_identifiers(summary, transcript)
        if unsourced:
            logger.warning(
                f"[history] 摘要含无来源标识符，本轮不压缩：{unsourced[:5]}"
            )
            stats.update(
                {
                    "fallback": True,
                    "reason": "unsourced_identifiers",
                    "unsourced": unsourced[:10],
                }
            )
            self._add_usage(stats, summary_usage)
            stats["duration_ms"] = self._ms(started)
            return list(messages), stats

        stats.update(
            {
                "fallback": False,
                "reused": reused,
                "summary_chars": len(summary.strip()),
                "summary_text": summary.strip(),
                "summarizer_model": self.summarizer_model,
            }
        )
        self._add_usage(stats, summary_usage)
        stats["duration_ms"] = self._ms(started)
        # 只返回保留段：摘要交给调用方并进 system 消息，不混进对话消息里。
        return list(messages[plan.boundary :]), stats

    @staticmethod
    def _add_usage(stats: dict, usage_list: Sequence[dict]) -> None:
        stats["summary_input_tokens"] = sum(
            int(u.get("input_tokens") or 0) for u in usage_list
        )
        stats["summary_output_tokens"] = sum(
            int(u.get("output_tokens") or 0) for u in usage_list
        )

    # ------------------------------------------------------------------ 摘要

    def _summary_for(
        self,
        thread_id: str,
        transcript: str,
        usage_sink: Callable[[dict], None] | None,
    ) -> tuple[str, bool, bool]:
        """折叠区没变就复用缓存摘要，变了就从**原文**重写一次。

        刻意不做「旧摘要 + 新增对话」的增量合并：实测轻量模型连着合并几轮后，
        早期事实（比如第一轮问的 E05）会被挤掉，还会自己补出对话里从没出现的
        型号 / 用户 ID / 日期，主模型再把这些当成用户说过的事实复述。摘要模型
        很便宜，宁可每轮从原文重写。
        """
        fingerprint = hashlib.md5(transcript.encode("utf-8")).hexdigest()
        cached = self._cache.get(thread_id) if thread_id else None
        if cached and cached[2] == fingerprint:
            return cached[1], True, False

        payload = transcript
        truncated = len(payload) > self.max_transcript_chars
        if truncated:
            # 超长只送最近的部分；stats 里会记下来，方便判断是不是丢了早期事实。
            payload = payload[-self.max_transcript_chars :]

        summary = self._summarize(payload, usage_sink)
        if thread_id:
            self._cache[thread_id] = (transcript, summary, fingerprint)
            while len(self._cache) > max(1, self.cache_threads):
                self._cache.pop(next(iter(self._cache)))
        return summary, False, truncated

    def _summarize(
        self,
        payload: str,
        usage_sink: Callable[[dict], None] | None,
    ) -> str:
        response = self._summarizer_model().invoke(
            [
                SystemMessage(
                    content=_INSTRUCTION.format(max_chars=self.max_summary_chars)
                ),
                HumanMessage(content=payload),
            ],
            config={"tags": [SUMMARY_STREAM_TAG]},
        )
        usage = extract_usage(response)
        if usage and usage_sink:
            usage_sink(usage)
        return message_text(getattr(response, "content", "")).strip()

    def _summarizer_model(self):
        if self._summarizer is not None:
            return self._summarizer
        if self._model is None:
            # 惰性构造：导入本模块不该拉起模型（P4）。
            from langchain_community.chat_models.tongyi import ChatTongyi

            # 摘要不需要增量输出，关掉流式能从源头少一批 token 事件。
            self._model = ChatTongyi(
                model=self.summarizer_model,
                temperature=0,
                disable_streaming=True,
            )
        return self._model

    @staticmethod
    def _ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)


__all__ = [
    "SUMMARY_STREAM_TAG",
    "SUMMARY_SYSTEM_TEMPLATE",
    "FoldPlan",
    "HistoryCompressor",
    "build_transcript",
    "find_unsourced_identifiers",
    "merge_summary_into_system",
    "message_text",
    "plan_fold",
    "summarize_history_stats",
]
