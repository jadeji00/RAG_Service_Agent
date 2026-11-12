import os
import sqlite3
from typing import Callable

from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.guardrails import new_guard_state
from agent.history_compressor import (
    SUMMARY_STREAM_TAG,
    HistoryCompressor,
    summarize_history_stats,
)
from agent.report_pipeline import generate_report, stream_report
from agent.scope_router import OUT_OF_SCOPE_ANSWER, route_intent
from agent.session_index import (
    checkpoint_time_ms,
    count_turns,
    delete_checkpoint_thread,
    is_session_thread,
    iso_from_ms,
    list_checkpoint_threads,
    make_title,
)
from agent.tools.agent_tools import (
    ensure_rag_ready,
    fetch_external_data,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)
from agent.tools.middleware import (
    compress_history,
    guard_before_model,
    log_before_model,
    monitor_tool,
    report_prompt_switch,
)
from observability.usage import TokenAccount, extract_usage
from rag.citations import merge_sources, summarize_retrieval_stats
from utils.config_handler import agent_conf, rag_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.prompt_loader import load_system_prompts

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:  # 没装 langgraph-checkpoint-sqlite 时退化为会话内记忆
    SqliteSaver = None


def _message_text(content) -> str:
    """把 str / 多模态 block 统一成可推送的文本。"""
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


class ReactAgent:
    def __init__(self):
        # 启动自检：空向量库自动重建，避免“md5 台账说做过、实际库是空的”。
        ensure_rag_ready()

        self._checkpoint_db_path = None
        self.checkpointer, self._checkpointer_conn = self._build_checkpointer()
        # 摘要器按 thread 缓存，必须跟着 Agent 单例活着；每次请求新建会丢掉复用。
        self.history_compressor = HistoryCompressor()
        self.agent = create_agent(
            model=self._build_stream_model(),
            tools=[rag_summarize, get_weather, get_user_location,
                   get_user_id, get_current_month,
                   fetch_external_data],
            system_prompt=load_system_prompts(),
            # compress_history 必须排在 report_prompt_switch 之后：后者是
            # dynamic_prompt，会整体替换 system_message；排前面追加的历史摘要
            # 会被它覆盖掉。列表顺序 = 由外到内，越靠后越晚执行。
            middleware=[monitor_tool, log_before_model, guard_before_model,
                        report_prompt_switch, compress_history],
            checkpointer=self.checkpointer,
        )

    def _build_stream_model(self):
        """Agent 主模型开增量输出；没有工具时 DashScope 默认就会增量，
        有工具时需要显式 `incremental_output`，否则最后一次性返回。"""
        return ChatTongyi(
            model=rag_conf["chat_model_name"],
            streaming=True,
            model_kwargs={"incremental_output": True},
        )

    def _build_checkpointer(self):
        """优先用 SQLite 把会话落到文件；依赖缺失时退回内存版。"""
        if SqliteSaver is None:
            logger.warning(
                "[记忆] 未安装 langgraph-checkpoint-sqlite，使用内存记忆（重启后丢失）"
            )
            return InMemorySaver(), None

        db_path = get_abs_path(
            agent_conf.get("checkpoint_db_path", "checkpoints/agent_memory.sqlite3")
        )
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()
        self._checkpoint_db_path = db_path
        logger.info(f"[记忆] 使用 SQLite checkpointer：{db_path}")
        return checkpointer, conn

    def generate_report(
        self,
        query: str,
        user_id: str | None = None,
        month: str | None = None,
        owner_user_id: str | None = None,
    ) -> dict:
        """走确定性报告子流程，不经过 ReAct 的工具选择。"""
        return generate_report(
            query, user_id=user_id, month=month, owner_user_id=owner_user_id
        )

    @property
    def checkpoint_db_path(self) -> str | None:
        """checkpoint 库路径；服务层用它给旧会话补归属，内存记忆时为 None。"""
        return getattr(self, "_checkpoint_db_path", None)

    def _remember_exchange(self, query: str, answer: str, thread_id: str) -> None:
        """报告/拒答子流程不经过 create_agent，手动把一问一答写回同一个 thread。"""
        try:
            self.agent.update_state(
                {"configurable": {"thread_id": thread_id}},
                {
                    "messages": [
                        HumanMessage(content=query),
                        AIMessage(content=answer),
                    ]
                },
            )
        except Exception as e:
            logger.warning(f"[记忆] 写入会话失败：{e}")

    def iter_chat(
        self,
        query: str,
        thread_id: str = "default",
        token_account: TokenAccount | None = None,
        owner_user_id: str | None = None,
    ):
        """产出路由、增量文本、工具状态与最终结果事件。

        ``owner_user_id`` 是登录账号绑定的业务用户 ID；报告链路会用它拦住
        「查别人使用记录」的请求。为空表示调用方没有登录态，不做归属校验。

        事件类型：
        - route:    {"route": "qa" | "report" | "out_of_scope"}
        - delta:    {"text": "..."}
        - tool:     {"name", "status", "args", "ok", "id", ...}
        - notice:   {"level", "text"} 报告失败降级提示
        - result:   {"route", "answer", "history", "sources", "retrieval"}
          ``sources`` 是这一轮引用到的资料条目（文件/小节/片段/相关度），
          ``retrieval`` 是合并后的检索统计；报告与越界两条路都不带引用。
        """
        account = token_account if token_account is not None else TokenAccount()
        route = route_intent(query)
        yield {"type": "route", "route": route}

        if route == "out_of_scope":
            self._remember_exchange(query, OUT_OF_SCOPE_ANSWER, thread_id)
            yield {"type": "delta", "text": OUT_OF_SCOPE_ANSWER}
            yield {
                "type": "result",
                "route": "out_of_scope",
                "answer": OUT_OF_SCOPE_ANSWER,
                "no_model_call": True,
            }
            return

        if route == "report":
            report_emitted = 0
            report_result = None
            try:
                for event in stream_report(
                    query, usage_sink=account.add, owner_user_id=owner_user_id
                ):
                    event_type = event.get("type")
                    if event_type == "delta":
                        report_emitted += len(event.get("text", ""))
                        yield event
                    elif event_type in {"meta", "tool"}:
                        yield event
                    elif event_type == "result":
                        report_result = event

                if not report_result or not str(report_result.get("answer", "")).strip():
                    raise RuntimeError("报告子流程没有返回内容")

                answer = str(report_result["answer"]).strip()
                self._remember_exchange(query, answer, thread_id)
                yield report_result
                return
            except Exception as e:
                logger.warning(f"[report pipeline] 流式执行失败：{e}")
                if report_emitted:
                    # 已经给用户推了半份报告，不能悄悄再跑一遍导致重复。
                    raise
                yield {
                    "type": "notice",
                    "level": "warning",
                    "text": "报告子流程失败，已切换到通用问答兜底。",
                }

        input_dict = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 16,
        }
        context = {
            "report": False,
            "guard": new_guard_state(),
            "thread_id": thread_id,
            "history_compressor": self.history_compressor,
            "usage_sink": account.add,
            "history_stats": [],
            "sources_raw": [],
            "retrieval_stats": [],
        }

        answer_parts: list[str] = []
        tool_args_by_id: dict[str, dict] = {}

        for mode, data in self.agent.stream(
            input_dict,
            stream_mode=["messages", "updates"],
            context=context,
            config=config,
        ):
            if mode == "messages":
                chunk, metadata = data
                meta = metadata or {}
                # 摘要模型是在 model 节点里调的，langgraph_node 同样是 "model"，
                # 只按节点过滤会把摘要正文当成回答推给用户，这里再按 tag 排掉。
                # 顺带避免摘要 token 被记两次账（compressor 已经通过 usage_sink 记过）。
                if SUMMARY_STREAM_TAG in (meta.get("tags") or []):
                    continue
                # tools 节点的内容不是模型回答，过滤掉。
                if meta.get("langgraph_node") != "model":
                    continue
                account.add(extract_usage(chunk))
                text = _message_text(getattr(chunk, "content", ""))
                if text:
                    answer_parts.append(text)
                    yield {"type": "delta", "text": text}
                continue

            if mode != "updates":
                continue

            for _node, update in (data or {}).items():
                for message in (update or {}).get("messages", []) or []:
                    tool_calls = getattr(message, "tool_calls", None)
                    if tool_calls:
                        for call in tool_calls:
                            call_id = call.get("id", "")
                            args = call.get("args") or {}
                            tool_args_by_id[call_id] = args
                            yield {
                                "type": "tool",
                                "name": call.get("name", ""),
                                "status": "start",
                                "args": args,
                                "id": call_id,
                                "ok": None,
                            }
                    if isinstance(message, ToolMessage):
                        call_id = getattr(message, "tool_call_id", "")
                        additional = getattr(message, "additional_kwargs", {}) or {}
                        yield {
                            "type": "tool",
                            "name": getattr(message, "name", "") or "tool",
                            "status": "end",
                            "args": tool_args_by_id.get(call_id, {}),
                            "id": call_id,
                            "ok": getattr(message, "status", "success") != "error",
                            "blocked": bool(additional.get("guardrail")),
                            "preview": _message_text(getattr(message, "content", ""))[:200],
                        }

        answer = "".join(answer_parts).strip()
        if not answer:
            raise RuntimeError("模型没有返回任何内容")

        yield {
            "type": "result",
            "route": "qa",
            "answer": answer,
            "history": summarize_history_stats(context["history_stats"]),
            "sources": merge_sources(context["sources_raw"]),
            "retrieval": summarize_retrieval_stats(context["retrieval_stats"]),
        }

    def execute_stream(self, query: str, thread_id: str = "default"):
        """兼容 Streamlit：只吐文本增量；路由/工具事件由 iter_chat 提供。"""
        for event in self.iter_chat(query, thread_id=thread_id):
            if event.get("type") == "delta":
                yield event.get("text", "")

    def load_history(self, thread_id: str) -> list[dict]:
        """从 checkpointer 还原某个会话的 user/assistant 消息，供界面恢复气泡。

        assistant 条目在能还原引用时会多带一个 ``sources`` 字段：引用条目存在
        ToolMessage.artifact 里（不进模型上下文，但会落 checkpoint），按「本轮工具
        结果 → 紧随其后的最终回答」配对，重新打开旧会话也能看到当时的资料出处。
        """
        try:
            snapshot = self.agent.get_state(
                {"configurable": {"thread_id": thread_id}}
            )
        except Exception as e:
            logger.warning(f"[记忆] 读取会话 {thread_id} 失败：{e}")
            return []

        messages = (snapshot.values or {}).get("messages", [])
        history: list[dict] = []
        pending_sources: list[list[dict]] = []
        for message in messages:
            if isinstance(message, ToolMessage):
                artifact = getattr(message, "artifact", None)
                if isinstance(artifact, dict) and artifact.get("sources"):
                    pending_sources.append(artifact["sources"])
                continue
            if isinstance(message, HumanMessage):
                # 护栏注入的提示不是用户真实输入，不回显。
                if (message.additional_kwargs or {}).get("guardrail"):
                    continue
                role = "user"
                # 新的一轮开始了，上一轮没被回答消费掉的引用不能顺延过来。
                pending_sources = []
            elif isinstance(message, AIMessage):
                # 带 tool_calls 的 AI 消息是中间步骤，不放进界面历史。
                if getattr(message, "tool_calls", None):
                    continue
                role = "assistant"
            else:
                continue

            content = message.content
            if not isinstance(content, str) or not content.strip():
                continue
            entry: dict = {"role": role, "content": content.strip()}
            if role == "assistant" and pending_sources:
                entry["sources"] = merge_sources(pending_sources)
                pending_sources = []
            history.append(entry)

        return history

    def list_sessions(
        self,
        limit: int = 20,
        thread_filter: Callable[[str], bool] | None = None,
    ) -> list[dict]:
        """列出可恢复的历史会话，供侧栏展示。

        只认两个前端生成的 32 位 hex 线程，标题取该线程的第一条用户消息，
        按最后写入顺序返回；没有用户消息的线程（例如只写了工具状态）跳过。
        ``thread_filter`` 由服务层传入做归属过滤，在截断 limit 之前生效，
        这样「别人的会话」不会挤掉自己的名额。
        """
        limit = max(1, min(int(limit), 50))
        db_path = getattr(self, "_checkpoint_db_path", None)
        if not db_path or not os.path.exists(db_path):
            return []
        try:
            threads = list_checkpoint_threads(db_path)
        except Exception as e:
            logger.warning(f"[记忆] 读取会话列表失败：{e}")
            return []

        sessions = []
        for item in threads:
            if len(sessions) >= limit:
                break
            if thread_filter is not None and not thread_filter(item["thread_id"]):
                continue
            history = self.load_history(item["thread_id"])
            title = make_title(history)
            if not title:
                continue
            sessions.append(
                {
                    "thread_id": item["thread_id"],
                    "title": title,
                    "turns": count_turns(history),
                    "updated_at": iso_from_ms(
                        checkpoint_time_ms(item.get("last_checkpoint"))
                    ),
                }
            )
        return sessions

    def delete_session(self, thread_id: str) -> dict[str, int]:
        """删除一个历史会话的 checkpoint 与 writes。

        只接受前端生成的 32 位 hex 线程，避免误删评测/调试线程；
        trace 由服务层根据同一个 ``thread_id`` 一并清理。
        """
        if not is_session_thread(thread_id):
            raise ValueError("无效的会话 ID")
        db_path = getattr(self, "_checkpoint_db_path", None)
        if not db_path or not os.path.exists(db_path):
            return {"checkpoints": 0, "writes": 0}
        return delete_checkpoint_thread(db_path, thread_id)


if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.execute_stream("给我生成我的使用报告", thread_id="cli-demo"):
        print(chunk, end="", flush=True)
