import os
import sqlite3

from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.guardrails import new_guard_state
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
    guard_before_model,
    log_before_model,
    monitor_tool,
    report_prompt_switch,
)
from observability.usage import TokenAccount, extract_usage
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
        self.agent = create_agent(
            model=self._build_stream_model(),
            tools=[rag_summarize, get_weather, get_user_location,
                   get_user_id, get_current_month,
                   fetch_external_data],
            system_prompt=load_system_prompts(),
            middleware=[monitor_tool, log_before_model, guard_before_model,
                        report_prompt_switch],
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
    ) -> dict:
        """走确定性报告子流程，不经过 ReAct 的工具选择。"""
        return generate_report(query, user_id=user_id, month=month)

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
    ):
        """产出路由、增量文本、工具状态与最终结果事件。

        事件类型：
        - route:    {"route": "qa" | "report" | "out_of_scope"}
        - delta:    {"text": "..."}
        - tool:     {"name", "status", "args", "ok", "id", ...}
        - notice:   {"level", "text"} 报告失败降级提示
        - result:   {"route", "answer", ...}
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
            }
            return

        if route == "report":
            report_emitted = 0
            report_result = None
            try:
                for event in stream_report(query, usage_sink=account.add):
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
                # tools 节点的内容不是模型回答，过滤掉。
                if (metadata or {}).get("langgraph_node") != "model":
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
        }

    def execute_stream(self, query: str, thread_id: str = "default"):
        """兼容 Streamlit：只吐文本增量；路由/工具事件由 iter_chat 提供。"""
        for event in self.iter_chat(query, thread_id=thread_id):
            if event.get("type") == "delta":
                yield event.get("text", "")

    def load_history(self, thread_id: str) -> list[dict[str, str]]:
        """从 checkpointer 还原某个会话的 user/assistant 消息，供界面恢复气泡。"""
        try:
            snapshot = self.agent.get_state(
                {"configurable": {"thread_id": thread_id}}
            )
        except Exception as e:
            logger.warning(f"[记忆] 读取会话 {thread_id} 失败：{e}")
            return []

        messages = (snapshot.values or {}).get("messages", [])
        history: list[dict[str, str]] = []
        for message in messages:
            if isinstance(message, HumanMessage):
                # 护栏注入的提示不是用户真实输入，不回显。
                if (message.additional_kwargs or {}).get("guardrail"):
                    continue
                role = "user"
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
            history.append({"role": role, "content": content.strip()})

        return history

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """列出可恢复的历史会话，供侧栏展示。

        只认两个前端生成的 32 位 hex 线程，标题取该线程的第一条用户消息，
        按最后写入顺序返回；没有用户消息的线程（例如只写了工具状态）跳过。
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
