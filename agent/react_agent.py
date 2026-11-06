import os
import sqlite3

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.tools.agent_tools import (
    ensure_rag_ready,
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)
from agent.tools.middleware import (
    log_before_model,
    monitor_tool,
    report_prompt_switch,
)
from model.factory import chat_model
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.prompt_loader import load_system_prompts

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:  # 没装 langgraph-checkpoint-sqlite 时退化为会话内记忆
    SqliteSaver = None


class ReactAgent:
    def __init__(self):
        # 启动自检：空向量库自动重建，避免“md5 台账说做过、实际库是空的”。
        ensure_rag_ready()

        self.checkpointer, self._checkpointer_conn = self._build_checkpointer()
        self.agent = create_agent(
            model=chat_model,
            tools=[rag_summarize, get_weather, get_user_location,
                   get_user_id, get_current_month,
                   fetch_external_data, fill_context_for_report],
            system_prompt=load_system_prompts(),
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
            checkpointer=self.checkpointer,
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
        logger.info(f"[记忆] 使用 SQLite checkpointer：{db_path}")
        return checkpointer, conn

    def execute_stream(self, query: str, thread_id: str = "default"):
        input_dict = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 20,
        }

        # 第三个参数 context 是 runtime 里的共享上下文，用来做报告提示词切换。
        for chunk in self.agent.stream(
            input_dict,
            stream_mode="values",
            context={"report": False},
            config=config,
        ):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"

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


if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.execute_stream("给我生成我的使用报告", thread_id="cli-demo"):
        print(chunk, end="", flush=True)
