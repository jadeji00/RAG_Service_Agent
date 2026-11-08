import uuid

import streamlit as st

from agent.react_agent import ReactAgent
from utils.logger_handler import logger

# 标题
st.title("智扫通机器人智能客服")
st.divider()

# 每个浏览器会话一个 thread_id；写进 URL，刷新后还能接上同一段记忆。
if "thread_id" not in st.session_state:
    thread_id = st.query_params.get("thread_id")
    if not thread_id:
        thread_id = uuid.uuid4().hex
        st.query_params["thread_id"] = thread_id
    st.session_state["thread_id"] = thread_id

if "agent" not in st.session_state:
    try:
        with st.spinner("正在检查知识库..."):
            st.session_state["agent"] = ReactAgent()
    except Exception:
        logger.error("Agent 初始化失败", exc_info=True)
        st.error("知识库初始化失败，请检查 data/ 目录、API Key 和 log/ 日志后重试。")
        st.stop()

if "message" not in st.session_state:
    try:
        st.session_state["message"] = st.session_state["agent"].load_history(
            st.session_state["thread_id"]
        )
    except Exception:
        logger.error("读取会话历史失败", exc_info=True)
        st.session_state["message"] = []

with st.sidebar:
    st.caption("会话记忆（SQLite）")
    st.caption(f"当前会话：{st.session_state['thread_id'][:8]}")
    if st.button("开启新会话", use_container_width=True):
        st.session_state["thread_id"] = uuid.uuid4().hex
        st.query_params["thread_id"] = st.session_state["thread_id"]
        st.session_state["message"] = []
        st.rerun()

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    try:
        res_stream = st.session_state["agent"].execute_stream(
            prompt,
            thread_id=st.session_state["thread_id"],
        )
        with st.chat_message("assistant"):
            # iter_chat 已经按 token 增量产出，直接交给 write_stream，不再逐字 sleep 伪装。
            answer = st.write_stream(res_stream)

        if isinstance(answer, list):
            answer = "".join(answer)
        answer = (answer or "").strip()
        if not answer:
            raise RuntimeError("模型没有返回任何内容")

        st.session_state["message"].append(
            {"role": "assistant", "content": answer}
        )
    except Exception:
        logger.error("对话执行失败", exc_info=True)
        error_text = "抱歉，服务暂时不可用，请稍后再试。"
        st.error(error_text)
        st.session_state["message"].append(
            {"role": "assistant", "content": error_text}
        )

    st.rerun()
