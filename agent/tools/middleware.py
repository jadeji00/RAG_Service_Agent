from typing import Callable

from langchain.agents import AgentState
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from agent.guardrails import (
    authorize_tool_call,
    new_guard_state,
    record_tool_result,
)
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts,load_report_prompts

# 正常报告请求走 agent.report_pipeline，不进入 ReAct。这里的切换只服务兜底：
# 路由没识别出报告意图、模型在普通对话里取到了使用记录时，下一次模型调用切到
# 报告提示词，避免拿主提示词硬写报告。
REPORT_CONTEXT_TOOLS = {"fetch_external_data"}


def should_switch_report(tool_name: str) -> bool:
    """工具执行成功后，是否需要把上下文切到报告场景。"""
    return tool_name in REPORT_CONTEXT_TOOLS


def _guard_state(runtime: Runtime) -> dict | None:
    """每次运行独立的护栏状态；没有 context 时退化为“只校验不累计”。"""
    context = getattr(runtime, "context", None)
    if context is None:
        return None
    guard = context.get("guard")
    if guard is None:
        guard = new_guard_state()
        context["guard"] = guard
    return guard


@wrap_tool_call
def monitor_tool(
    # 请求的数据封装
    request: ToolCallRequest,
    # 执行的函数本身
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:            # 工具执行的监控
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args") or {}
    logger.info(f"[tool monitor]执行工具: {tool_name}")
    logger.info(f"[tool monitor]传入参数: {tool_args}")

    guard = _guard_state(request.runtime)
    key = ""
    if guard is not None:
        allowed, reason, key = authorize_tool_call(tool_name, tool_args, guard)
        if not allowed:
            logger.warning(f"[tool guard]拦截 {tool_name}: {reason}")
            return ToolMessage(
                content=f"【护栏拦截】{reason}",
                tool_call_id=request.tool_call.get("id", ""),
                name=tool_name,
                status="error",
                additional_kwargs={"guardrail": True, "reason": reason},
            )

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{tool_name}调用成功")
        if guard is not None and key:
            record_tool_result(guard, key, True)
        context = getattr(request.runtime, "context", None)
        if should_switch_report(tool_name) and context is not None:
            context["report"] = True
        return result
    except Exception as e:
        logger.error(
            f"工具{tool_name}调用失败, 原因: {str(e)}",
            exc_info=True,
        )
        if guard is not None and key:
            record_tool_result(guard, key, False)
        # 把错误作为 ToolMessage 交还给模型，让它有机会换参数或改用其他工具，
        # 而不是直接把整轮会话打断。
        return ToolMessage(
            content=f"工具 {tool_name} 执行失败：{str(e)}",
            tool_call_id=request.tool_call.get("id", ""),
            name=tool_name,
            status="error",
        )

@before_model
def guard_before_model(
    state: AgentState,
    runtime: Runtime,
):
    """连续拦截后注入一次收口提示，避免模型反复试错。

    ChatTongyi 要求 system 消息必须在首位，所以这里用带标记的 HumanMessage；
    ``load_history`` 会跳过这类消息，不会污染界面历史。
    """
    context = getattr(runtime, "context", None) or {}
    guard = context.get("guard")
    if not guard:
        return None
    if int(guard.get("blocked", 0)) < 2 or guard.get("force_injected"):
        return None

    guard["force_injected"] = True
    logger.warning("[tool guard]连续拦截，注入收口提示")
    return {
        "messages": [
            HumanMessage(
                content=(
                    "【系统护栏】工具调用已被连续拦截。现在必须停止调用工具，"
                    "直接基于已有信息回答；如果信息不足，明确说明不知道。"
                ),
                additional_kwargs={"guardrail": True},
            )
        ]
    }


@before_model
def log_before_model(
     state: AgentState,            # 整个Agent智能体中的状态记录
     runtime: Runtime,             # 记录了整个执行过程中的上下文信息
):
    # 在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型, 带有{len(state['messages'])}条消息。")

    last_message = state['messages'][-1]
    content = last_message.content if isinstance(last_message.content, str) else str(last_message.content)
    logger.debug(f"[log_before_model]{type(last_message).__name__} | {content.strip()}")
    return None

@dynamic_prompt #每一次生成提示词之前，调用此函数
def report_prompt_switch(requests:ModelRequest): # 报告提示符切换
    is_report=requests.runtime.context.get('report', False)
    if is_report:   #是报告生成场景，返回报告生成提示词内容
        return load_report_prompts()
    return load_system_prompts()


