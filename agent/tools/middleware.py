from typing import Callable

from langchain.agents import AgentState
from langchain_core.messages import ToolMessage
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts,load_report_prompts

# 正常报告请求走 agent.report_pipeline，不进入 ReAct。这里的切换只服务兜底：
# 路由没识别出报告意图、模型在普通对话里取到了使用记录时，下一次模型调用切到
# 报告提示词，避免拿主提示词硬写报告。
REPORT_CONTEXT_TOOLS = {"fetch_external_data"}


def should_switch_report(tool_name: str) -> bool:
    """工具执行成功后，是否需要把上下文切到报告场景。"""
    return tool_name in REPORT_CONTEXT_TOOLS


@wrap_tool_call
def monitor_tool(
    # 请求的数据封装
    request: ToolCallRequest,
    # 执行的函数本身
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:            # 工具执行的监控
    logger.info(f"[tool monitor]执行工具: {request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数: {request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")
        if should_switch_report(request.tool_call['name']):
            request.runtime.context['report'] = True
        return result
    except Exception as e:
        logger.error(
            f"工具{request.tool_call['name']}调用失败, 原因: {str(e)}",
            exc_info=True,
        )
        # 把错误作为 ToolMessage 交还给模型，让它有机会换参数或改用其他工具，
        # 而不是直接把整轮会话打断。
        return ToolMessage(
            content=f"工具 {request.tool_call['name']} 执行失败：{str(e)}",
            tool_call_id=request.tool_call.get("id", ""),
            name=request.tool_call["name"],
            status="error",
        )

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


