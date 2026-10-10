# lc_agent.py —— langchain 版 agent 核心逻辑（供网页与命令行复用）

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

# 复用 tools.py 里已验证的真实工具函数
from tools import list_data_filies, emg_basic_stats, analyze_emg, compare_emg

# @tool 把函数包装成 langchain 认识的"工具"，自动从 docstring + 参数类型生成工具名片
TOOLS = [
    tool(list_data_filies),
    tool(emg_basic_stats),
    tool(analyze_emg),
    tool(compare_emg),
]

# 工具名 -> 原函数 的查找表（t.func 取得被 @tool 包装前的原始函数）
TOOL_MAP = {t.name: t.func for t in TOOLS}

LLM = ChatDeepSeek(model="deepseek-chat").bind_tools(TOOLS)

SYSTEM_PROMPT = (
    "你是专业运动生理学助手，回答通俗语言。"
    "分析数据时请自主调用工具：先列出文件，再对每个文件做疲劳分析，"
    "需要对比时调用对比工具。"
)


def start_session():
    """新建一个会话，返回以系统提示开头的 BaseMessage 列表"""
    return [SystemMessage(content=SYSTEM_PROMPT)]


def run_turn(messages, user_input, trace=None):
    """处理一轮对话，内部循环直到 LLM 不再要求调用工具。
    messages 为 BaseMessage 列表（由 start_session() 创建，全局保存以记住上下文）。
    若传入 trace（列表），会把每一步工具调用追加进去，供前端展示。
    """
    messages.append(HumanMessage(content=user_input))

    while True:
        reply = LLM.invoke(messages)
        if not reply.tool_calls:              # 模型不调工具 => 直接回答
            messages.append(reply)
            return reply.content

        messages.append(reply)                # 工具结果必须跟在含 tool_calls 的 assistant 回复之后
        for call in reply.tool_calls:
            name = call["name"]
            args = call["args"]
            call_id = call["id"]
            result = TOOL_MAP[name](**args)
            if trace is not None:
                trace.append({"tool": name, "args": args, "result": result})
            messages.append(ToolMessage(content=str(result), tool_call_id=call_id))