# lc graph 用langgraph实现agent

import re
from operator import add as op_add
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END, START, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import lc_agent
from tools import list_data_filies

# 定义状态，所有节点共享一个字典，add messages表示messages字段是累加的
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_logs: Annotated[list, op_add]   # 每次返回本轮增量list，跨轮累加，随checkpointer持久化
    awaiting_input: bool                  # 上一轮是否在等用户补齐（追问态）
    evidence_done: bool                   # 本轮是否已注入过依据，防死循环

# 可用数据文件集合，用于校验用户提到的文件名是否存在
AVAILABLE = set(list_data_filies())


def extract_filenames(text):
    """从用户消息里提取文件名：命中可用文件，或任何以 .csv 结尾的词"""
    return [t for t in text.split() if t in AVAILABLE or t.endswith(".csv")]


# 定义节点，输入所有state，返回要写入state的增量
def call_llm(state):
    return {"messages": [lc_agent.LLM.invoke(state["messages"])]}


def validate_input(state):
    """完整性校验：文件缺失或不存在则追问，否则放行去分析"""
    if state.get("tool_logs"):             # 已分析过，进入普通问答，不再强制追问文件
        return {"awaiting_input": False}
    if state.get("awaiting_input"):        # 上轮在追问，本轮是补充，放行
        return {"awaiting_input": False}
    last = state["messages"][-1]
    files = extract_filenames(last.content)
    missing = [f for f in files if f not in AVAILABLE]
    if not files:                          # 首轮/没提到文件名
        prompt = ("请问要分析哪个文件？请从以下列表选择: "
                  + ", ".join(sorted(AVAILABLE)))
        return {"messages": [SystemMessage(content=prompt)], "awaiting_input": True}
    if missing:                            # 提到但文件不存在
        prompt = ("以下文件不存在: " + ", ".join(missing)
                  + "。可用文件: " + ", ".join(sorted(AVAILABLE)) + "，请重新指定。")
        return {"messages": [SystemMessage(content=prompt)], "awaiting_input": True}
    return {"awaiting_input": False}


def execute_tools(state):
    """执行工具（含失败恢复 + 记录真实输出到 tool_logs）"""
    last_msg = state["messages"][-1]       # 含 tool_calls 的 AIMessage
    new_msgs, new_logs = [], []
    for call in last_msg.tool_calls:
        name, args, cid = call["name"], call["args"], call["id"]
        try:
            result = lc_agent.TOOL_MAP[name](**args)   # 成功
            content = str(result)
        except Exception as e:                         # 失败：把错误回喂LLM自主恢复
            content = f"工具执行出错: {e!r}。如需重试或更正参数请告知。"
        new_msgs.append(ToolMessage(content=content, tool_call_id=cid))
        new_logs.append({"name": name, "args": args, "result": content})
    return {"messages": new_msgs, "tool_logs": new_logs}


def evidence(state):
    """判断依据：把真实 tool_logs 注入给 LLM，让它据此作答（不编造）"""
    logs = state.get("tool_logs") or []
    if logs:
        text = "\n".join(f"#{i} {l['name']}({l['args']}) => {l['result']}"
                         for i, l in enumerate(logs[-8:]))
        return {"messages": [HumanMessage(
            content="以下是此前的真实工具调用记录，请据此作答（不要编造）:\n" + text)],
            "evidence_done": True}
    return {"evidence_done": True}


def _last_user_content(messages):
    """从消息里找最近一条用户输入（含 evidence 注入前的）"""
    for m in reversed(messages):
        if getattr(m, "type", "") == "human":
            return m.content
    return ""


def route_after_llm(state):
    """llm 之后的路由：调工具 / 回答依据 / 结束"""
    if state["messages"][-1].tool_calls:
        return "tools"
    if (state.get("evidence_done") is not True
            and (state.get("tool_logs") or [])
            and any(k in _last_user_content(state["messages"])
                    for k in ("依据", "为什么", "basis", "根据"))):
        return "evidence"
    return END


builder = StateGraph(AgentState)
builder.add_node("llm", call_llm)
builder.add_node("validate", validate_input)
builder.add_node("tools", execute_tools)
builder.add_node("evidence", evidence)
builder.add_edge(START, "validate")
# validate 之后：追问态则结束（等下一轮），否则继续分析
builder.add_conditional_edges("validate",
    lambda s: END if s.get("awaiting_input") else "llm",
    {END: END, "llm": "llm"})
# llm 之后：有 tool_calls 走 tools；问依据走 evidence；否则结束
builder.add_conditional_edges("llm", route_after_llm,
    {"tools": "tools", "evidence": "evidence", END: END})
builder.add_edge("tools", "llm")
builder.add_edge("evidence", "llm")
app = builder.compile(checkpointer=MemorySaver())

THREAD_ID = "web"


def run_turn(user_input, initial=False):
    """处理一轮对话。同一 THREAD_ID 跨调用由 checkpointer 记住历史。
    initial=True 表示本轮为会话首轮，需要带上系统提示。
    返回最终回答字符串。
    """
    return run_turn_full(user_input, initial)["messages"][-1].content


def run_turn_full(user_input, initial=False):
    """同 run_turn，但返回最终完整 state（含 messages 历史），供网页提取 trace"""
    if initial:
        state = {"messages": [SystemMessage(content=lc_agent.SYSTEM_PROMPT),
                              HumanMessage(content=user_input)]}
    else:
        state = {"messages": [HumanMessage(content=user_input)]}
    return app.invoke(state, config={"configurable": {"thread_id": THREAD_ID}})


def _extract_trace(messages):
    """从最终 state 的 messages 里提取工具调用过程，形状与前端期望的 {tool, args, result} 兼容"""
    trace = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", []):
            name = call["name"]
            args = call["args"]
            # 找紧随其后的 ToolMessage 作为结果
            result = next(
                (m.content for m in messages
                 if isinstance(m, ToolMessage) and m.tool_call_id == call["id"]),
                None,
            )
            trace.append({"tool": name, "args": args, "result": result})
    return trace