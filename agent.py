# agent.py —— agent核心逻辑

import json
import client
import config
from tools import TOOLS, TOOL_MAP

SYSTEM_PROMPT = (
    "你是专业运动生理信号分析助手，可以分析 EMG (肌电) 信号 ,NIRS (近红外谱) 信号。\n"
    "你可以根据用户的问题，调用工具来分析数据。\n"
    "工具列表:\n"
    "- list_data_filies:查看 data 目录有哪些数据文件 \n"
    "- emg_basic_stats:分析 EMG 信号的基本统计量 \n"
    "- analyze_emg:分析 EMG 信号的详细特征 \n"
    "- compare_emg:对比两个 EMG 数据文件的疲劳程度，返回双方 rms/mdf 斜率和更疲劳的文件 \n"
    "使用建议：先调用 list_data_filies 查看数据文件，确认文件是否存在。如果文件不存在，先列文件查看。 \n"
    "如果文件存在，再调用 emg_basic_stats 查看基本统计量。 \n"
    "回答时用通俗语言，并解释背后的生理学意义。"
)

def _run_tool(name, arguments_json):
    try:
        args = json.loads(arguments_json)
        result = TOOL_MAP[name](**args)
    except Exception as e:
        return {"error": str(e)}
    return result

def run_turn(messages, user_input, trace=None):
    """处理一轮对话，内部循环直到llm不再要求调用工具"""
    if len(messages) > config.MAX_MEMORY:
        system = messages[0]
        messages[:] = [system] + messages[-(config.MAX_MEMORY-1):]

    messages.append({"role": "user", "content": user_input})

    while True:
        reply = client.chat(messages, tools=TOOLS)
        # print("模型回复的键:", list(reply.keys()))
        messages.append(reply)

        calls = reply.get("tool_calls")

        if not calls:
            break
        
        for call in calls:
            name = call["function"]["name"]
            arguments_json = call["function"]["arguments"]
            result = _run_tool(name, arguments_json)
            if trace is not None:
                trace.append({"tool": name, "result": result})
            messages.append({"role": "tool",
                             "tool_call_id": call["id"],
                             "content": json.dumps(result),
            })
    return messages[-1]["content"]
            
