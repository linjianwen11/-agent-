# graph_server.py —— LangGraph 版网页后端:复用 lc_graph 的有状态工作流
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import lc_graph

app = FastAPI(title="生理信号分析 Agent (LangGraph)")

# 页面路径跟着文件走,避免"从别的目录启动就找不到"
INDEX_HTML = Path(__file__).parent / "index.html"

# 会话状态由 LangGraph 的 checkpointer + 固定 thread_id 自动记住,无需全局列表


class ChatBody(BaseModel):
    message: str


@app.get("/")
def index():
    """首页:返回聊天界面(复用前端)"""
    return FileResponse(INDEX_HTML)


@app.post("/api/chat")
def chat(body: ChatBody):
    """聊天接口:跑一轮 LangGraph 工作流,返回回答、工具调用过程和追问态"""
    result = lc_graph.run_turn_full(body.message)
    answer = result["messages"][-1].content
    trace = lc_graph._extract_trace(result["messages"])
    return {
        "answer": answer,
        "trace": trace,
        "tool_logs": result.get("tool_logs", []),
        "is_ask": result.get("awaiting_input", False),
    }


if __name__ == "__main__":
    # 用 8002 端口,与手写版 8000、LangChain 版 8001 三者并行
    uvicorn.run(app, host="127.0.0.1", port=8002)