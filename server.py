# server.py —— FastAPI 后端:提供页面 + 聊天接口,agent 核心全部复用
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import agent

app = FastAPI(title="生理信号分析 Agent")

# 页面路径跟着文件走,避免"从别的目录启动就找不到"
INDEX_HTML = Path(__file__).parent / "index.html"

# 本地单会话:对话记忆放内存(演示够用,服务重启即清空)
messages = [{"role": "system", "content": agent.SYSTEM_PROMPT}]


class ChatBody(BaseModel):
    message: str


@app.get("/")
def index():
    """首页:返回聊天界面"""
    return FileResponse(INDEX_HTML)


@app.post("/api/chat")
def chat(body: ChatBody):
    """聊天接口:调 agent 核心,返回回答和工具调用过程"""
    trace = []
    answer = agent.run_turn(messages, body.message, trace=trace)
    return {"answer": answer, "trace": trace}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
