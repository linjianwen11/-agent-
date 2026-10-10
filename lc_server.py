# lc_server.py —— langchain 版网页后端:提供页面 + 聊天接口,复用 lc_agent
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import lc_agent

app = FastAPI(title="生理信号分析 Agent (LangChain)")

# 页面路径跟着文件走,避免"从别的目录启动就找不到"
INDEX_HTML = Path(__file__).parent / "index.html"

# 本地单会话:对话记忆放内存(BaseMessage 列表),服务重启即清空
messages = lc_agent.start_session()


class ChatBody(BaseModel):
    message: str


@app.get("/")
def index():
    """首页:返回聊天界面(复用手写版的前端)"""
    return FileResponse(INDEX_HTML)


@app.post("/api/chat")
def chat(body: ChatBody):
    """聊天接口:调 lc_agent 核心,返回回答和工具调用过程"""
    trace = []
    answer = lc_agent.run_turn(messages, body.message, trace=trace)
    return {"answer": answer, "trace": trace}


if __name__ == "__main__":
    # 用 8001 端口,与手写版 server.py 的 8000 并行不冲突
    uvicorn.run(app, host="127.0.0.1", port=8001)