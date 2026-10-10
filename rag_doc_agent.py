# rag_doc_agent.py —— 可中断/可恢复的 RAG 文档助手
#
# 场景:用户提问 → 检索完成 → 暂停(进程可退出) → 重启进程 → 从断点继续生成答案 + 自我审查
# 关键:状态经 LangGraph SQLite 持久化 checkpointer 落盘,跨进程保留;不同 thread_id = 独立会话互不影响。

import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage

# 复用 simple_rag 的检索能力,不重写索引/embedding
import simple_rag

DB_PATH = Path(__file__).parent / "rag_assistant.db"
DOC_PATH = Path(__file__).parent.parent / "面试笔记.md"   # 语料在 d:\javabird\ 下,不在本文件目录
LLM = ChatDeepSeek(model="deepseek-chat")

# 索引较重,只构建一次(模块级缓存);由检索节点闭包使用
_INDEX = None


def get_index():
    """惰性构建:读文档 → 分块 → 建词表 → 每块向量。返回 (chunks, vectors, vocab)"""
    global _INDEX
    if _INDEX is None:
        text = DOC_PATH.read_text(encoding="utf-8")
        chunks = simple_rag.chunk_markdown(text)
        vocab = simple_rag.build_vocab([c["text"] for c in chunks])
        vectors = [simple_rag.embed(c["text"], vocab) for c in chunks]
        _INDEX = (chunks, vectors, vocab)
    return _INDEX


class DocAssistantState(TypedDict):
    question: str
    context: str   # 检索到的 top-k 拼接,跨进程随 checkpoint 保留
    answer: str
    review: str


def retrieve_node(state):
    """检索:用 simple_rag 取 top-k 并拼 context,写入 state(后续要靠它恢复)"""
    chunks, vectors, vocab = get_index()
    hits = simple_rag.retrieve(state["question"], chunks, vectors, vocab, k=3)
    context = "\n\n".join(f"[{i}] {c['text']}" for i, (s, c) in enumerate(hits))
    return {"context": context}


def pause_node(state):
    """中断点:检索完成后挂起,状态已落盘,进程可正常退出。重启后用 Command(resume=..) 续跑"""
    interrupt({"question": state["question"], "context": state["context"]})
    return {}


def generate_node(state):
    """生成:基于(已持久化的)context 让大模型作答"""
    prompt = ("请只依据下面的资料回答用户问题,资料里没有的信息不要说。\n"
              "资料:\n" + state["context"] + "\n问题: " + state["question"])
    return {"answer": LLM.invoke([HumanMessage(content=prompt)]).content}


def review_node(state):
    """自我审查:让大模型基于同一资料检查并完善上面的回答"""
    prompt = ("请审核下面这份回答,检查它是否严格基于资料、有无遗漏或编造,"
              "然后给出修正后的完整答案。\n资料:\n" + state["context"]
              + "\n\n原回答:\n" + state["answer"])
    return {"review": LLM.invoke([HumanMessage(content=prompt)]).content}


def build_graph(checkpointer):
    """组装图:START → 检索 → 中断 → 生成 → 审查 → END"""
    g = StateGraph(DocAssistantState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("pause", pause_node)
    g.add_node("generate", generate_node)
    g.add_node("review", review_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "pause")
    g.add_edge("pause", "generate")
    g.add_edge("generate", "review")
    g.add_edge("review", END)
    return g.compile(checkpointer=checkpointer)


def make_app():
    """创建带 SQLite 持久化 checkpointer 的已编译应用。返回 (app, saver_context)。
    saver 用 context manager 持有连接,期间 app 都应在其生命周期内使用。"""
    ctx = SqliteSaver.from_conn_string(str(DB_PATH))  # 3.x 返回 context manager
    return build_graph(ctx.__enter__()), ctx


def run_start(app, thread_id, question):
    """段 1:提问并跑到中断点即暂停,打印信息后由进程退出。
    本版 langgraph 的 interrupt() 不抛异常:invoke 正常返回,结果 dict 带 __interrupt__ 键表示已挂起。"""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"question": question}, config)
    if "__interrupt__" in result:
        state = result
        print("已暂停(检索完成,状态已落盘)。可关闭程序,之后运行 resume 继续。")
        print(f"[interrupt 信息] 会话={thread_id} 问题={question}")
        if state.get("context"):
            print(f"[已保存 context 长度] {len(state['context'])} 字符")
        return
    print("（未触发中断,请检查图结构）")


def run_resume(app, thread_id):
    """段 2:同一 thread_id 用 Command(resume=..) 续跑,完成生成+审查"""
    config = {"configurable": {"thread_id": thread_id}}
    command = Command(resume="继续")
    result = app.invoke(command, config)
    print("\n===== 最终答案 =====")
    print(result["answer"])
    print("\n===== 自我审查后 =====")
    print(result["review"])


if __name__ == "__main__":
    app, ctx = make_app()
    try:
        # 用法: python rag_doc_agent.py start "问题" [thread_id]
        #       python rag_doc_agent.py resume [thread_id]
        action = sys.argv[1] if len(sys.argv) > 1 else "start"

        if action == "start":
            thread = sys.argv[3] if len(sys.argv) > 3 else "session-a"
            question = sys.argv[2] if len(sys.argv) > 2 else input("请输入问题: ").strip()
            run_start(app, thread, question)
        elif action == "resume":
            thread = sys.argv[2] if len(sys.argv) > 2 else "session-a"
            run_resume(app, thread)
        else:
            print("用法: start \"问题\" | resume")
    finally:
        ctx.__exit__(None, None, None)