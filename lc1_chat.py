# lc1_chat.py —— chat model + message 无工具对话

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatDeepSeek(model="deepseek-chat")

messages = [SystemMessage(content="你是专业运动生理学助手，回答通俗语言")]

while True:
    user = input("用户：")
    if user.strip().lower() in ("exit", "quit"):
        break
    messages.append(HumanMessage(content=user))
    reply = llm.invoke(messages)
    print("Agent:", reply.content)
    messages.append(reply)