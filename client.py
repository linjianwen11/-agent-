# client.py —— 只负责"发请求给大模型、拿回结果"
import requests
import config

def chat(messages, tools=None):
    body = {"model": config.MODEL, "messages": messages}
    if tools:
        body["tools"] = tools
    resp = requests.post(
        config.BASE_URL + "/chat/completions",
        headers={"Authorization": "Bearer " + config.API_KEY,
                 "Content-Type": "application/json"},
        json=body,
    )
    data = resp.json()
    if not resp.ok or "error" in data:
        print("状态码:", resp.status_code)
        print("API 原始响应:", data)
    return data["choices"][0]["message"]