
import agent

def main():
    print("生理信号分析 Agent 就绪，输入 'exit' 退出")
    messages = [{"role": "system", "content":agent.SYSTEM_PROMPT}]

    while True:
        user_input = input("请输入: ")
        if user_input.strip().lower() in ("exit", "quit", "退出"):
            break

        answer = agent.run_turn(messages, user_input)
        print("\nAgent", answer, "\n")

if __name__ == "__main__":
    main()
    