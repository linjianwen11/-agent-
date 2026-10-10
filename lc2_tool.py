import lc_agent

messages = lc_agent.start_session()

while True:
    user = input("用户：")
    if user.strip().lower() in ("exit", "quit"):
        break
    answer = lc_agent.run_turn(messages, user)

    print("Agent:", answer)
