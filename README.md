一个分析疲劳的agent，主要是调用deepseek，然后用工具，返回message给deepseek
流程是：用户提需求-> 上传到llm-> 然后llm返回tool_calls 调用工具-> 结果变成一个message-> 大模型解读
<img width="808" height="1284" alt="image" src="https://github.com/user-attachments/assets/ecbf07cb-8ecb-48df-9aeb-3176c7eec1a1" />
<img width="809" height="1277" alt="image" src="https://github.com/user-attachments/assets/b5ed7c16-f7ab-40fb-9cca-ac7d9d56f042" />

2026 9.15 使用最基本的agent_loop 下一步学一下langchain
