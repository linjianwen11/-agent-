一个分析疲劳的agent，主要是调用deepseek，然后用工具，返回message给deepseek
流程是：用户提需求-> 上传到llm-> 然后llm返回tool_calls 调用工具-> 结果变成一个message-> 大模型解读
<img width="808" height="1284" alt="image" src="https://github.com/user-attachments/assets/ecbf07cb-8ecb-48df-9aeb-3176c7eec1a1" />
<img width="809" height="1277" alt="image" src="https://github.com/user-attachments/assets/b5ed7c16-f7ab-40fb-9cca-ac7d9d56f042" />

2026 9.15 使用最基本的agent_loop 下一步学一下langchain

2026.10.10 已学完langchain langgraph rag等知识，感觉agent的概念很宽泛啊，这些东西本质上都是一个框架，想找到一个能够达成loop的路子，但是我暂时没有找到这些东西能够在哪里产生价值

怎么说呢，就是本质上都是在帮助我们更好的使用ai大模型对吧，因为agent框架本质上是基于让大模型，更像一个超级智能的人，老美最近搞出一个把si替代ai，蛮有意思的，确实现在不应该叫人工智能了，super intelligence确实更加贴切

agent框架我觉得真正有价值的，是当能够让robot理解世界的ai出来之时，能够在极短的时间内，让他能够替代大部分人的工作，所以真正有意义的研究在于world llm 或者是 agi，都是一个东西

robot + agent + wllm 是我认为的终极答案，暂时，或许很多年之后回看，不，或许是几个月后回看，我大概率也会改变自己的想法。
