import re

### chunking，把文本分块

def chunk_markdown(text):
    """
    把Markdown文本分块,返回 [{"title":..., "text":...}, ...]
    """
    chunks = []
    cur_title, cur_lines = None, []
    for line in text.splitlines():
        if line.startswith("## "):
            if cur_title is not None:
                chunks.append({"title": cur_title, "text": "\n".join(cur_lines)})
            cur_title, cur_lines = line, []
        elif cur_title is not None:
            cur_lines.append(line)
    if cur_title is not None:
        chunks.append({"title": cur_title, "text": "\n".join(cur_lines)})
    return chunks

### embedding，把文本转换为向量

def clean_text(text):
    """ 移除文本中的特殊字符和空格 """
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text.lower())

def ngrams(text, n = 2):
    """ 生成 n-gram 序列 """
    return [text[i:i+n] for i in range(max(0, len(text)-n+1))]

def build_vocab(texts):
    """统计全局所有 n-gram，建成固定词表 order=>index """
    all_ngrams = set()
    for text in texts:
        all_ngrams.update(ngrams(clean_text(text)))
    return {g: i for i, g in enumerate(sorted(all_ngrams))}

def embed(text, vocab):
    """ 
    TF 向量：text 里每个n-gram 出现的次数， 定长 = len(vocab)， 再进行归一化
    """
    from numpy import zeros
    vec = zeros(len(vocab), dtype=float)
    for g in ngrams(clean_text(text)):
        if g in vocab:
            vec[vocab[g]] += 1
    norm = vec.sum() or 1.0
    return vec / norm


### 相似度查询

def cosine(a, b):
    """ 余弦相似度 """
    from numpy import dot, linalg
    return dot(a,b) / (linalg.norm(a) * linalg.norm(b) + 1e-9)

def retrieve(query, chunks, vectors,vocab, k=3):
    """计算 query与每个块的相似度，返回 top-k 的 （分数，块）"""
    qv = embed(query, vocab)
    scores = sorted(
        ((cosine(qv, v), c) for v, c in zip(vectors, chunks)),
        key = lambda x: x[0], reverse = True
    )
    return scores[:k]

# def generate_answer(query, chunks, vectors, vocab):
#     # 检索top-k的块
#     hits = retrieve(query, chunks, vectors,vocab, k=3)

#     # 拼context，把每块的text按index排好
#     context = "\n\n".join(f"[{i}] {c['text']}" for i, (s, c) in enumerate(hits))
    
#     # 拼prompt
#     prompt = ( "请只依据下面的参考资料回答用户问题，" "资料里没有的信息不要编造，明确说不知道。\n" + "资料:\n" + context + "\n\n问题: " + query)

#     # 调用大模型
#     from langchain_deepseek import ChatDeepSeek
#     from langchain_core.messages import HumanMessage
#     llm = ChatDeepSeek(model = "deepseek-chat")
#     reply = llm.invoke([HumanMessage(content=prompt)])
#     return reply.content

def generate_answer(query, chunks, vectors, vocab, max_rounds=3, k=3):
    """迭代式检索：检索→判断够不够→不够改写查询再检索→达上限说明缺什么"""
    from langchain_deepseek import ChatDeepSeek
    from langchain_core.messages import HumanMessage
    llm = ChatDeepSeek(model="deepseek-chat")

    current_q = query
    seen = set()  # 已见过的块，避免重复累积

    for i in range(max_rounds):
        hits = retrieve(current_q, chunks, vectors, vocab, k=k)

        # 只拼新块的文本，去掉见过的那部分，避免一轮比一轮臃肿
        new_hits = [c for s, c in hits if c["text"] not in seen]
        seen.update(c[ "text" ] for s, c in hits)
        context = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(new_hits))

        # 让 LLM 二选一：足够就答，不足就给 [NEED] 改写查询
        prompt = ("请只依据下面的资料回答。若资料足以回答，直接回答；"
                  "若不足，只输出一行 '[NEED] 改写后的查询词'，不要编造。\n"
                  "资料:\n" + context + "\n问题: " + current_q)
        reply = llm.invoke([HumanMessage(content=prompt)]).content

        if reply.startswith("[NEED]"):     # 不够 → 改写查询，下一轮重检索
            current_q = reply[len("[NEED]"):].strip()
            continue
        return reply                        # 够 → 直接返回答案

    # 达上限仍不足：让 LLM 说明缺什么信息
    prompt = ("依据上面的资料回答，若资料不足，请明确说明缺少哪些信息"
              "才能回答用户问题。\n问题: " + current_q)
    return llm.invoke([HumanMessage(content=prompt)]).content

if __name__ == "__main__":
    # 读资料
    from pathlib import Path
    text = Path("面试笔记.md").read_text(encoding="utf-8")

    # 分块
    chunks = chunk_markdown(text)

    # 新建词表 给每块算向量
    voc = build_vocab([c["text"] for c in chunks])
    vectors = [embed(c["text"], voc) for c in chunks]

    # 循环问query
    while True:
        q = input("请输入问题: ").strip()
        if q in ["exit", "quit"]:
            break
        print(generate_answer(q, chunks, vectors, voc))
