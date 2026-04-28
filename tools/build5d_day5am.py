"""Build Day 5 上午 · Agentic RAG notebook."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_lib import (
    save_nb, make_md, make_code, add_tag,
    make_path_fix_cell, make_lecture_note,
    load_nb, cell_source,
)

OUTPUT = Path("assets/enterprise_5days/instructor/Day5_上午_Agentic_RAG.ipynb")


def build():
    cells = []

    cells.append(make_md("""# Day 5 上午 · Agentic RAG（3h）

## 学习目标

1. 复盘 Day 3 基础 RAG 的 4 大失败模式
2. **Self-RAG**：让模型自己反思检索质量、不够则重查
3. **CRAG (Corrective RAG)**：用打分器纠正坏检索 + query rewriting
4. **Hybrid Retrieval**：Vector + BM25 + RRF 融合
5. **Cross-Encoder Reranker**：用 bge-reranker 做精排
6. **MMR**：多样性去重，避免 top-k 全是同一个意思

## 前置

- Day 3 上午基础 RAG (向量检索 + 失败模式手册)
- Day 3 下午 Capstone (RAG + Agent 集成)
- 已 `pip install rank-bm25 sentence-transformers>=3.0`
"""))

    cells.append(make_lecture_note(
        title="""Day 5 上午 · Agentic RAG（3h）""",
        duration_min=180,
        opener="""问：『基础 RAG 检索到 5 段似是而非的内容，模型直接拼答。这样真的能用？』 → 引出 Agentic RAG = RAG + 自我反思 + 主动纠错。""",
        key_points=[
            """**Self-RAG**：模型每次回答前评估检索是否足够，不够主动重查（自适应循环）""",
            """**CRAG**：打分器 + 三档处理（高分用 / 中分纠正 / 低分丢弃 + fallback）""",
            """**Hybrid = Vector + BM25**：互补 (Vector 抓语义，BM25 抓精确词) → RRF 融合""",
            """**Cross-Encoder Rerank**：top-10 → top-3 精排，比双塔强很多""",
            """**MMR**：top-k 之间也算相似度，去重避免冗余""",
        ],
        misconceptions=[
            """学员以为更复杂 RAG = 更好 → 强调每个组件都加延迟/成本，按场景选""",
            """学员以为 BM25 老古董 → 强调对专有词 / 数字 / ID 反而 BM25 更稳""",
        ],
        interaction="""现场让学员给 1 个『公司专有缩写』+ 1 个『模糊概念』，跑 Vector vs BM25 vs Hybrid 3 路看差异。""",
        if_short_on_time="""跳过 MMR (只保留 reranker)，保 Self-RAG + CRAG + Hybrid 主线。""",
    ))

    cells.append(make_path_fix_cell())

    cells.append(make_code("""# 导入：LLM + Embedding + RAG 基础设施
from utils.config import env
from utils.embedding_backend import SimpleVectorStore
import numpy as np
import json
from collections import defaultdict

llm = env.get_llm()
embedder = env.get_embedder()
print(f"✓ LLM 与 Embedder 就位")

# 准备一个企业小知识库（混合 HR / 技术 / 产品 文档片段）
KNOWLEDGE_DOCS = [
    {"id": "hr_01", "text": "公司年假政策：入职 5 年以下每年 5 天，5-10 年 10 天，10 年以上 15 天。", "category": "hr"},
    {"id": "hr_02", "text": "病假需出示三甲医院证明，全薪连续不超过 30 天。", "category": "hr"},
    {"id": "tech_01", "text": "API 限流：免费版 60 req/min，企业版 6000 req/min。可申请扩容。", "category": "tech"},
    {"id": "tech_02", "text": "API 鉴权使用 Bearer Token；token 由 CONSOLE 生成，30 天过期。", "category": "tech"},
    {"id": "tech_03", "text": "出现 429 (Too Many Requests) 时建议指数退避重试，3 次失败后告警。", "category": "tech"},
    {"id": "prod_01", "text": "StarLink 基础版 199 元/月，支持 5 路并发；企业版 1999 元/月，支持 100 路并发。", "category": "product"},
    {"id": "prod_02", "text": "StarView 是一款数据可视化产品，支持仪表板、自动报表与告警。", "category": "product"},
    {"id": "prod_03", "text": "SKU-A100 是 StarLink 入门套件，包含 1 个网关 + 5 个传感器，售价 4999 元。", "category": "product"},
]

# 建一个简单 vector store
vector_store = SimpleVectorStore()
docs_text = [d["text"] for d in KNOWLEDGE_DOCS]
docs_meta = [{"id": d["id"], "category": d["category"]} for d in KNOWLEDGE_DOCS]
vector_store.add_documents(docs_text, embedder, metadatas=docs_meta)
print(f"✓ 知识库就位：{len(KNOWLEDGE_DOCS)} 个文档")
"""))

    # ── Part 1: 复盘失败模式 ──
    cells.append(make_md("""---

## Part 1 · 基础 RAG 的 4 大失败模式（15 min）

回顾 Day 3 上午我们识别的 RAG 失败模式：

| 失败 | 表现 | Agentic RAG 怎么解 |
|---|---|---|
| **OOD (越界)** | 问题超出知识库 | Self-RAG 反思+confidence 阈值 |
| **召回不准** | 检索结果与问题相关度低 | Reranker + Hybrid Retrieval |
| **跨文档** | 答案要拼接多文档 | Self-RAG 多轮 + MMR 多样性 |
| **冗余** | top-5 全是同义内容 | MMR 强制多样化 |

下面 4 个 Part 一一对应这些解药。先看基础版的痛点：
"""))

    cells.append(make_code("""# 演示：基础 RAG 在『模糊查询』和『需要多文档』时的表现
def basic_rag(query, k=3):
    \"\"\"最简单的 RAG：检索 + 拼接 + 让 LLM 答\"\"\"
    results = vector_store.search(query, embedder, top_k=k)
    if not results:
        return "未找到相关内容"
    context = "\\n".join(f"- {r['document']}" for r in results)
    prompt = f"基于下面信息回答问题:\\n{context}\\n\\n问题: {query}"
    return llm.generate(prompt, temperature=0.1).strip()

# 案例 1：精确查询（基础 RAG 通常够用）
print("Q: 入职 7 年有几天年假？")
print(f"A: {basic_rag('入职 7 年有几天年假？')}\\n")

# 案例 2：跨文档拼接（基础 RAG 易漏）
print("Q: API 出现 429 错误怎么办？背景是什么？")
print(f"A: {basic_rag('API 出现 429 错误怎么办？背景是什么？')}\\n")

# 案例 3：OOD（基础 RAG 会硬答）
print("Q: 公司有没有医疗保险？")
print(f"A: {basic_rag('公司有没有医疗保险？')}")
"""))

    # ── Part 2: Self-RAG ──
    cells.append(make_md("""---

## Part 2 · Self-RAG：模型自我反思检索质量（45 min）

**核心思想**：模型每次拿到检索结果后**自评**：
1. 检索结果**相关吗**？
2. 信息**够回答吗**？
3. 不够就**重新检索**（换 query / 加 keyword）

```
Query → Retrieve → 反思: 够不够？
                    ├─ 够 → 生成答案
                    └─ 不够 → 重写 query → 再 Retrieve（循环最多 N 次）
```
"""))

    cells.append(make_code("""# Self-RAG 实现
def self_rag(query, max_iterations=3, k=3):
    \"\"\"自反思循环：判断检索质量，不够就改写 query 重查\"\"\"
    history = []
    current_query = query
    for it in range(max_iterations):
        results = vector_store.search(current_query, embedder, top_k=k)
        context = "\\n".join(f"- [{r['metadata']['id']}] {r['document']}" for r in results) if results else "(无)"

        # 反思：检索是否够回答原问题？
        reflect_prompt = f'''原始问题: {query}
当前检索 query: {current_query}
检索到的内容:
{context}

请评估：检索结果是否足够回答原始问题？
- 如果够，回复 'SUFFICIENT'，然后另起一行写最终答案。
- 如果不够，回复 'INSUFFICIENT'，然后另起一行写**改写后的 query**（更具体 / 加同义词 / 拆子问题）。'''
        reflection = llm.generate(reflect_prompt, temperature=0.0).strip()
        history.append({"iter": it + 1, "query": current_query, "reflection": reflection[:100]})
        if reflection.startswith("SUFFICIENT"):
            answer = reflection.split("\\n", 1)[1].strip() if "\\n" in reflection else "(LLM 未提供答案)"
            return {"answer": answer, "iterations": it + 1, "history": history, "status": "sufficient"}
        # INSUFFICIENT → 取改写后的 query
        if "\\n" in reflection:
            current_query = reflection.split("\\n", 1)[1].strip()
        else:
            break  # 没给 query 改写，退出
    # 最大次数耗尽，用最后一次结果硬答
    final_prompt = f"基于以下信息回答（信息不全请如实说）:\\n{context}\\n\\n问题: {query}"
    answer = llm.generate(final_prompt, temperature=0.1).strip()
    return {"answer": answer, "iterations": max_iterations, "history": history, "status": "max_iter"}


# Demo: OOD case
print("=" * 60); print("Self-RAG: '公司有没有医疗保险？' (OOD)"); print("=" * 60)
r = self_rag("公司有没有医疗保险？", max_iterations=2)
print(f"\\n答案: {r['answer']}")
print(f"轮数: {r['iterations']}  状态: {r['status']}")
for h in r['history']:
    print(f"  iter {h['iter']}: query='{h['query']}'  反思={h['reflection'][:60]}")
"""))

    # ── Exercise 1: Self-RAG ──
    cells.append(make_code('''# ============================================================
# 练习 1 | Self-RAG with Confidence Threshold
# ============================================================
#
# 【基础】（人人必做，10 min）
#   实现 self_rag_basic(query, k=3)：跑一轮 Self-RAG（max_iter=2）
#
# 【进阶】（技术学员选做，15 min）
#   实现 self_rag_with_confidence(query, threshold=0.6)：
#   - 每轮检索后让 LLM 给 confidence 0-1 分（"我对这个答案的把握有多大"）
#   - confidence < threshold 自动重查
#   - 返回 (answer, final_confidence, iterations)
# ============================================================

def self_rag_basic(query, k=3):
    """【基础】调 self_rag(...)"""
    # ↓↓↓ 【基础】填空（约 2 行）↓↓↓
    return self_rag(query, max_iterations=2, k=k)
    # ↑↑↑ 【基础】结束 ↑↑↑


def self_rag_with_confidence(query, threshold=0.6, max_iter=3, k=3):
    """【进阶】confidence threshold 决定是否继续重查"""
    # ↓↓↓ 【进阶】填空（约 18 行）↓↓↓
    current_query = query
    for it in range(max_iter):
        results = vector_store.search(current_query, embedder, top_k=k)
        context = "\\n".join(f"- {r['document']}" for r in results) if results else "(无)"
        # 让 LLM 答 + 自评 confidence
        prompt = f"""根据下面信息回答问题。最后一行单独输出 'CONFIDENCE: <0-1>' 表示你对答案的把握度。

信息:
{context}

问题: {query}"""
        response = llm.generate(prompt, temperature=0.1).strip()
        # 解析 confidence
        conf = 0.5  # default
        for line in reversed(response.split("\\n")):
            if "CONFIDENCE" in line.upper():
                try:
                    conf = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
                break
        if conf >= threshold:
            return {"answer": response, "confidence": conf, "iterations": it + 1, "status": "confident"}
        # 不够 → 改写 query 重试
        rewrite = llm.generate(
            f"问题 '{query}' 检索结果不够（confidence={conf}），请改写一个更具体的 query。只输出新 query。",
            temperature=0.3,
        ).strip()
        current_query = rewrite
    return {"answer": response, "confidence": conf, "iterations": max_iter, "status": "max_iter"}
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】self_rag_basic"); print("=" * 56)
    try:
        r = self_rag_basic("入职 5 年年假几天？")
        assert "answer" in r and "iterations" in r
        print(f"  iter={r['iterations']}  status={r['status']}")
        print(f"  answer: {r['answer'][:120]}")
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】self_rag_with_confidence (threshold=0.6)"); print("=" * 56)
    try:
        r = self_rag_with_confidence("StarLink 企业版 vs 基础版差别？", threshold=0.6, max_iter=2)
        print(f"  conf={r['confidence']:.2f}  iter={r['iterations']}  status={r['status']}")
        print(f"  answer: {r['answer'][:120]}")
        assert 0 <= r['confidence'] <= 1
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''))

    # ── Part 3: CRAG ──
    cells.append(make_md("""---

## Part 3 · CRAG (Corrective RAG)：纠正坏检索（30 min）

**核心思想**：用一个**轻量打分器**评估每条检索结果的质量，分三档：

| 分数 | 处理 |
|---|---|
| **High (>0.7)** | 直接用 |
| **Medium (0.3-0.7)** | 改写 query 重检索 |
| **Low (<0.3)** | 抛弃 + fallback（web search / 安全模板） |

CRAG vs Self-RAG：
- Self-RAG 让 LLM 自己评估（贵 + 慢，但准）
- CRAG 用轻量 scorer (可以是小模型 / embedding 相似度)（快但精度低）

工业上常**两者结合**：先 CRAG 粗筛，再 Self-RAG 反思。
"""))

    cells.append(make_code("""# CRAG 简化实现：用 embedding 相似度做 scorer
def crag(query, k=3, high=0.7, low=0.3):
    \"\"\"基于 embedding 相似度三档处理\"\"\"
    results = vector_store.search(query, embedder, top_k=k)
    if not results:
        return {"answer": "[Fallback] 知识库未覆盖此问题，建议联系业务部门或查官方文档", "tier": "no_results"}

    scores = [r["score"] for r in results]
    avg = sum(scores) / len(scores)

    if avg > high:
        # High: 直接用
        ctx = "\\n".join(f"- {r['document']}" for r in results)
        ans = llm.generate(f"基于下面信息答:\\n{ctx}\\n\\n问题: {query}", temperature=0.1).strip()
        return {"answer": ans, "tier": "high", "avg_score": avg}
    elif avg > low:
        # Medium: 改写 query 重检索
        rewrite = llm.generate(
            f"问题 '{query}' 检索效果一般，请改写一个更具体的 query (加专有词 / 替换同义词)。只输出新 query。",
            temperature=0.3,
        ).strip()
        new_results = vector_store.search(rewrite, embedder, top_k=k)
        ctx = "\\n".join(f"- {r['document']}" for r in new_results) if new_results else "(无)"
        ans = llm.generate(f"基于下面信息答:\\n{ctx}\\n\\n问题: {query}", temperature=0.1).strip()
        return {"answer": ans, "tier": "medium-rewritten", "rewrite": rewrite, "avg_score": avg}
    else:
        # Low: fallback
        return {"answer": "[Fallback] 检索结果可信度太低，建议人工介入", "tier": "low-fallback", "avg_score": avg}


# Demo
for q in ["入职 5 年有几天年假?", "API 限流是多少?", "公司有没有团建活动?"]:
    print(f"\\nQ: {q}")
    r = crag(q)
    print(f"  tier={r['tier']}  avg_score={r.get('avg_score', 0):.3f}")
    print(f"  A: {r['answer'][:100]}")
"""))

    # ── Exercise 2: CRAG ──
    cells.append(make_code('''# ============================================================
# 练习 2 | CRAG with Query Rewriter
# ============================================================
#
# 【基础】（人人必做，10 min）
#   实现 crag_basic(query)：直接调 crag(...)
#
# 【进阶】（技术学员选做，15 min）
#   实现 crag_with_decomposition(query)：
#   - 复杂 query (含『和』『以及』『同时』『分别') 先用 LLM 拆成 2-3 个子 query
#   - 每个子 query 跑 CRAG
#   - 最后 LLM 综合所有结果给统一答案
# ============================================================

def crag_basic(query):
    """【基础】"""
    # ↓↓↓ 【基础】填空（约 1 行）↓↓↓
    return crag(query)
    # ↑↑↑ 【基础】结束 ↑↑↑


def crag_with_decomposition(query):
    """【进阶】拆复合 query 后并行 CRAG，最后综合"""
    # ↓↓↓ 【进阶】填空（约 14 行）↓↓↓
    # Step 1: 让 LLM 决定要不要拆
    decomp_prompt = f"""判断下面问题是否包含多个独立子问题。
- 如果是，输出 JSON: {{"sub_queries": ["...", "..."]}}（2-4 个子问题）
- 如果不是，输出 JSON: {{"sub_queries": []}}

问题: {query}
只输出 JSON。"""
    raw = llm.generate(decomp_prompt, temperature=0).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        decomp = json.loads(raw)
        subs = decomp.get("sub_queries", [])
    except Exception:
        subs = []
    if not subs:
        return {"answer": crag(query)["answer"], "decomposed": False}
    # Step 2: 每个子问题跑 CRAG
    sub_results = []
    for sq in subs:
        r = crag(sq)
        sub_results.append({"q": sq, "a": r["answer"][:200], "tier": r["tier"]})
    # Step 3: 综合
    combined = "\\n".join(f"Q: {s['q']}\\nA: {s['a']}" for s in sub_results)
    final = llm.generate(
        f"原问题: {query}\\n\\n各子问题答:\\n{combined}\\n\\n请综合给一个连贯回答。",
        temperature=0.2,
    ).strip()
    return {"answer": final, "decomposed": True, "sub_results": sub_results}
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】crag_basic"); print("=" * 56)
    try:
        r = crag_basic("API 限流多少？")
        assert "answer" in r and "tier" in r
        print(f"  tier: {r['tier']}")
        print(f"  answer: {r['answer'][:100]}")
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】crag_with_decomposition"); print("=" * 56)
    try:
        # 复合问题
        r = crag_with_decomposition("入职 5 年有几天年假？病假怎么算？API 限流多少？")
        print(f"  decomposed: {r['decomposed']}")
        if r["decomposed"]:
            for sr in r["sub_results"]:
                print(f"    - {sr['q']} → tier={sr['tier']}")
        print(f"  最终: {r['answer'][:200]}")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''))

    # ── Part 4: Hybrid Retrieval ──
    cells.append(make_md("""---

## Part 4 · Hybrid Retrieval：Vector + BM25 + RRF（45 min）

**为什么 hybrid**：

| Retrieval 方式 | 强项 | 弱项 |
|---|---|---|
| **Vector (语义)** | 语义近的能召回 | 专有词 / 数字 / ID 抓不到 |
| **BM25 (关键词)** | 精确词匹配强 | 同义改写就漏 |

工业上 BM25 + Vector **互补**，用 **RRF (Reciprocal Rank Fusion)** 融合排名：

```
RRF_score(doc) = Σ_retrievers  1 / (k + rank_in_retriever)
```

`k` 是常数（通常 60），rank 越靠前贡献越大。
"""))

    cells.append(make_code("""# 装 BM25 + 实现 hybrid
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠ rank_bm25 未装。运行 `pip install rank-bm25` 后重启 kernel")


def tokenize_chinese(text):
    \"\"\"简易中文 tokenizer：按字 + 简单切英文词\"\"\"
    import re
    # 中文按字 + 英文按词
    tokens = []
    for chunk in re.findall(r'[a-zA-Z0-9]+|[\\u4e00-\\u9fa5]', text):
        if chunk[0].isascii():
            tokens.append(chunk.lower())
        else:
            tokens.append(chunk)
    return tokens


# 建 BM25 索引
if BM25_AVAILABLE:
    docs_tokens = [tokenize_chinese(d["text"]) for d in KNOWLEDGE_DOCS]
    bm25 = BM25Okapi(docs_tokens)
    print(f"✓ BM25 索引就位 ({len(docs_tokens)} 文档)")


def search_bm25(query, k=5):
    if not BM25_AVAILABLE:
        return []
    q_tokens = tokenize_chinese(query)
    scores = bm25.get_scores(q_tokens)
    idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [{"document": KNOWLEDGE_DOCS[i]["text"], "score": float(scores[i]),
              "metadata": {"id": KNOWLEDGE_DOCS[i]["id"]}} for i in idx if scores[i] > 0]


def search_vector(query, k=5):
    return vector_store.search(query, embedder, top_k=k)


def rrf_fusion(*retriever_results, k=60, top_k=5):
    \"\"\"Reciprocal Rank Fusion: 合并多个 retriever 的结果\"\"\"
    scores = defaultdict(float)
    docs_by_id = {}
    for results in retriever_results:
        for rank, r in enumerate(results):
            doc_id = r["metadata"]["id"]
            scores[doc_id] += 1.0 / (k + rank + 1)
            docs_by_id[doc_id] = r
    sorted_ids = sorted(scores, key=lambda d: -scores[d])[:top_k]
    return [{**docs_by_id[did], "rrf_score": scores[did]} for did in sorted_ids]


# Demo: 对比 3 路
query = "SKU-A100 多少钱？"  # 含专有 ID
print(f"Q: {query}")
print(f"\\nVector:")
for r in search_vector(query, k=3):
    print(f"  {r['score']:.3f} | {r['document'][:50]}")
print(f"\\nBM25:")
for r in search_bm25(query, k=3):
    print(f"  {r['score']:.3f} | {r['document'][:50]}")
print(f"\\nHybrid (RRF):")
for r in rrf_fusion(search_vector(query, k=5), search_bm25(query, k=5), top_k=3):
    print(f"  {r['rrf_score']:.4f} | {r['document'][:50]}")
"""))

    # ── Exercise 3: Hybrid + Reranker ──
    cells.append(make_code('''# ============================================================
# 练习 3 | Hybrid Retrieval + Cross-Encoder Reranker
# ============================================================
#
# 【基础】（人人必做，10 min）
#   实现 hybrid_search(query, top_k=3)：vector + bm25，RRF 融合
#
# 【进阶】（技术学员选做，15 min）
#   实现 hybrid_with_rerank(query, top_k=3)：
#   - 先 hybrid 取 top_10
#   - 用 sentence-transformers 的 cross-encoder 给每个 (query, doc) 打分
#   - 按 cross-encoder 分数取最终 top_k
#   - 提示：from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')
#   - 如果模型没下载，第一次会从 HF 拉 (~400MB)
# ============================================================
_reranker_cache = {"model": None}


def hybrid_search(query, top_k=3):
    """【基础】hybrid retrieval (vector + bm25 + RRF)"""
    # ↓↓↓ 【基础】填空（约 3 行）↓↓↓
    v = search_vector(query, k=10)
    b = search_bm25(query, k=10)
    return rrf_fusion(v, b, top_k=top_k)
    # ↑↑↑ 【基础】结束 ↑↑↑


def hybrid_with_rerank(query, top_k=3):
    """【进阶】hybrid 取 top10 后用 cross-encoder rerank"""
    # ↓↓↓ 【进阶】填空（约 14 行）↓↓↓
    candidates = hybrid_search(query, top_k=10)
    if not candidates:
        return []
    # Lazy load cross-encoder
    if _reranker_cache["model"] is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker_cache["model"] = CrossEncoder("BAAI/bge-reranker-base")
        except Exception as e:
            print(f"  ⚠ Cross-Encoder 加载失败 ({e}); 退化为 hybrid 直接 top_k")
            return candidates[:top_k]
    reranker = _reranker_cache["model"]
    pairs = [(query, c["document"]) for c in candidates]
    rerank_scores = reranker.predict(pairs)
    # 按 reranker 分数排序
    enriched = [{**c, "rerank_score": float(s)} for c, s in zip(candidates, rerank_scores)]
    enriched.sort(key=lambda x: -x["rerank_score"])
    return enriched[:top_k]
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】hybrid_search"); print("=" * 56)
    try:
        results = hybrid_search("StarLink 企业版价格", top_k=3)
        assert len(results) > 0
        print(f"  Top {len(results)}:")
        for r in results:
            print(f"    rrf={r['rrf_score']:.4f} | {r['document'][:60]}")
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】hybrid_with_rerank"); print("=" * 56)
    try:
        print("  (首次跑会下载 bge-reranker-base ~400MB，请耐心)")
        results = hybrid_with_rerank("API 怎么处理 429 错误", top_k=3)
        for r in results:
            sc = r.get("rerank_score", "n/a")
            print(f"    rerank={sc if isinstance(sc, str) else f'{sc:.4f}'} | {r['document'][:60]}")
        print("  💡 cross-encoder 比 RRF 准很多，但慢 5-10x；适合 top-10 → top-3 精排")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''))

    # ── Exercise 4: MMR ──
    cells.append(make_code('''# ============================================================
# 练习 4 | MMR (Maximal Marginal Relevance) 多样性去重
# ============================================================
#
# 【基础】（人人必做，10 min）
#   实现 mmr_select(query_vec, candidates, lambda_=0.7, top_k=3)：
#   - candidates: list of {"vec": np.array, "doc": str}
#   - 每步选『相似 query 最高 + 与已选最不重复』的 doc
#   - score = lambda * sim(q, doc) - (1-lambda) * max sim(doc, already_picked)
#
# 【进阶】（技术学员选做，10 min）
#   把 mmr_select 跟 hybrid_search 串起来：
#   实现 hybrid_diverse(query, top_k=3, lambda_=0.6) → 用 MMR 在 hybrid top_10 中选最多样的 top_k
# ============================================================
def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def mmr_select(query_vec, candidates, lambda_=0.7, top_k=3):
    """【基础】MMR 选 top_k"""
    # ↓↓↓ 【基础】填空（约 12 行）↓↓↓
    selected = []
    selected_idxs = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < top_k:
        best_score, best_idx = -float("inf"), None
        for i in remaining:
            relevance = cosine_sim(query_vec, candidates[i]["vec"])
            redundancy = max(
                (cosine_sim(candidates[i]["vec"], candidates[j]["vec"]) for j in selected_idxs),
                default=0.0,
            )
            score = lambda_ * relevance - (1 - lambda_) * redundancy
            if score > best_score:
                best_score, best_idx = score, i
        selected.append({**candidates[best_idx], "mmr_score": best_score})
        selected_idxs.append(best_idx)
        remaining.remove(best_idx)
    return selected
    # ↑↑↑ 【基础】结束 ↑↑↑


def hybrid_diverse(query, top_k=3, lambda_=0.6):
    """【进阶】hybrid + MMR"""
    # ↓↓↓ 【进阶】填空（约 8 行）↓↓↓
    candidates = hybrid_search(query, top_k=10)
    if not candidates:
        return []
    q_vec = np.array(embedder.embed([query])[0])
    docs_with_vec = []
    for c in candidates:
        v = np.array(embedder.embed([c["document"]])[0])
        docs_with_vec.append({"vec": v, "doc": c["document"], "metadata": c.get("metadata", {})})
    return mmr_select(q_vec, docs_with_vec, lambda_=lambda_, top_k=top_k)
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】mmr_select"); print("=" * 56)
    try:
        # 构造 5 个 doc，2 组各 2 个相似 + 1 个独特
        np.random.seed(0)
        q = np.array([1, 0, 0])
        cands = [
            {"vec": np.array([0.9, 0.1, 0]), "doc": "topic A v1"},
            {"vec": np.array([0.85, 0.15, 0]), "doc": "topic A v2"},  # 与 v1 几乎一样
            {"vec": np.array([0.7, 0.0, 0.3]), "doc": "topic B"},
            {"vec": np.array([0.65, 0.0, 0.35]), "doc": "topic B v2"},
            {"vec": np.array([0.5, 0.5, 0.5]), "doc": "topic C"},
        ]
        selected = mmr_select(q, cands, lambda_=0.5, top_k=3)
        chosen_docs = [s["doc"] for s in selected]
        print(f"  MMR 选出: {chosen_docs}")
        # 应该选出来自 3 组 (A / B / C) 而不是同组 2 个
        assert len(set(d.split(" v")[0] for d in chosen_docs)) >= 2, "MMR 应有多样性"
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】hybrid_diverse"); print("=" * 56)
    try:
        results = hybrid_diverse("公司有什么产品和价格", top_k=3, lambda_=0.6)
        print(f"  Top {len(results)} (MMR 多样化):")
        for r in results:
            print(f"    mmr={r['mmr_score']:+.3f} | {r['doc'][:60]}")
        assert len(results) > 0
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''))

    # ── 总结 ──
    cells.append(make_md("""---

## Day 5 上午 总结

### 你学到了
1. **Self-RAG**：自反思循环，confidence 阈值控制
2. **CRAG**：三档 (high/medium/low) + query rewriting + fallback
3. **Hybrid Retrieval**：Vector + BM25 + RRF 融合
4. **Cross-Encoder Rerank**：top-10 → top-3 精排
5. **MMR**：强制多样性，避免冗余

### 何时用哪个

| 场景 | 推荐 |
|---|---|
| 简单查询 / 文档库小 | 基础 RAG 够 |
| 模糊查询 / 多文档拼接 | Self-RAG |
| 噪声大 / 有 OOD 风险 | CRAG（fallback 安全） |
| 含专有词 / 数字 / ID | Hybrid (Vector + BM25) |
| 严苛精度需求 | Hybrid + Cross-Encoder Rerank |
| 长输出避免重复 | + MMR |

### 与下午 Capstone 衔接

下午 Day 5 下午 **升级 Capstone** 会把今天学的 Agentic RAG **替换** Day 3 Capstone 里的基础 RAG 层，看准确率有多大提升（用 Bonus_B 评测框架打分）。

### 推荐资料
- Self-RAG 原论文：『Self-RAG: Learning to Retrieve, Generate, and Critique』
- CRAG 原论文：『Corrective Retrieval Augmented Generation』
- BAAI bge-reranker：HuggingFace 上免费 SOTA reranker
- LangChain / LlamaIndex 都有现成 Hybrid + Reranker 实现，工程化时可直接用
"""))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    save_nb(nb, OUTPUT)

    nb2 = load_nb(OUTPUT)
    n_tagged = 0
    for c in nb2["cells"]:
        if c["cell_type"] != "code":
            continue
        src = cell_source(c)
        first = src.strip().split("\n")[0]
        if first.startswith("# =====") and "练习" in src[:200]:
            add_tag(c, "fillin")
            add_tag(c, "batch5")
            n_tagged += 1
    save_nb(nb2, OUTPUT)
    print(f"✓ Built {OUTPUT}")
    print(f"  Total cells: {len(cells)} | tagged {n_tagged} fillin")


if __name__ == "__main__":
    build()
