"""Build Day 5 下午 · LLMOps + 升级 Capstone notebook."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_lib import (
    save_nb, make_md, make_code, add_tag,
    make_path_fix_cell, make_lecture_note,
    load_nb, cell_source,
)

OUTPUT = Path("assets/enterprise_5days/instructor/Day5_下午_LLMOps与生产Capstone.ipynb")


def build():
    cells = []

    cells.append(make_md("""# Day 5 下午 · LLMOps + 升级 Capstone（3h）

## 学习目标

**块 1（LLMOps，90 min）**
1. 理解可观测性 3 大支柱：trace / metric / alert
2. 用 **Langfuse** 给 LLM/Agent 调用加 `@observe`
3. 跨 Agent 的 **trace_id 传递**（OTEL 风格）

**块 2（升级 Capstone，90 min + 演示 + 复盘）**
4. 把 Day 3 Capstone（基础 RAG+Agent）升级为：
   - **Multi-Agent**（Day4 上午 Planner/Worker/Reviewer）
   - **MCP 工具层**（Day4 下午 server）
   - **Agentic RAG**（Day5 上午 Hybrid + Self-RAG）
   - **LLMOps 全程 trace**（Langfuse 看 token / 延迟 / 失败）
5. 用 `data/eval_dataset.jsonl` 评测对比基础版 vs 升级版

## 前置

- Day 1-3 全部内容
- Day 4 上午 Multi-Agent / Day 4 下午 MCP
- Day 5 上午 Agentic RAG
- 已 `pip install langfuse` (可选 — 没装会自动 fallback 到 MockObserver)
"""))

    cells.append(make_lecture_note(
        title="""Day 5 下午 · LLMOps + 升级 Capstone（3h）""",
        duration_min=180,
        opener="""问：『Capstone 上线第 1 个月，老板问"我们烧了多少 token / p95 延迟多少 / 哪些问题答错"，你能秒答吗？』 → 引出可观测性。""",
        key_points=[
            """**3 大支柱**：trace（每条调用链）/ metric（QPS/cost/latency）/ alert（异常告警）""",
            """**Langfuse @observe** 一行装饰器搞定 trace；dashboard 直接看""",
            """**trace_id 跨 Agent 传递**：父子 span 关系让你看『哪个 agent 哪步慢』""",
            """**升级 Capstone**：本节是『5 天合体』演示，所有前面学的组件全用上""",
            """讲师版**预跑跑通**，学员重点改 1-2 个模块（不要求从 0 写）""",
        ],
        misconceptions=[
            """学员以为 LLMOps 等于 ChatGPT 的『查询历史』 → 强调 trace = 完整结构化调用链""",
            """学员以为 token usage 自动有 → 强调要主动 record_tokens 才能看到""",
        ],
        interaction="""现场让学员看 Capstone 的 trace tree，找出最慢的一步并讨论怎么优化。""",
        if_short_on_time="""跳过 inter-rater agreement 进阶；保 Langfuse 集成 + Capstone 全集成 + 评测主线。""",
    ))

    cells.append(make_path_fix_cell())

    cells.append(make_code("""# 导入：所有 5 天组件 + 新加 observability
from utils.config import env
from utils.multi_agent import Message, MessageType, BaseAgent, Orchestrator
from utils.mcp_helpers import EduMCPServer, EduMCPClient, ToolDef, tool_from_function
from utils.embedding_backend import SimpleVectorStore
from utils.observability import observer, observe, span, get_backend
import json, time, sys
from pathlib import Path

# 引入 Day5 上午写的 Agentic RAG 工具 (重新建 vector store + bm25)
sys.path.insert(0, str(Path('mcp_server_demo')))

llm = env.get_llm()
embedder = env.get_embedder()

print(f"✓ 5 天组件全就位")
print(f"✓ Observability backend: {get_backend()}  (mock=本地; langfuse=已配 LANGFUSE_*)")
"""))

    # ── Block 1.1: LLMOps 3 支柱 ──
    cells.append(make_md("""---

## 块 1 · LLMOps 可观测性

### 3 大支柱

| 支柱 | 能回答的问题 | 对应工具 |
|---|---|---|
| **Trace（调用链）** | 这条用户请求经过了哪些 LLM/tool 调用？哪步慢？ | Langfuse / LangSmith |
| **Metric（指标）** | 整体 QPS / token 消耗 / p95 延迟？ | Prometheus / Datadog |
| **Alert（告警）** | 失败率突变？token 暴涨？ | PagerDuty / 企业微信 |

LLM 调用 vs 普通 web 后端的特殊性：
- **每次调用花钱**：没 trace 你不知道谁烧的
- **延迟天然慢**：1-10s 是常态；要看 p95/p99 而不是 avg
- **失败模式多**：超时 / 限流 / 模型生成 bad output / tool 失败 / hallucination
- **质量难量化**：HTTP 200 不代表答对；要 LLM-as-Judge / 人工抽检

### 我们今天用什么

`utils/observability.py` 已写好两个后端：
- **Langfuse**（如果已 `pip install langfuse` + 配 env）→ 真实 dashboard
- **MockObserver**（默认）→ 内存记录，本地 print 看
"""))

    cells.append(make_code("""# 演示：用 @observe 自动追踪
observer.reset()

@observe("retrieve_doc")
def my_retrieve(query):
    time.sleep(0.05)  # 模拟检索延迟
    return f"docs for '{query}'"

@observe("generate_answer")
def my_generate(query, docs):
    time.sleep(0.1)
    return f"Answer based on {docs[:30]}"

@observe("rag_pipeline")
def my_rag(query):
    docs = my_retrieve(query)
    return my_generate(query, docs)

# 跑一次
result = my_rag("什么是 LoRA？")
print(f"Result: {result}\\n")

print("Trace tree:")
observer.print_tree()
print("\\nSummary:")
print(json.dumps(observer.summary(), indent=2))
"""))

    # ── Exercise 1: Langfuse @observe ──
    cells.append(make_code('''# ============================================================
# 练习 1 | @observe 包装 LLM 调用 + 计时
# ============================================================
#
# 【基础】（人人必做，10 min）
#   实现 traced_llm_call(prompt)：用 @observe("llm.generate") 包装一次 LLM 调用
#   返回 LLM 的输出
#
# 【进阶】（技术学员选做，10 min）
#   实现 traced_pipeline(query)：组合 retrieve → generate，每步独立 trace
#   - retrieve 用 vector_store（如有）或 mock
#   - 用 with span() 手动控制；记录 token 数 (估算 = len(prompt) / 4)
# ============================================================

@observe("llm.generate")
def traced_llm_call(prompt):
    """【基础】wrap LLM call with trace"""
    # ↓↓↓ 【基础】填空（约 1 行）↓↓↓
    return llm.generate(prompt, temperature=0.1).strip()
    # ↑↑↑ 【基础】结束 ↑↑↑


def traced_pipeline(query):
    """【进阶】retrieve + generate 各自 span，记录 token 估算"""
    # ↓↓↓ 【进阶】填空（约 14 行）↓↓↓
    with span("pipeline.full", input={"query": query}) as p:
        # Retrieve step
        with span("pipeline.retrieve", input={"query": query}) as r:
            time.sleep(0.05)  # mock
            docs = f"mock docs about '{query}'"
            observer.record_tokens(prompt_tokens=len(query) // 4, completion_tokens=0)
        # Generate step
        prompt = f"基于 {docs} 答 {query}"
        with span("pipeline.generate", input={"prompt_len": len(prompt)}) as g:
            answer = llm.generate(prompt, temperature=0.1).strip()
            observer.record_tokens(
                prompt_tokens=len(prompt) // 4,
                completion_tokens=len(answer) // 4,
                cost_usd=0.0001,  # mock pricing
            )
        return answer
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】traced_llm_call"); print("=" * 56)
    try:
        observer.reset()
        ans = traced_llm_call("什么是 RAG？一句话")
        assert isinstance(ans, str) and len(ans) > 0
        print(f"  LLM 答: {ans[:120]}")
        print(f"  Trace 数: {len(observer.spans)}")
        observer.print_tree()
        assert len(observer.spans) >= 1
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】traced_pipeline (含 span + tokens)"); print("=" * 56)
    try:
        observer.reset()
        ans = traced_pipeline("LoRA 与全参微调差别？")
        print(f"  Answer: {ans[:120]}")
        print(f"\\n  Trace tree:")
        observer.print_tree()
        summary = observer.summary()
        print(f"\\n  Summary: {json.dumps(summary, indent=2)}")
        assert summary["tokens"]["total"] > 0
        assert summary["n_total_spans"] >= 3  # full + retrieve + generate
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''))

    # ── Exercise 2: trace propagation ──
    cells.append(make_code('''# ============================================================
# 练习 2 | Multi-Agent 间的 trace_id 传递
# ============================================================
#
# 【基础】（人人必做，10 min）
#   定义 2 个 Agent (PlannerAgent, WorkerAgent)，每个都用 @observe 包装其 receive()
#   PlannerAgent → WorkerAgent 顺序调用
#
# 【进阶】（技术学员选做，10 min）
#   实现 trace tree 可视化：
#   - 用 with span("user_request") as root: 包裹整个流程
#   - 让 Planner 和 Worker 的 span 都嵌套在 root 下
#   - 最后输出 tree 看到嵌套关系
# ============================================================

@observe("planner.run")
def planner_run(task):
    """【基础】Planner 接收任务"""
    return llm.generate(f"把『{task}』拆成 2 个子任务，编号列出。", temperature=0.1).strip()


@observe("worker.run")
def worker_run(plan):
    """【基础】Worker 执行计划"""
    return llm.generate(f"按计划执行：\\n{plan}\\n\\n给出简短结果。", temperature=0.2).strip()


def basic_two_agent_pipeline(task):
    """【基础】依次调 planner → worker"""
    # ↓↓↓ 【基础】填空（约 3 行）↓↓↓
    plan = planner_run(task)
    result = worker_run(plan)
    return {"plan": plan, "result": result}
    # ↑↑↑ 【基础】结束 ↑↑↑


def two_agent_with_root_span(task):
    """【进阶】整体放入 root span，看到嵌套 trace tree"""
    # ↓↓↓ 【进阶】填空（约 6 行）↓↓↓
    with span("user_request", input={"task": task}) as root:
        plan = planner_run(task)
        result = worker_run(plan)
        return {"plan": plan, "result": result, "root_span": root.name}
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】basic_two_agent_pipeline"); print("=" * 56)
    try:
        observer.reset()
        r = basic_two_agent_pipeline("写一个简短的 Python 缓存装饰器")
        print(f"  Plan: {r['plan'][:100]}")
        print(f"  Result: {r['result'][:100]}")
        print(f"\\n  Trace tree:")
        observer.print_tree()
        # 应有 2 个根级 span (planner + worker)
        assert len(observer.spans) == 2
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】嵌套 root span"); print("=" * 56)
    try:
        observer.reset()
        r = two_agent_with_root_span("写一个简短的 Python 缓存装饰器")
        print(f"  Trace tree (含 root span 嵌套):")
        observer.print_tree()
        # 应只有 1 个根 span，下面套 2 个 children
        assert len(observer.spans) == 1
        root = observer.spans[0]
        assert len(root.children) == 2
        print(f"\\n  根 span '{root.name}' 含 {len(root.children)} 个子调用")
        print("  💡 生产场景下，trace_id 会跨进程透传 (用 OTEL header 传)，这样微服务也能看完整 trace")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''))

    # ── Block 2: 升级 Capstone ──
    cells.append(make_md("""---

## 块 2 · 升级 Capstone：5 天合体

### 项目题目：升级版企业知识助手

**架构**：

```
                           ┌───────────────────┐
                           │   User Question    │
                           └─────────┬─────────┘
                                     ▼
                       ┌──────────────────────────┐
              with span("capstone")
                       │   PlannerAgent (Multi-Agent) │
                       │   决定走哪条路径           │
                       └──────┬─────────┬─────────┘
                              │         │
                ┌─────────────┘         └────────────┐
                ▼                                     ▼
       ┌─────────────────┐                ┌────────────────────┐
       │  Agentic RAG     │                │  MCP Tool Call    │
       │  (Hybrid + Self) │                │  (订单/库存/通知) │
       └────────┬─────────┘                └─────────┬─────────┘
                │                                     │
                └──────────┐         ┌────────────────┘
                           ▼         ▼
                   ┌────────────────────────────┐
                   │  ReviewerAgent             │
                   │  检查回答质量              │
                   └─────────┬──────────────────┘
                             ▼
                   ┌──────────────────────┐
                   │  Final Answer        │
                   └──────────────────────┘

           整个过程：Langfuse 全程 trace
```

**4 大组件来源**：
| 组件 | 来源 |
|---|---|
| Multi-Agent (Planner/Worker/Reviewer) | Day 4 上午 + `utils/multi_agent.py` |
| MCP Server (订单/库存/通知) | Day 4 下午 + `mcp_server_demo/server.py` |
| Agentic RAG (Hybrid + Self) | Day 5 上午（在此简化版用 vector + 简单 self-check） |
| Observability | Day 5 下午（本节）+ `utils/observability.py` |

讲师版下面**完整跑通一遍**给学员看；学员在 Exercise 3 自己改其中 1-2 个组件。
"""))

    cells.append(make_code("""# ── Step 1: 建知识库 (复用 Day5 上午的 KNOWLEDGE_DOCS) ──
KNOWLEDGE_DOCS = [
    {"id": "hr_01", "text": "公司年假政策：入职 5 年以下每年 5 天，5-10 年 10 天，10 年以上 15 天。", "category": "hr"},
    {"id": "hr_02", "text": "病假需出示三甲医院证明，全薪连续不超过 30 天。", "category": "hr"},
    {"id": "tech_01", "text": "API 限流：免费版 60 req/min，企业版 6000 req/min。", "category": "tech"},
    {"id": "tech_02", "text": "API 鉴权使用 Bearer Token；token 由 CONSOLE 生成，30 天过期。", "category": "tech"},
    {"id": "tech_03", "text": "出现 429 (Too Many Requests) 时建议指数退避重试。", "category": "tech"},
    {"id": "prod_01", "text": "StarLink 基础版 199 元/月，5 路并发；企业版 1999 元/月，100 路并发。", "category": "product"},
    {"id": "prod_03", "text": "SKU-A100 是 StarLink 入门套件，包含 1 个网关 + 5 个传感器，售价 4999 元。", "category": "product"},
]

vector_store = SimpleVectorStore()
vector_store.add_documents(
    [d["text"] for d in KNOWLEDGE_DOCS],
    embedder,
    metadatas=[{"id": d["id"], "category": d["category"]} for d in KNOWLEDGE_DOCS],
)
print(f"✓ 知识库就位：{len(KNOWLEDGE_DOCS)} 条")
"""))

    cells.append(make_code("""# ── Step 2: 建 MCP server (复用 Day4 下午的 mcp_server_demo) ──
from server import build_server as build_mcp_server
mcp_server = build_mcp_server()
mcp_client = EduMCPClient(user_id="capstone-user")
mcp_client.connect(mcp_server)
print(f"✓ MCP server 就位：{len(mcp_client.list_all_tools())} 个 tool")
"""))

    cells.append(make_code("""# ── Step 3: 建 Multi-Agent (Planner / RAG_Worker / MCP_Worker / Reviewer) ──
PLANNER_CAPSTONE_PROMPT = '''你是 **路由 Planner**。
判断用户问题应走哪条路径，输出 JSON：
- 知识/文档/政策类 → {"path": "rag"}
- 订单/库存/通知 → {"path": "mcp"}
- 闲聊/无法判断 → {"path": "direct"}

只输出 JSON。'''

RAG_WORKER_PROMPT = '''你是 **RAG Worker**。基于检索到的文档简洁答用户问题。
如果文档不够，明确说『信息不足』。'''

MCP_WORKER_PROMPT = '''你是 **MCP Worker**。
用户问题: {query}
可用 MCP tools: {tools}
请输出 JSON: {{"tool": "...", "arguments": {{...}}}}。'''

REVIEWER_CAPSTONE_PROMPT = '''你是 **回答审查者**。
判断这个回答是否：(1) 正面回答问题  (2) 不含编造  (3) 长度适中
APPROVE 或 REJECT，附一句理由。'''

planner = BaseAgent("Planner", llm, PLANNER_CAPSTONE_PROMPT, temperature=0.0)
rag_worker = BaseAgent("RAGWorker", llm, RAG_WORKER_PROMPT, temperature=0.1)
reviewer = BaseAgent("Reviewer", llm, REVIEWER_CAPSTONE_PROMPT, temperature=0.0)

print("✓ 4 Agent 就位 (Planner / RAGWorker / MCPWorker(动态) / Reviewer)")
"""))

    cells.append(make_code("""# ── Step 4: 写组合主循环 (含 trace) ──
@observe("capstone.route")
def route_query(query):
    \"\"\"Planner 决定走 rag / mcp / direct\"\"\"
    plan_msg = Message("user", "Planner", MessageType.TASK, query)
    raw = planner.receive(plan_msg).payload.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(raw).get("path", "direct")
    except Exception:
        return "direct"

@observe("capstone.rag_branch")
def rag_branch(query):
    results = vector_store.search(query, embedder, top_k=3)
    if not results:
        return "[Fallback] 知识库未覆盖"
    ctx = "\\n".join(f"- {r['document']}" for r in results)
    rag_msg = Message("Planner", "RAGWorker", MessageType.TASK,
                       f"问题: {query}\\n\\n检索到:\\n{ctx}")
    return rag_worker.receive(rag_msg).payload

@observe("capstone.mcp_branch")
def mcp_branch(query):
    tools = mcp_client.list_all_tools()
    tools_desc = "; ".join(f"{t['name']}({list(t['parameters']['properties'].keys())})" for t in tools)
    decide_prompt = MCP_WORKER_PROMPT.format(query=query, tools=tools_desc)
    raw = llm.generate(decide_prompt, temperature=0.0).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        plan = json.loads(raw)
        result = mcp_client.call(mcp_server.name, plan["tool"], **plan["arguments"])
        return f"调用 MCP tool {plan['tool']}({plan['arguments']}) 得到: {result}"
    except Exception as e:
        return f"[MCP 调用失败] {e}"

@observe("capstone.direct_answer")
def direct_branch(query):
    return llm.generate(f"用一两句话简短回答: {query}", temperature=0.3).strip()

@observe("capstone.review")
def review_answer(query, answer):
    rev_msg = Message("user", "Reviewer", MessageType.REVIEW,
                      f"问题: {query}\\n\\n回答: {answer}")
    return reviewer.receive(rev_msg).payload

@observe("capstone.upgraded_pipeline")
def upgraded_pipeline(query):
    \"\"\"完整升级 Capstone 流程\"\"\"
    path = route_query(query)
    if path == "rag":
        answer = rag_branch(query)
    elif path == "mcp":
        answer = mcp_branch(query)
    else:
        answer = direct_branch(query)
    review = review_answer(query, answer)
    return {"path": path, "answer": answer, "review": review}


# Demo: 跑 3 个不同类型的 query
observer.reset()
for q in [
    "入职 8 年有几天年假？",            # → rag
    "查一下订单 ORD-002",                # → mcp
    "你好",                              # → direct
]:
    print("=" * 60)
    print(f"Q: {q}")
    r = upgraded_pipeline(q)
    print(f"  路径: {r['path']}")
    print(f"  答: {r['answer'][:150]}")
    print(f"  Review: {r['review'][:80]}")

print("\\n" + "=" * 60)
print("Trace tree (3 个 query 合并):")
observer.print_tree()
print("\\nSummary:")
print(json.dumps(observer.summary(), indent=2))
"""))

    # ── Capstone Checkpoint 1 ──
    cells.append(make_code('''# ============================================================
# Checkpoint 1 | 全栈 Capstone 跑通 + trace 验证
# ============================================================
#
# 【基础】（人人必做，10 min）
#   实现 capstone_quick_test()：跑 3 个 query (rag/mcp/direct 各一)，验证：
#   - 每个 query 都返回 {path, answer, review}
#   - observer 至少记录了 3 个根 span
#
# 【进阶】（技术学员选做，15 min）
#   实现 capstone_with_fallback(query)：
#   - 走 rag → 如果 review 含 'REJECT' 或 answer 含 'Fallback' → 退化到 direct
#   - 走 mcp → 如果调用失败 → 退化到 direct
#   - 返回 {path, fallback_used, answer}
# ============================================================

def capstone_quick_test():
    """【基础】跑 3 个 query 验证 pipeline + trace 工作"""
    # ↓↓↓ 【基础】填空（约 8 行）↓↓↓
    observer.reset()
    queries = ["StarLink 基础版价格", "查订单 ORD-001", "你好"]
    results = []
    for q in queries:
        r = upgraded_pipeline(q)
        results.append(r)
    return {"results": results, "n_traces": len(observer.spans), "summary": observer.summary()}
    # ↑↑↑ 【基础】结束 ↑↑↑


def capstone_with_fallback(query):
    """【进阶】RAG/MCP 失败时降级 direct"""
    # ↓↓↓ 【进阶】填空（约 14 行）↓↓↓
    path = route_query(query)
    fallback_used = False
    try:
        if path == "rag":
            answer = rag_branch(query)
            if "Fallback" in answer or "信息不足" in answer:
                answer = direct_branch(query)
                fallback_used = True
                path = "rag→direct"
        elif path == "mcp":
            answer = mcp_branch(query)
            if "失败" in answer:
                answer = direct_branch(query)
                fallback_used = True
                path = "mcp→direct"
        else:
            answer = direct_branch(query)
    except Exception as e:
        answer = direct_branch(query)
        fallback_used = True
        path = f"{path}→direct (error: {e})"
    return {"path": path, "fallback_used": fallback_used, "answer": answer}
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】capstone_quick_test"); print("=" * 56)
    try:
        result = capstone_quick_test()
        for r in result["results"]:
            assert "path" in r and "answer" in r and "review" in r
        print(f"  3 query 全部跑通")
        print(f"  根 span 数: {result['n_traces']}")
        print(f"  Summary: {result['summary']}")
        assert result["n_traces"] >= 3
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】capstone_with_fallback"); print("=" * 56)
    try:
        # 触发 OOD → fallback
        r1 = capstone_with_fallback("公司有没有团建活动？")
        print(f"  '公司有团建吗?' → path={r1['path']}, fallback={r1['fallback_used']}")
        # 正常 RAG
        r2 = capstone_with_fallback("入职 5 年年假几天？")
        print(f"  '入职 5 年年假?' → path={r2['path']}, fallback={r2['fallback_used']}")
        assert "path" in r1 and "answer" in r1
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''))

    # ── Capstone Checkpoint 2: Eval ──
    cells.append(make_code('''# ============================================================
# Checkpoint 2 | 用 eval_dataset 评测对比
# ============================================================
#
# 【基础】（人人必做，10 min）
#   实现 batch_eval(pipeline_fn, dataset)：
#   - 跑 dataset 中所有 query
#   - 检查 answer 是否包含 expected_keywords 中的至少 1 个
#   - 返回 success_rate + per-category accuracy
#
# 【进阶】（技术学员选做，10 min）
#   实现 inter_rater_check(query, answer, n=3)：
#   - 让 LLM 用同一 judge prompt 跑 3 次给分 (1-5)
#   - 返回 (avg_score, max_disagreement)
#   - max_disagreement >= 2 表示 judge 不稳定
# ============================================================
import random


def batch_eval(pipeline_fn, dataset_path):
    """【基础】关键词包含率 + 分 category 准确率"""
    # ↓↓↓ 【基础】填空（约 16 行）↓↓↓
    items = [json.loads(line) for line in open(dataset_path, encoding="utf-8")]
    results = []
    cat_correct = {}
    cat_total = {}
    for it in items:
        out = pipeline_fn(it["query"])
        answer = out.get("answer", "") if isinstance(out, dict) else str(out)
        ok = any(kw.lower() in answer.lower() for kw in it["expected_keywords"])
        cat = it["category"]
        cat_total[cat] = cat_total.get(cat, 0) + 1
        if ok:
            cat_correct[cat] = cat_correct.get(cat, 0) + 1
        results.append({"query": it["query"], "answer": answer[:100], "ok": ok, "cat": cat})
    success = sum(1 for r in results if r["ok"])
    per_cat = {c: cat_correct.get(c, 0) / cat_total[c] for c in cat_total}
    return {
        "success_rate": success / len(items),
        "per_category": per_cat,
        "details": results,
    }
    # ↑↑↑ 【基础】结束 ↑↑↑


def inter_rater_check(query, answer, n=3):
    """【进阶】N 次 judge 看一致性"""
    # ↓↓↓ 【进阶】填空（约 14 行）↓↓↓
    judge_prompt = f"""问题: {query}
回答: {answer}

请评分 1-5（1=完全错；5=完美准确）。只输出一个数字。"""
    scores = []
    for _ in range(n):
        raw = llm.generate(judge_prompt, temperature=0.0).strip()
        try:
            scores.append(int(raw[0]))
        except (ValueError, IndexError):
            pass
    if not scores:
        return {"avg": 0, "max_disagreement": 0, "scores": scores}
    return {"avg": sum(scores) / len(scores),
            "max_disagreement": max(scores) - min(scores),
            "scores": scores}
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】batch_eval (Capstone vs basic RAG)"); print("=" * 56)
    try:
        # 评测升级 Capstone
        eval_path = "data/eval_dataset.jsonl"
        result = batch_eval(upgraded_pipeline, eval_path)
        print(f"  升级 Capstone 准确率: {result['success_rate']:.0%}")
        print(f"  分类别:")
        for cat, acc in result["per_category"].items():
            print(f"    {cat:<10} {acc:.0%}")
        print(f"\\n  失败 case (前 3):")
        fails = [r for r in result["details"] if not r["ok"]][:3]
        for f in fails:
            print(f"    Q: {f['query']}")
            print(f"      A: {f['answer'][:80]}")
        assert "success_rate" in result
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】inter_rater_check"); print("=" * 56)
    try:
        check = inter_rater_check(
            "API 限流多少？",
            "免费版 60 req/min，企业版 6000 req/min。",
            n=3,
        )
        print(f"  3 次 judge 分数: {check['scores']}")
        print(f"  平均: {check['avg']:.2f}  |  最大分歧: {check['max_disagreement']}")
        if check["max_disagreement"] >= 2:
            print(f"  ⚠ judge 不稳定 (分歧 ≥ 2)，需要更明确的评分标准")
        else:
            print(f"  ✓ judge 一致性可接受")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''))

    # ── Capstone Checkpoint 3: 复盘 ──
    cells.append(make_code('''# ============================================================
# Checkpoint 3 | 系统级反思：『下周回公司能用上的一个改造点』
# ============================================================
#
# 【基础】（人人必做，5 min）
#   实现 reflect_on_capstone()：跑 Capstone + Eval，列 3 条最大失败 + 1 个改造建议
#
# 【进阶】（技术学员选做，10 min）
#   实现 trace_top_slow_paths(observer, n=3)：
#   - 看 observer 里的 trace tree
#   - 找出最慢的 N 个 span
#   - 给出优化建议（缓存 / 并行 / 降模型 / 等）
# ============================================================

def reflect_on_capstone():
    """【基础】跑评测 + 列失败 + 给改造建议"""
    # ↓↓↓ 【基础】填空（约 6 行）↓↓↓
    eval_result = batch_eval(upgraded_pipeline, "data/eval_dataset.jsonl")
    failures = [r for r in eval_result["details"] if not r["ok"]][:3]
    suggestion = "建议：扩充 ood 类别的 fallback 模板 + 给 Reviewer 加更严格的事实校验 prompt"
    return {
        "success_rate": eval_result["success_rate"],
        "top_3_failures": failures,
        "improvement_suggestion": suggestion,
    }
    # ↑↑↑ 【基础】结束 ↑↑↑


def trace_top_slow_paths(observer_obj, n=3):
    """【进阶】找最慢的 n 个 span 并给建议"""
    # ↓↓↓ 【进阶】填空（约 14 行）↓↓↓
    all_spans = []
    def collect(span_obj, depth=0):
        if span_obj.duration_ms is not None:
            all_spans.append({"name": span_obj.name, "depth": depth, "ms": span_obj.duration_ms})
        for child in span_obj.children:
            collect(child, depth + 1)
    for root in observer_obj.spans:
        collect(root)
    all_spans.sort(key=lambda x: -x["ms"])
    top = all_spans[:n]
    suggestions = []
    for s in top:
        if "rag" in s["name"].lower() or "retrieve" in s["name"].lower():
            tip = "考虑：缓存常见 query 的检索结果"
        elif "llm" in s["name"].lower() or "generate" in s["name"].lower():
            tip = "考虑：用更小模型 (qwen-turbo) / 减少 max_tokens"
        elif "mcp" in s["name"].lower():
            tip = "考虑：MCP 调用并发化"
        else:
            tip = "考虑：分析此 span 占比是否合理"
        suggestions.append({"span": s["name"], "ms": s["ms"], "tip": tip})
    return suggestions
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】reflect_on_capstone"); print("=" * 56)
    try:
        report = reflect_on_capstone()
        print(f"  整体准确率: {report['success_rate']:.0%}")
        print(f"  Top 3 失败:")
        for f in report["top_3_failures"]:
            print(f"    - {f['query']}")
        print(f"\\n  改造建议: {report['improvement_suggestion']}")
        assert "success_rate" in report
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】trace_top_slow_paths"); print("=" * 56)
    try:
        observer.reset()
        for q in ["StarLink 基础版多少钱？", "查订单 ORD-001"]:
            upgraded_pipeline(q)
        slow = trace_top_slow_paths(observer, n=3)
        print(f"  最慢的 3 个 span:")
        for s in slow:
            print(f"    {s['ms']:>7.1f}ms  {s['span']}")
            print(f"             → {s['tip']}")
        assert len(slow) >= 1
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''))

    # ── 总结 ──
    cells.append(make_md("""---

## Day 5 下午 总结

### 你学到了
1. **LLMOps 3 支柱**：trace / metric / alert
2. **Langfuse @observe + span 上下文管理**：一行装饰器搞定可观测
3. **trace_id 跨 Agent 传递**：父子 span 让你看『哪个 Agent 哪步慢』
4. **升级 Capstone**：5 天合体——Multi-Agent + MCP + Agentic RAG + LLMOps
5. **评测 + 反思**：用 eval dataset 量化质量、用 trace 找性能瓶颈

### 5 天回顾

| Day | 主题 |
|---|---|
| 1 | 文本→向量 + Transformer |
| 2 | 预训练 + SFT + LoRA + DPO + 评测 |
| 3 | RAG + Agent + 小 Capstone |
| 4 | Multi-Agent + MCP |
| 5 | Agentic RAG + LLMOps + 升级 Capstone |

### 下一步建议（出培训后）

1. **回公司挑 1 个真实场景**做迷你 Capstone（HR 助手 / 内部 wiki 问答 / 工单分流）
2. **接入真 Langfuse / OTEL**：先看 trace，再优化
3. **2026 还可以学**：Vision LLM / Voice Agent / Reasoning Models / Computer Use / Safety —— 这些是本课程**out of scope**，但都是工业热点
4. **持续读论文**：Anthropic / OpenAI 的 cookbook，LangChain / LlamaIndex 的 production guide

### 复盘环节

每位学员说一个『下周能用上的一个改造点』 → 5 天的最大收获 ✓
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
        if first.startswith("# =====") and ("练习" in src[:200] or "Checkpoint" in src[:200]):
            add_tag(c, "fillin")
            add_tag(c, "batch5")
            n_tagged += 1
    save_nb(nb2, OUTPUT)
    print(f"✓ Built {OUTPUT}")
    print(f"  Total cells: {len(cells)} | tagged {n_tagged} fillin")


if __name__ == "__main__":
    build()
