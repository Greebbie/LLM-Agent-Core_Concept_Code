---
name: enterprise-knowledge-assistant
description: 企业知识助手：回答 HR 政策、产品、技术 API、订单/库存等内部问题。生产版先用确定性规则识别 ORD/SKU/通知等强结构请求，再在 RAG、MCP tools、direct LLM 之间路由。Use for company-internal policy/product/API/order questions.
allowed-tools: [mcp__enterprise-demo__query_order, mcp__enterprise-demo__check_inventory, mcp__enterprise-demo__send_notification]
model: claude-3-5-sonnet
version: "1.0"
---

# Enterprise Knowledge Assistant Skill

The full Day-5-Capstone packaged as a reusable Skill. This is the **"5 天合体"**
deliverable: Multi-Agent + MCP + Agentic RAG + LLMOps wrapped into a single
folder you can drop into `~/.claude/skills/` or any Claude Agent SDK project.

## Architecture

```
User question
    ↓
Route guard + PlannerAgent (decides path: rag / mcp / direct)
    ↓
┌────────────────┬─────────────────┬────────────────┐
│  Agentic RAG   │  MCP tool call  │  Direct answer │
│  (Hybrid+Self) │  (订单/库存等)   │                │
└────────────────┴─────────────────┴────────────────┘
    ↓
ReviewerAgent (quality check)
    ↓
Final answer
```

All steps wrapped in Langfuse-compatible `@observe` for trace + token tracking.
Full architecture details: `reference/architecture.md` (loaded on demand).

## When to use

- "What is the company's leave policy?"
- "Look up order ORD-002"
- "How does the API rate limit work?"
- "Hi, can you help me?"

The skill auto-routes. Strong identifiers (`ORD-*`, `SKU-*`, notification intent)
are handled by deterministic guards before falling back to the LLM Planner.

## How to invoke programmatically

```python
from pipeline import upgraded_pipeline
result = upgraded_pipeline("入职 7 年有几天年假？")
print(result["answer"])
```

Returns:
```python
{
    "path": "rag" | "mcp" | "direct",
    "answer": str,
    "review": str,
}
```

## Evaluation

`eval.py` provides `batch_eval(dataset_path)` returning success rate +
per-category accuracy. See `reference/eval_cases.jsonl` for default cases.

## Configuration

Required environment variables:

```bash
DASHSCOPE_API_KEY=           # fill with your own key, or use another LLM provider
LLM_BACKEND=dashscope
EMBEDDING_BACKEND=dashscope
LANGFUSE_PUBLIC_KEY=         # optional — falls back to MockObserver
LANGFUSE_SECRET_KEY=         # optional
```

## Limitations

- Knowledge base is hardcoded in `pipeline.py:KNOWLEDGE_DOCS` (production should load from external store)
- MCP server runs in-process — production should use stdio transport to remote
- No auth / multi-tenant isolation — add before exposing to real users

## Versioning

- `0.x`: development
- `1.0`: trained on 5-day Capstone, ready for internal demo
- `2.0+`: future — include Vision / Voice / Reasoning models
