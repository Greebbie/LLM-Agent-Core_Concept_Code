---
name: enterprise-knowledge-assistant
description: Company-internal assistant for HR policy, product pricing, technical API docs, and order/inventory/notification workflows. Use deterministic guards for ORD/SKU/notification requests before routing between RAG, MCP tools, and direct answers.
allowed-tools: [mcp__enterprise-demo__query_order, mcp__enterprise-demo__check_inventory, mcp__enterprise-demo__send_notification]
model: claude-3-5-sonnet
version: "1.0"
---

# Enterprise Knowledge Assistant Skill

This is the Day 5 capstone packaged as a reusable Skill. It combines RAG, deterministic route guards, MCP tool calls, review, and evaluation. The teaching point is not "write a bigger prompt"; it is defining clear boundaries and verifying them with eval cases.

## Architecture

```text
User question
  -> Route guards + PlannerAgent
  -> one of:
       - Agentic RAG for HR/product/API knowledge
       - MCP tool call for orders, inventory, notifications
       - Direct answer or safe refusal
  -> ReviewerAgent quality check
  -> Final answer
```

All major steps are wrapped with Langfuse-compatible `@observe` spans. Full implementation notes are in `reference/architecture.md`.

## When To Use

- "入职 7 年有几天年假？"
- "API 限流是多少？"
- "StarLink 企业版多少钱？"
- "查订单 ORD-002"
- "SKU-A100 库存还有多少？"

The skill should refuse or defer when the knowledge base does not cover the topic, rather than guessing.

## Programmatic Entry Point

```python
from pipeline import upgraded_pipeline

result = upgraded_pipeline("入职 7 年有几天年假？")
print(result["path"])
print(result["answer"])
```

Return shape:

```python
{
    "path": "rag" | "mcp" | "direct",
    "answer": str,
    "review": str,
}
```

## Evaluation

`eval.py` provides `batch_eval(dataset_path)` and `reference/eval_cases.jsonl` contains the default teaching eval set. The eval is intentionally small; its purpose is to catch routing and grounding regressions during the course.

## Configuration

```bash
DASHSCOPE_API_KEY=           # fill with your own key
LLM_BACKEND=dashscope
EMBEDDING_BACKEND=dashscope
LANGFUSE_PUBLIC_KEY=         # optional; falls back to MockObserver
LANGFUSE_SECRET_KEY=         # optional
```

## Limitations

- The demo knowledge base is in `pipeline.py:KNOWLEDGE_DOCS`; production should load documents from a managed store.
- The MCP server is local and educational; production should add auth, audit logs, and tenant isolation.
- The reviewer is a lightweight quality gate, not a formal safety system.
