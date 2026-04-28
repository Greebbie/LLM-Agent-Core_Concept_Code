# Enterprise Knowledge Assistant — Architecture Deep Dive

> Loaded on demand by Claude when the user asks about implementation details
> ("how does this work?" / "what's inside?" / "how to extend?").

## Component Map

```
                    ┌─────────────────┐
                    │ User Question    │
                    └────────┬────────┘
                             ▼
                ┌──────────────────────┐
                │ PlannerAgent         │
                │ (routes by intent)   │
                └──┬───────────┬───┬───┘
                   │           │   │
       ┌───────────┘           │   └────────────┐
       ▼                       ▼                ▼
┌──────────────┐    ┌─────────────────┐   ┌──────────────┐
│ Agentic RAG  │    │ MCP Tool Call   │   │ Direct LLM   │
│ (Hybrid+Self)│    │ (订单/库存等)    │   │              │
└──────┬───────┘    └────────┬────────┘   └──────┬───────┘
       │                     │                   │
       └─────────┐    ┌──────┘    ┌──────────────┘
                 ▼    ▼    ▼
           ┌──────────────────────┐
           │ ReviewerAgent        │
           │ (quality gate)       │
           └────────┬─────────────┘
                    ▼
           ┌──────────────────┐
           │ Final Answer     │
           └──────────────────┘

Wrapped throughout by Langfuse @observe for trace + tokens + cost.
```

## File responsibilities

| File | Purpose |
|---|---|
| `SKILL.md` | Skill definition + when to invoke + workflow guide |
| `pipeline.py` | All agents, vector store, MCP client, branches, main `upgraded_pipeline()` |
| `eval.py` | Batch evaluation against `reference/eval_cases.jsonl` |
| `reference/architecture.md` | This file — loaded on demand |
| `reference/eval_cases.jsonl` | 10 evaluation queries with expected keywords |

## Extension points

### Add a new branch (e.g. "code")
1. Define a new agent with appropriate system_prompt
2. Add a `@observe("skill.code") def code_branch(query)` function
3. Add `"code"` to PlannerAgent's prompt options
4. Update routing in `upgraded_pipeline()`

### Swap the RAG strategy
- Replace `vector_store.search(query, top_k=3)` in `rag_branch` with a hybrid version (BM25 + Vector + RRF) — see Day 5 上午 Agentic RAG materials.

### Swap the LLM
- Set `LLM_BACKEND=openai` (or other) + corresponding `LLM_MODEL` in `.env`. Code unchanged.

### Add new MCP tools
- Edit `mcp_server_demo/server.py`, add a function, register as `ToolDef`. Capstone auto-discovers.

## Performance characteristics

Typical latency (p50 / p95) on dashscope qwen-plus, no caching:

| Branch | p50 | p95 | LLM calls |
|---|---|---|---|
| direct | 1.5s | 3s | 1 |
| rag | 3-4s | 6s | 2 (planner + rag_worker) + reviewer = 3 |
| mcp | 2-3s | 5s | 2 (planner + mcp decider) + reviewer = 3 |

Cost per query (qwen-plus rates): ~$0.001-0.005 USD depending on context size.

## Observability

Langfuse spans created per call:
- `skill.upgraded_pipeline` (root)
  - `skill.route` (planner LLM)
  - `skill.{rag,mcp,direct}` (chosen branch)
  - `skill.review` (reviewer LLM)

Token usage recorded via `observer.record_tokens()` if running with MockObserver,
or auto-tracked by Langfuse cloud / self-host.

## Known issues

1. Planner sometimes routes "查 SKU-A100 是什么" to `direct` instead of `mcp` — description in PlannerAgent's prompt could be sharper.
2. Reviewer is too lenient (rarely REJECTs) — temperature is 0 but prompt could explicitly require rejecting hallucinations.
3. No caching — same query repeated hits LLM every time. Add LRU cache on `route_query` for hot queries.
