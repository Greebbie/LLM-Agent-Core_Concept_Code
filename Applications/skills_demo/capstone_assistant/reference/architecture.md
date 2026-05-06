# Enterprise Knowledge Assistant: Architecture Notes

This file is loaded on demand when the user asks how the capstone assistant works or how to extend it.

## Component Map

```text
User Question
    |
    v
Route Guards + PlannerAgent
    |
    +--> RAG Branch
    |      - retrieve relevant HR/product/API docs
    |      - answer with sources
    |
    +--> MCP Branch
    |      - query_order
    |      - check_inventory
    |      - send_notification
    |
    +--> Direct Branch
           - answer simple chat
           - refuse uncovered internal-policy questions

All branches -> ReviewerAgent -> Final Answer
```

## File Responsibilities

| File | Purpose |
|---|---|
| `SKILL.md` | Skill definition, routing description, usage workflow |
| `pipeline.py` | Agents, vector store, route guards, MCP calls, public `upgraded_pipeline()` |
| `eval.py` | Batch evaluation against `reference/eval_cases.jsonl` |
| `reference/eval_cases.jsonl` | Small regression set for routing and grounding |

## Routing Design

The production path uses deterministic guards before the LLM planner:

- `ORD-*` -> MCP `query_order`
- `SKU-*` with inventory intent -> MCP `check_inventory`
- notification intent -> MCP `send_notification`
- known HR/product/API terms -> RAG
- unsupported internal-policy topics -> direct safe refusal

The planner remains useful for ambiguous queries, but high-signal identifiers should not depend on model judgment.

## Extension Points

### Add A New Knowledge Area

1. Add documents to `KNOWLEDGE_DOCS` or replace it with a loader.
2. Add matching terms to `SUPPORTED_RAG_TERMS`.
3. Add eval cases to `reference/eval_cases.jsonl`.

### Add A New Tool

1. Add the tool in `mcp_server_demo/server.py`.
2. Update `MCP_WORKER_PROMPT`.
3. Add a deterministic extraction path when the tool has strong identifiers.
4. Add eval cases for the tool boundary.

### Improve Observability

Current spans:

- `skill.upgraded_pipeline`
- `skill.route`
- `skill.rag`
- `skill.mcp`
- `skill.direct`
- `skill.review`

In production, add token accounting, latency percentiles, error rates, and eval trend dashboards.

## Known Limitations

1. The document set is small and in-memory.
2. The MCP server is a teaching server, not a secure multi-tenant backend.
3. The reviewer catches obvious issues but should not be treated as a safety guarantee.
4. Eval coverage is intentionally small; real deployment needs cases from real logs.
