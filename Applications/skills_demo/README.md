# Skills Demo

- `code_review/`: a compact skill for structured Python code review.
- `db_query/`: a skill that routes order, inventory, and notification requests to the demo MCP server.
- `capstone_assistant/`: the assistant built in App8.

`SKILL.md` tells the model when to use the skill and what workflow to follow. Optional helper scripts contain executable logic. `reference/` holds supporting material that should only be loaded when needed.

```
skills_demo/
├── code_review/
│   ├── SKILL.md
│   ├── helper.py
│   └── reference/checklist.md
├── db_query/
│   ├── SKILL.md
│   └── reference/sql_examples.md
└── capstone_assistant/
    ├── SKILL.md
    ├── pipeline.py
    ├── eval.py
    ├── answer_checks.py
    └── reference/
```

## SKILL.md and supporting files

1. `name`: short, stable identifier.
2. `description`: the routing contract. Write when to use it and what boundary it owns.
3. `Workflow`: 3-6 operational steps the model can follow.
4. `allowed-tools`: the tool surface the skill is allowed to call.
5. `reference/`: optional material loaded only when the task needs it.

Avoid vague descriptions such as "AI assistant". Prefer descriptions like "summarize meeting transcripts, extract decisions, and list action items with owners and deadlines".

## Skills And MCP

Skills tell the model what process to follow. MCP gives the model callable tools.

- `db_query` tells the model how to map natural language to `query_order`, `check_inventory`, and `send_notification`.
- `mcp_server_demo/server.py` exposes those tools over a small JSON-RPC stdio protocol.
- `capstone_assistant` combines RAG, deterministic route guards, MCP calls, and evaluation.

## Quick Check

From the repository root, run this in Python.

```python
from utils.skills_helpers import discover_skills, validate_skill

for skill in discover_skills("Applications/skills_demo"):
    result = validate_skill(skill.skill_dir)
    print(skill.name, result["ok"], skill.description[:70])
```
