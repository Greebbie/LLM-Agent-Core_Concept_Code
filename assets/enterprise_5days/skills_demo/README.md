# Skills Demo

This folder contains **3 example Anthropic Skills** used in Day 4 下午 (Skills 90min) and Day 5 下午 (Capstone-as-Skill 收尾).

```
skills_demo/
├── code_review/             # Day4 下午 练习 1 用 — 简单 skill 示例
├── db_query/                # Day4 下午 练习 3 用 — Skill 调 MCP server 的集成模式
└── capstone_assistant/      # Day5 下午 收尾 — 把整个升级 Capstone 打包成 Skill
```

## What is a Skill?

> **A Skill is a folder of reusable expertise that Claude can selectively load.**

Each skill folder contains:

- **`SKILL.md`** — required. YAML frontmatter (name + description) + body (instructions)
- **`*.py`** — optional helper scripts Claude can call
- **`reference/`** — optional. Docs loaded on-demand (progressive disclosure)

## How a Skill is invoked

Claude (Claude Code / Claude.ai / Claude Agent SDK) goes through:

1. **Discovery**: scan a skills directory, read all `SKILL.md` frontmatter
2. **Routing**: when a user query comes in, Claude looks at each skill's `description` and picks the most relevant (or none)
3. **Loading**: load that skill's `SKILL.md` body + invoke any helper scripts
4. **Progressive disclosure**: only load `reference/*` files when relevant — saves context

## Skills × MCP

Skills tell Claude **what to do**; MCP gives Claude **what to call**.

Example: `code_review` skill body says『run tests + check style』; the actual `run_tests` and `check_style` tools come from an MCP server.

This decoupling means:
- Same MCP tools serve many skills
- Same skill can run on different MCP backends

## Try it

```bash
# In assets/enterprise_5days/
python -c "
import sys; sys.path.insert(0, 'utils')
from skills_helpers import discover_skills, validate_skill
for s in discover_skills('skills_demo'):
    v = validate_skill(s.skill_dir)
    print(f'{s.name}: ok={v[\"ok\"]}  desc=\"{s.description[:60]}...\"')
"
```
