---
name: code-review
description: Performs a structured code review of Python changes — runs ruff for style, mypy for types, then summarizes findings using a checklist. Use when the user asks to review, audit, or check code quality.
allowed-tools: [bash, read_file]
version: "0.2"
---

# Code Review Skill

A pragmatic code reviewer for Python projects. Runs static checks and applies a
team checklist, then produces a prioritized report.

## When to use

- "Review this PR / file / function"
- "Check the code quality of X"
- "Audit this module for issues"

## Workflow

1. **Identify scope**: ask which files / diff to review
2. **Run static checks** (use `helper.py:run_checks()`):
   - `ruff check` for style / lint
   - `mypy --strict` for type safety
3. **Apply the team checklist** (see `reference/checklist.md`):
   - Naming clarity
   - Error handling completeness
   - Test coverage of new code
   - Docs updated
4. **Prioritize findings**:
   - **🔴 Blocking**: would break prod / security
   - **🟡 Should fix**: bad practice but works
   - **🟢 Nice to have**: style / micro-optimization
5. **Output**: structured markdown report (≤ 200 words)

## Output format

```markdown
## Code Review Summary

**Files reviewed**: ...
**Critical**: <count>  **Important**: <count>  **Nits**: <count>

### 🔴 Blocking
- ...

### 🟡 Should fix
- ...

### 🟢 Nice to have
- ...
```

## What this skill does NOT do

- Run the actual tests (delegate to a test-runner skill)
- Auto-fix issues (delegate to a refactor skill)
- Commit / push (delegate to a git skill)
