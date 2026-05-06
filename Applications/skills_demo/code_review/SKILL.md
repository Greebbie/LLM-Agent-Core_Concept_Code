---
name: code-review
description: Review Python code or a pull request for correctness, maintainability, tests, and security-sensitive mistakes. Use when the user asks to review, audit, or check code quality.
allowed-tools: [bash, read_file]
version: "0.2"
---

# Code Review Skill

A pragmatic code reviewer for Python projects. It combines static checks with a team checklist, then returns prioritized findings instead of generic advice.

## When To Use

- "Review this PR / file / function"
- "Check the code quality of this module"
- "Audit this Python change for risks"

## Workflow

1. Identify the review scope: files, diff, or module.
2. Run static checks with `helper.py:run_checks()` when the environment supports it:
   - `ruff check` for lint and style issues.
   - `mypy --no-error-summary` for type issues.
3. Load `reference/checklist.md` only when a deeper checklist is needed.
4. Prioritize findings:
   - `[P1] Blocking`: likely bug, security issue, or production break.
   - `[P2] Should fix`: maintainability, missing validation, weak tests.
   - `[P3] Nice to have`: naming, polish, small refactor.
5. Output a concise markdown report with file references and concrete fixes.

## Output Format

```markdown
## Code Review Summary

**Files reviewed**: ...
**Critical**: <count> **Important**: <count> **Nits**: <count>

### [P1] Blocking
- ...

### [P2] Should fix
- ...

### [P3] Nice to have
- ...
```

## Boundaries

- Do not commit or push changes.
- Do not auto-fix unless the user explicitly asks.
- Do not replace a real test run with speculation. If checks cannot run, say what was not verified.
