# Team Code Review Checklist

Load this checklist only when a detailed review is needed.

## 1. Naming

- Do function and variable names explain intent without reading the implementation?
- Are abbreviations limited to team-standard terms such as `db`, `cfg`, or `ctx`?
- Do boolean names use `is_`, `has_`, `can_`, or `should_` where helpful?

## 2. Error Handling

- Are IO, network, subprocess, and parsing boundaries handled explicitly?
- Are exceptions logged or returned with enough context to debug?
- Is there any empty `except: pass` that hides real failures?
- Are resources closed through context managers?

## 3. Tests

- Is there at least one meaningful happy-path test?
- Are boundary cases covered: empty input, large input, malformed input, unavailable dependency?
- Do assertions prove behavior rather than just executing code?
- Are mocks close enough to real behavior to catch regressions?

## 4. Documentation

- Do public functions have docstrings when the behavior is not obvious?
- Are complex blocks commented for why the approach exists, not just what each line does?
- Did README or usage docs change when the behavior changed?

## 5. Security

- Is untrusted input validated at the boundary?
- Is there any SQL or shell command built by string concatenation?
- Are secrets loaded from environment variables rather than committed files?
- Are dependency and file-system assumptions clear?

## 6. Performance

- Is repeated IO avoided inside hot loops?
- Are large datasets streamed or batched where appropriate?
- Are expensive model calls cached or gated when repeated?
