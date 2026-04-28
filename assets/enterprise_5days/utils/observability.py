"""LLMOps observability — minimal Langfuse wrapper with mock fallback.

Design:
- If Langfuse is installed AND LANGFUSE_PUBLIC_KEY is in env → real client
- Otherwise → in-memory MockObserver that records traces locally

Both expose the same API:
    @observe()
    def my_func(...): ...

    with span("agent_call", input=...) as s:
        ...
        s.end(output=...)

    print(observer.summary())   # show all recorded traces
"""
from __future__ import annotations
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional


# ── Detect real Langfuse availability ──────────────────────
_LANGFUSE_CLIENT = None
_BACKEND = "mock"

try:
    from langfuse import Langfuse  # type: ignore
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        _LANGFUSE_CLIENT = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        _BACKEND = "langfuse"
except ImportError:
    pass


# ============================================================
# MockObserver — pure-Python in-memory trace store
# ============================================================
@dataclass
class TraceSpan:
    name: str
    start_time: float
    end_time: Optional[float] = None
    input: Any = None
    output: Any = None
    metadata: dict = field(default_factory=dict)
    children: list["TraceSpan"] = field(default_factory=list)
    parent_id: Optional[int] = None
    span_id: int = 0

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000


class MockObserver:
    """In-memory trace recorder. API mirrors Langfuse for teaching."""

    def __init__(self):
        self.spans: list[TraceSpan] = []
        self._stack: list[TraceSpan] = []
        self._counter = 0
        self.token_usage = {"prompt": 0, "completion": 0, "total": 0}
        self.cost_usd = 0.0

    def _new_span_id(self) -> int:
        self._counter += 1
        return self._counter

    def start_span(self, name: str, input: Any = None, metadata: dict | None = None) -> TraceSpan:
        span = TraceSpan(
            name=name,
            start_time=time.time(),
            input=input,
            metadata=metadata or {},
            parent_id=self._stack[-1].span_id if self._stack else None,
            span_id=self._new_span_id(),
        )
        if self._stack:
            self._stack[-1].children.append(span)
        else:
            self.spans.append(span)
        self._stack.append(span)
        return span

    def end_span(self, span: TraceSpan, output: Any = None, **metadata) -> None:
        span.end_time = time.time()
        if output is not None:
            span.output = output
        span.metadata.update(metadata)
        if self._stack and self._stack[-1] is span:
            self._stack.pop()

    def record_tokens(self, prompt_tokens: int = 0, completion_tokens: int = 0, cost_usd: float = 0.0) -> None:
        self.token_usage["prompt"] += prompt_tokens
        self.token_usage["completion"] += completion_tokens
        self.token_usage["total"] += prompt_tokens + completion_tokens
        self.cost_usd += cost_usd

    def reset(self) -> None:
        self.spans.clear()
        self._stack.clear()
        self._counter = 0
        self.token_usage = {"prompt": 0, "completion": 0, "total": 0}
        self.cost_usd = 0.0

    def summary(self) -> dict:
        def flatten(span: TraceSpan, depth: int = 0) -> list:
            items = [{"depth": depth, "name": span.name, "duration_ms": span.duration_ms}]
            for child in span.children:
                items.extend(flatten(child, depth + 1))
            return items
        all_spans = []
        for root in self.spans:
            all_spans.extend(flatten(root))
        total_duration = sum(s["duration_ms"] or 0 for s in all_spans if s["depth"] == 0)
        return {
            "n_traces": len(self.spans),
            "n_total_spans": len(all_spans),
            "total_duration_ms": round(total_duration, 1),
            "tokens": dict(self.token_usage),
            "cost_usd": round(self.cost_usd, 4),
        }

    def print_tree(self) -> None:
        def render(span: TraceSpan, depth: int = 0) -> None:
            ms = f"{span.duration_ms:.0f}ms" if span.duration_ms else "running"
            print("  " * depth + f"├─ {span.name} ({ms})")
            for child in span.children:
                render(child, depth + 1)
        for root in self.spans:
            render(root)


observer = MockObserver()


# ============================================================
# Public API: @observe / span / get_backend
# ============================================================
def get_backend() -> str:
    return _BACKEND


def observe(name: Optional[str] = None) -> Callable:
    """Decorator to auto-trace a function call.

    Auto-estimates token usage from string lengths of args/result (rough but
    consistent — divide-by-4 is a common ASCII heuristic; Chinese closer to /2,
    so we use /3 as compromise). For exact counts integrate tiktoken in prod.
    """
    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            args_str = " ".join(str(a) for a in args) + " " + " ".join(f"{k}={v}" for k, v in kwargs.items())
            input_tokens = len(args_str) // 3  # rough estimate (Chinese-friendly)
            if _BACKEND == "langfuse" and _LANGFUSE_CLIENT is not None:
                trace = _LANGFUSE_CLIENT.trace(name=span_name, input={"args": str(args)[:500], "kwargs": str(kwargs)[:500]})
                try:
                    out = fn(*args, **kwargs)
                    output_tokens = len(str(out)) // 3
                    trace.update(output=str(out)[:500])
                    # Real Langfuse usage tracking would go here via trace.usage
                    return out
                except Exception as e:
                    trace.update(level="ERROR", status_message=str(e)[:200])
                    raise
            else:
                span = observer.start_span(span_name, input={"args": str(args)[:200], "kwargs": str(kwargs)[:200]})
                try:
                    result = fn(*args, **kwargs)
                    output_tokens = len(str(result)) // 3
                    # Auto-record tokens (Chinese ≈ 3 chars/token rough)
                    observer.record_tokens(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        cost_usd=(input_tokens + output_tokens) * 1e-6,  # mock pricing
                    )
                    observer.end_span(span, output=str(result)[:300], tokens=input_tokens + output_tokens)
                    return result
                except Exception as e:
                    observer.end_span(span, output=f"[ERROR] {e}", error=str(e))
                    raise
        return wrapper
    return decorator


@contextmanager
def span(name: str, input: Any = None, **metadata):
    """Context manager for manual span control."""
    if _BACKEND == "langfuse" and _LANGFUSE_CLIENT is not None:
        trace = _LANGFUSE_CLIENT.trace(name=name, input=input, metadata=metadata)
        try:
            yield trace
        except Exception as e:
            trace.update(level="ERROR", status_message=str(e)[:200])
            raise
    else:
        s = observer.start_span(name, input=input, metadata=metadata)
        try:
            yield s
        finally:
            observer.end_span(s)
