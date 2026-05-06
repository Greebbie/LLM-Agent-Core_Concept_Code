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

__all__ = [
    "TraceSpan",
    "MockObserver",
    "SpanHandle",
    "observer",
    "observe",
    "span",
    "get_backend",
]


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


def _start_langfuse_span(name: str, input: Any = None, metadata: dict | None = None) -> Any:
    """Start a Langfuse observation across SDK v2/v3 API differences."""
    if _LANGFUSE_CLIENT is None:
        raise RuntimeError("Langfuse client is not initialized")

    kwargs = {"name": name, "input": input}
    if metadata:
        kwargs["metadata"] = metadata

    if hasattr(_LANGFUSE_CLIENT, "start_span"):
        return _LANGFUSE_CLIENT.start_span(**kwargs)
    if hasattr(_LANGFUSE_CLIENT, "trace"):
        return _LANGFUSE_CLIENT.trace(**kwargs)
    raise RuntimeError("Unsupported Langfuse SDK: expected start_span() or trace()")


def _update_langfuse_span(handle: Any, **kwargs: Any) -> None:
    clean = {k: v for k, v in kwargs.items() if v is not None}
    if clean and hasattr(handle, "update"):
        handle.update(**clean)


def _end_langfuse_span(handle: Any) -> None:
    if hasattr(handle, "end"):
        handle.end()


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
            ms = f"{span.duration_ms:.0f}ms" if span.duration_ms is not None else "running"
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


# 单值序列化上限：超过这个长度直接用 type 名字代替，防止 numpy 数组 / DataFrame 等
# 触发 str() 慢爆 + 估算的 token 数完全失真
_MAX_REPR_CHARS = 2000


def _safe_repr(obj: Any, limit: int = _MAX_REPR_CHARS) -> str:
    """对单个对象做安全字符串化：小对象 str()，大对象用 type + 形状提示替代。

    异常处理策略：
    - len() 抛 TypeError（对象没 __len__）→ 跳过 size 快路径
    - str() 抛 TypeError / ValueError（坏 __str__ 或 numpy 真值歧义等）→ 返回
      稳定占位，避免 tracing 把用户 pipeline 一并打断
    """
    type_name = type(obj).__name__
    # 字符串和数字直接处理
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        s = str(obj)
        return s if len(s) <= limit else f"<{type_name} len={len(s)} ...{s[-30:]}>"
    # 序列/字典先看 len，再决定是否完整 str()
    try:
        n = len(obj)
        if n > 200:
            return f"<{type_name} len={n}>"
    except TypeError:
        pass
    # 对其他对象，尝试 str() 但带超时保护（用 truncation 而不是真 timeout）
    try:
        s = str(obj)
    except (TypeError, ValueError):
        return f"<{type_name} __str__ raised>"
    return s if len(s) <= limit else f"<{type_name} repr_len={len(s)}>"


def _estimate_tokens(*objs: Any) -> int:
    """安全估算 token 数：对每个对象用 _safe_repr 后求总长 / 3。"""
    parts = [_safe_repr(o) for o in objs]
    return sum(len(p) for p in parts) // 3


def observe(name: Optional[str] = None) -> Callable:
    """Decorator to auto-trace a function call.

    Auto-estimates token usage from string lengths of args/result. **大对象保护**:
    单值超过 _MAX_REPR_CHARS（2000）直接用 type 名字替代，避免 numpy/DataFrame
    等触发 str() 慢爆 + token 数失真。

    估算精度: ASCII ~ /4, 中文 ~ /2, 我们用 /3 折中。生产请用 tiktoken 精确计数。
    """
    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            input_tokens = _estimate_tokens(*args, *kwargs.values())
            input_repr = {"args": _safe_repr(args, 500), "kwargs": _safe_repr(kwargs, 500)}
            if _BACKEND == "langfuse" and _LANGFUSE_CLIENT is not None:
                trace = _start_langfuse_span(span_name, input=input_repr)
                try:
                    out = fn(*args, **kwargs)
                    _update_langfuse_span(trace, output=_safe_repr(out, 500))
                    return out
                except Exception as e:
                    _update_langfuse_span(trace, level="ERROR", status_message=str(e)[:200])
                    raise
                finally:
                    _end_langfuse_span(trace)
            else:
                span = observer.start_span(span_name, input=input_repr)
                try:
                    result = fn(*args, **kwargs)
                    output_tokens = _estimate_tokens(result)
                    observer.record_tokens(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        cost_usd=(input_tokens + output_tokens) * 1e-6,  # mock pricing
                    )
                    observer.end_span(span, output=_safe_repr(result, 300), tokens=input_tokens + output_tokens)
                    return result
                except Exception as e:
                    observer.end_span(span, output=f"[ERROR] {e}", error=str(e))
                    raise
        return wrapper
    return decorator


class SpanHandle:
    """统一 span 抽象 — 让两种后端 (Langfuse + MockObserver) 暴露同样的 API.

    用户写：
        with span("step") as s:
            result = do_work()
            s.update(output=result)

    无论后端是 Langfuse 还是 Mock，update() 行为一致。
    """
    def __init__(self, backend: str, handle: Any):
        self._backend = backend
        self._handle = handle
        self._closed = False

    def update(self, output: Any = None, **metadata):
        """记录 output / metadata 到 span。两种后端通用。"""
        if self._backend == "langfuse":
            _update_langfuse_span(
                self._handle,
                output=_safe_repr(output, 500) if output is not None else None,
                **metadata,
            )
        else:
            # Mock: 直接改 span 字段
            if output is not None:
                self._handle.output = _safe_repr(output, 500)
            self._handle.metadata.update(metadata)

    def end(self, output: Any = None, **metadata) -> None:
        """Finish the current span, optionally recording output and metadata first."""
        if output is not None or metadata:
            self.update(output=output, **metadata)
        if self._closed:
            return
        if self._backend == "langfuse":
            _end_langfuse_span(self._handle)
        else:
            observer.end_span(self._handle)
        self._closed = True


@contextmanager
def span(name: str, input: Any = None, **metadata):
    """Context manager for manual span control.

    用法:
        with span("my_step", input={"q": query}) as s:
            result = do_work()
            s.update(output=result, status="ok")
    """
    if _BACKEND == "langfuse" and _LANGFUSE_CLIENT is not None:
        safe_input = _safe_repr(input, 500) if input is not None else None
        trace = _start_langfuse_span(name, input=safe_input, metadata=metadata)
        handle = SpanHandle("langfuse", trace)
        try:
            yield handle
        except Exception as e:
            handle.update(level="ERROR", status_message=str(e)[:200])
            raise
        finally:
            handle.end()
    else:
        s = observer.start_span(name, input=input, metadata=metadata)
        handle = SpanHandle("mock", s)
        try:
            yield handle
        finally:
            handle.end()
