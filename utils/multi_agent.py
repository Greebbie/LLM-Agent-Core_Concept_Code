"""Multi-Agent helpers shared across Day 4 上午 and Day 5 下午 (Capstone).

Provides:
- Message dataclass — agent-to-agent messages
- BaseAgent — minimal LLM-driven agent
- Orchestrator — Hierarchical / Debate / Handoff coordination
- CircuitBreaker — fault tolerance for unreliable agents

Designed to be small, transparent, and customizable. Not a replacement for
LangGraph or CrewAI — a teaching scaffold so learners see the moving parts.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

__all__ = [
    "MessageType",
    "Message",
    "BaseAgent",
    "Orchestrator",
    "CircuitBreaker",
]


class MessageType(str, Enum):
    TASK = "task"
    RESULT = "result"
    REVIEW = "review"
    HANDOFF = "handoff"
    ERROR = "error"


@dataclass
class Message:
    """Inter-agent message. `from_` and `to` are agent names; `payload` is free-form text."""
    from_: str
    to: str
    type: MessageType
    payload: str
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return f"Message({self.from_} → {self.to} | {self.type.value}: {self.payload[:50]!r})"


class BaseAgent:
    """Minimal LLM-backed agent. Subclass and override `system_prompt` for specialization.

    `max_history` 限制每个 Agent 内部 history list 的长度（防长会话内存泄漏）。
    超过时丢最旧的，保留最近 `max_history` 条 (FIFO)。
    """

    def __init__(self, name: str, llm, system_prompt: str = "", temperature: float = 0.3,
                 max_history: int = 50):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt or f"You are {name}."
        self.temperature = temperature
        self.max_history = max_history
        self.history: list[Message] = []

    def _truncate_history(self) -> None:
        """超过 max_history 时丢最旧的（防内存泄漏）。"""
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def receive(self, msg: Message) -> Message:
        """Process incoming message, return response message."""
        self.history.append(msg)
        self._truncate_history()
        prompt = f"{self.system_prompt}\n\n收到来自 {msg.from_} 的{msg.type.value}：\n{msg.payload}\n\n请回复："
        response = self.llm.generate(prompt, temperature=self.temperature)
        out = Message(
            from_=self.name, to=msg.from_, type=MessageType.RESULT, payload=response.strip(),
        )
        self.history.append(out)
        self._truncate_history()
        return out


class Orchestrator:
    """Coordinates multiple agents in 3 modes: hierarchical / debate / handoff."""

    def __init__(self, agents: dict[str, BaseAgent], trace: Optional[Callable[[Message], None]] = None):
        self.agents = agents
        self.trace = trace or (lambda msg: None)
        self.transcript: list[Message] = []

    def _send(self, msg: Message) -> Message:
        self.transcript.append(msg)
        self.trace(msg)
        if msg.to not in self.agents:
            return Message(from_="orchestrator", to=msg.from_, type=MessageType.ERROR,
                           payload=f"Unknown agent: {msg.to}")
        response = self.agents[msg.to].receive(msg)
        self.transcript.append(response)
        self.trace(response)
        return response

    def run_hierarchical(self, task: str, planner: str, workers: list[str], reviewer: str,
                          max_revisions: int = 2) -> dict:
        """Planner → Worker(s) in parallel → Reviewer. Reviewer can request revision."""
        plan_msg = Message(from_="user", to=planner, type=MessageType.TASK, payload=task)
        plan_out = self._send(plan_msg)
        worker_results = []
        for w in workers:
            w_msg = Message(from_=planner, to=w, type=MessageType.TASK, payload=plan_out.payload)
            worker_results.append(self._send(w_msg).payload)
        combined = "\n\n---\n\n".join(f"[{w}] {r}" for w, r in zip(workers, worker_results))
        for revision in range(max_revisions + 1):
            review_msg = Message(from_=planner, to=reviewer, type=MessageType.REVIEW,
                                  payload=f"任务：{task}\n\n各 worker 输出：\n{combined}")
            review_out = self._send(review_msg)
            if "approve" in review_out.payload.lower() or "通过" in review_out.payload:
                return {"final": combined, "review": review_out.payload, "revisions": revision}
            # Reviewer rejected — re-plan
            if revision < max_revisions:
                plan_msg = Message(from_=reviewer, to=planner, type=MessageType.TASK,
                                    payload=f"原任务：{task}\n\nReviewer 反馈：{review_out.payload}\n请改进计划。")
                plan_out = self._send(plan_msg)
                worker_results = []
                for w in workers:
                    w_msg = Message(from_=planner, to=w, type=MessageType.TASK, payload=plan_out.payload)
                    worker_results.append(self._send(w_msg).payload)
                combined = "\n\n---\n\n".join(f"[{w}] {r}" for w, r in zip(workers, worker_results))
        return {"final": combined, "review": review_out.payload, "revisions": max_revisions, "status": "max_revisions_reached"}

    def run_debate(self, question: str, debaters: list[str], judge: str, rounds: int = 1) -> dict:
        """Each debater answers; judge picks the best (or aggregates)."""
        answers = {}
        for d in debaters:
            msg = Message(from_="user", to=d, type=MessageType.TASK, payload=question)
            answers[d] = self._send(msg).payload
        # Judge phase
        ans_text = "\n\n".join(f"[{d}] {a}" for d, a in answers.items())
        judge_msg = Message(from_="user", to=judge, type=MessageType.REVIEW,
                             payload=f"问题：{question}\n\n各方答案：\n{ans_text}\n\n请选出最佳答案并说明理由。")
        judgment = self._send(judge_msg).payload
        return {"answers": answers, "judgment": judgment}

    def run_handoff(self, task: str, start_agent: str, max_hops: int = 5) -> dict:
        """Agent decides whether to handle or hand off.

        Agent's response should include `HANDOFF: <agent_name>` (case-insensitive)
        — robust to surrounding markdown, code fences, punctuation.
        """
        import re as _re
        # 严格 regex：HANDOFF: 后跟 word 字符（字母/数字/下划线），忽略大小写
        # 这样能正确处理 "HANDOFF: Agent1" / "```\nHANDOFF: Agent1\n```" / "HANDOFF: Agent1." 等
        _handoff_re = _re.compile(r"HANDOFF\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", _re.IGNORECASE)

        current = start_agent
        payload = task
        path = [current]
        for hop in range(max_hops):
            msg = Message(from_="user" if hop == 0 else path[-2], to=current,
                           type=MessageType.HANDOFF if hop > 0 else MessageType.TASK, payload=payload)
            response = self._send(msg)
            text = response.payload
            m = _handoff_re.search(text)
            if m:
                next_agent = m.group(1)
                if next_agent in self.agents and next_agent != current:
                    current = next_agent
                    payload = text
                    path.append(current)
                    continue
            return {"final": text, "path": path, "hops": hop + 1}
        return {"final": payload, "path": path, "hops": max_hops, "status": "max_hops_reached"}


class CircuitBreaker:
    """Simple circuit breaker: open after N consecutive failures, half-open after timeout."""

    def __init__(self, failure_threshold: int = 3, reset_timeout_s: float = 30.0):
        self.threshold = failure_threshold
        self.reset_timeout = reset_timeout_s
        self.failures = 0
        self.opened_at: Optional[float] = None

    def call(self, fn: Callable, *args, **kwargs):
        # Half-open check
        if self.opened_at is not None:
            if time.time() - self.opened_at < self.reset_timeout:
                raise RuntimeError(f"Circuit breaker OPEN; retry after {self.reset_timeout}s")
            self.opened_at = None
            self.failures = 0
        try:
            result = fn(*args, **kwargs)
            self.failures = 0
            return result
        except Exception:
            self.failures += 1
            if self.failures >= self.threshold:
                self.opened_at = time.time()
            raise

    @property
    def state(self) -> str:
        if self.opened_at is not None:
            return "open"
        if self.failures > 0:
            return f"half-open ({self.failures}/{self.threshold} failures)"
        return "closed"
