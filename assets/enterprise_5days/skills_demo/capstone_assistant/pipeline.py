"""Enterprise Knowledge Assistant packaged pipeline.

This is the Day 5 production capstone extracted from the notebook.  The key
lesson is not "make the prompt longer"; it is showing a real before/after:

- baseline_rag_pipeline: always uses retrieval
- llm_planner_pipeline: lets the LLM decide the route
- production_pipeline: keeps the LLM, but adds deterministic routing guards for
  high-signal business identifiers such as ORD-* and SKU-*

`upgraded_pipeline` is kept as the public Skill entrypoint and points to the
production version.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# Locate course root so utils/ is importable regardless of cwd.
_root = Path(__file__).resolve().parent
for _ in range(5):
    if (_root / "utils").is_dir() and (_root / "data").is_dir():
        break
    _root = _root.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "utils"))
sys.path.insert(0, str(_root / "mcp_server_demo"))

from utils.config import setup
from utils.multi_agent import BaseAgent, Message, MessageType
from utils.mcp_helpers import EduMCPClient
from utils.embedding_backend import SimpleVectorStore
from utils.observability import observe


env = setup()
llm = env.get_llm()
embedder = env.get_embedder()


KNOWLEDGE_DOCS = [
    {"id": "hr_01", "text": "公司年假政策：入职 5 年以下每年 5 天，5-10 年 10 天，10 年以上 15 天。", "category": "hr"},
    {"id": "hr_02", "text": "病假需出示三甲医院证明，全薪连续不超过 30 天。", "category": "hr"},
    {"id": "tech_01", "text": "API 限流：免费版 60 req/min，企业版 6000 req/min。", "category": "tech"},
    {"id": "tech_02", "text": "API 鉴权使用 Bearer Token；token 由 CONSOLE 生成，30 天过期。", "category": "tech"},
    {"id": "tech_03", "text": "出现 429 (Too Many Requests) 时建议指数退避重试。", "category": "tech"},
    {"id": "prod_01", "text": "StarLink 基础版 199 元/月，5 路并发；企业版 1999 元/月，100 路并发。", "category": "product"},
    {"id": "prod_03", "text": "SKU-A100 是 StarLink 入门套件，包含 1 个网关 + 5 个传感器，售价 4999 元。", "category": "product"},
]

vector_store = SimpleVectorStore(embedding_backend=embedder)
vector_store.add_documents(
    [d["text"] for d in KNOWLEDGE_DOCS],
    metadata=[{"id": d["id"], "category": d["category"]} for d in KNOWLEDGE_DOCS],
)


from server import build_server as build_mcp_server  # type: ignore  # noqa: E402

mcp_server = build_mcp_server()
mcp_client = EduMCPClient(user_id="capstone-skill")
mcp_client.connect(mcp_server)


PLANNER_PROMPT = """你是路由 Planner。判断用户问题应走哪条路径，输出 JSON：
- 知识/文档/政策/产品/API 类 -> {"path": "rag"}
- 订单/库存/通知 -> {"path": "mcp"}
- 闲聊/无法判断 -> {"path": "direct"}
只输出 JSON。"""

RAG_PROMPT = """你是 RAG Worker。只能基于检索到的文档回答。
如果文档不覆盖问题，明确说「知识库未覆盖」，不要编造。"""

REVIEWER_PROMPT = """你是回答审查者。
判断回答是否正面回答、是否基于已给信息、长度是否适中。
APPROVE 或 REJECT，附一句理由。"""

MCP_WORKER_PROMPT = """可用 MCP tools: {tools}
用户问题: {query}
请输出 JSON: {{"tool": "...", "arguments": {{...}}}}。只输出 JSON。"""

planner = BaseAgent("Planner", llm, PLANNER_PROMPT, temperature=0.0)
rag_worker = BaseAgent("RAGWorker", llm, RAG_PROMPT, temperature=0.1)
reviewer = BaseAgent("Reviewer", llm, REVIEWER_PROMPT, temperature=0.0)


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM output: {raw[:120]}")
    return json.loads(match.group(0))


SUPPORTED_RAG_TERMS = [
    "年假", "病假", "限流", "鉴权", "Bearer", "token", "429",
    "StarLink", "基础版", "企业版", "SKU-A100", "API",
]
UNSUPPORTED_COMPANY_TERMS = ["医疗保险", "加班补贴", "团建", "报销", "福利", "公司有"]


def _is_supported_rag_query(query: str) -> bool:
    return any(term.lower() in query.lower() for term in SUPPORTED_RAG_TERMS)


def _rule_route(query: str) -> str | None:
    q_upper = query.upper()
    if re.search(r"\bORD-[A-Z0-9]+\b", q_upper) or "订单" in query:
        return "mcp"
    if re.search(r"\bSKU-[A-Z0-9]+\b", q_upper) and any(term in query for term in ["库存", "还有多少", "stock", "inventory"]):
        return "mcp"
    if any(term in query for term in ["通知", "提醒", "告知"]):
        return "mcp"
    if _is_supported_rag_query(query):
        return "rag"
    if any(term in query for term in UNSUPPORTED_COMPANY_TERMS):
        return "direct"
    return None


@observe("skill.route_llm")
def route_query_llm(query: str) -> str:
    raw = planner.receive(Message("user", "Planner", MessageType.TASK, query)).payload
    try:
        return _extract_json_object(raw).get("path", "direct")
    except Exception:
        return "direct"


@observe("skill.route")
def route_query(query: str) -> str:
    """Production route: deterministic guards first, LLM planner second."""
    return _rule_route(query) or route_query_llm(query)


@observe("skill.rag")
def rag_branch(query: str) -> str:
    results = vector_store.search(query, top_k=3)
    if not results:
        return "知识库未覆盖这个问题，建议补充文档后再回答。"
    ctx = "\n".join(f"- {r['document']}" for r in results)
    answer = rag_worker.receive(
        Message("Planner", "RAGWorker", MessageType.TASK, f"问题: {query}\n\n检索到:\n{ctx}")
    ).payload.strip()
    sources = "\n".join(
        f"[{r['metadata'].get('id')}] {r['document']}" for r in results[:2]
    )
    return f"{answer}\n\n依据文档:\n{sources}"


def _fallback_mcp_plan(query: str) -> dict[str, Any] | None:
    q_upper = query.upper()
    order = re.search(r"\bORD-[A-Z0-9]+\b", q_upper)
    if order:
        return {"tool": "query_order", "arguments": {"order_id": order.group(0)}}
    sku = re.search(r"\bSKU-[A-Z0-9]+\b", q_upper)
    if sku:
        return {"tool": "check_inventory", "arguments": {"sku": sku.group(0)}}
    notify = re.search(r"(?:通知|提醒|告知)\s*([A-Za-z][\w-]*)", query)
    if notify:
        return {
            "tool": "send_notification",
            "arguments": {"user_id": notify.group(1), "message": query},
        }
    return None


def _normalize_mcp_plan(plan: dict[str, Any], query: str) -> dict[str, Any]:
    """Keep LLM tool choice, but trust exact IDs from the user query."""
    normalized = {
        "tool": plan.get("tool"),
        "arguments": dict(plan.get("arguments") or {}),
    }
    q_upper = query.upper()
    order = re.search(r"\bORD-[A-Z0-9]+\b", q_upper)
    if normalized["tool"] == "query_order" and order:
        normalized["arguments"]["order_id"] = order.group(0)
    sku = re.search(r"\bSKU-[A-Z0-9]+\b", q_upper)
    if normalized["tool"] == "check_inventory" and sku:
        normalized["arguments"]["sku"] = sku.group(0)
    return normalized


@observe("skill.mcp")
def mcp_branch(query: str) -> str:
    tools = mcp_client.list_all_tools()
    desc = "; ".join(f"{t['name']}({list(t['parameters']['properties'].keys())})" for t in tools)
    raw = llm.generate(MCP_WORKER_PROMPT.format(query=query, tools=desc), temperature=0.0)
    planner_source = "llm"
    try:
        plan = _extract_json_object(raw)
    except Exception:
        plan = _fallback_mcp_plan(query)
        planner_source = "rule-fallback"
    if plan is None:
        return f"[MCP 调用失败] 无法从问题中提取 tool 参数；LLM 输出: {raw[:120]}"
    plan = _normalize_mcp_plan(plan, query)
    try:
        result = mcp_client.call(mcp_server.name, plan["tool"], **plan["arguments"])
        return f"调用 MCP tool {plan['tool']}({plan['arguments']}) [{planner_source}] 得到: {result}"
    except Exception as e:
        return f"[MCP 调用失败] {e}"


@observe("skill.direct")
def direct_branch(query: str) -> str:
    if any(term in query for term in UNSUPPORTED_COMPANY_TERMS):
        return "当前知识库未覆盖这个问题，不能确认是否有相关政策；建议联系 HR 或系统管理员补充文档后再答复。"
    return llm.generate(f"用一两句话简短回答: {query}", temperature=0.3).strip()


@observe("skill.review")
def review_answer(query: str, answer: str) -> str:
    return reviewer.receive(
        Message("user", "Reviewer", MessageType.REVIEW, f"Q: {query}\nA: {answer}")
    ).payload


@observe("skill.baseline_rag_pipeline")
def baseline_rag_pipeline(query: str) -> dict[str, str]:
    answer = rag_branch(query)
    return {"path": "rag-only", "answer": answer, "review": ""}


@observe("skill.llm_planner_pipeline")
def llm_planner_pipeline(query: str) -> dict[str, str]:
    path = route_query_llm(query)
    if path == "rag":
        answer = rag_branch(query)
    elif path == "mcp":
        answer = mcp_branch(query)
    else:
        answer = direct_branch(query)
    review = review_answer(query, answer)
    return {"path": path, "answer": answer, "review": review}


@observe("skill.naive_keyword_pipeline")
def naive_keyword_pipeline(query: str) -> dict[str, str]:
    """Before version: broad product/SKU keywords are routed to MCP too eagerly."""
    q_upper = query.upper()
    if "ORD-" in q_upper or "SKU-" in q_upper or "STARLINK" in q_upper or "订单" in query or "库存" in query:
        path = "mcp"
    elif _is_supported_rag_query(query):
        path = "rag"
    else:
        path = "direct"
    if path == "rag":
        answer = rag_branch(query)
    elif path == "mcp":
        answer = mcp_branch(query)
    else:
        answer = direct_branch(query)
    review = review_answer(query, answer)
    return {"path": path, "answer": answer, "review": review}


@observe("skill.production_pipeline")
def production_pipeline(query: str) -> dict[str, str]:
    path = route_query(query)
    if path == "rag":
        answer = rag_branch(query)
    elif path == "mcp":
        answer = mcp_branch(query)
    else:
        answer = direct_branch(query)
    review = review_answer(query, answer)
    return {"path": path, "answer": answer, "review": review}


@observe("skill.upgraded_pipeline")
def upgraded_pipeline(query: str) -> dict[str, str]:
    """Public Skill entrypoint: production version of the capstone."""
    return production_pipeline(query)


if __name__ == "__main__":
    for q in ["入职 7 年有几天年假？", "查订单 ORD-001", "SKU-A100 是什么？多少钱？", "公司有医疗保险吗？"]:
        print("=" * 60)
        print(f"Q: {q}")
        r = upgraded_pipeline(q)
        print(f"  path={r['path']}")
        print(f"  answer={r['answer'][:160]}")
