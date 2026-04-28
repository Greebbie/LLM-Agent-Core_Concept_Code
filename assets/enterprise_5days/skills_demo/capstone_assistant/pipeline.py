"""Enterprise Knowledge Assistant — packaged pipeline.

Extracted from Day 5 下午 升级 Capstone notebook so the full system can be
imported as `from pipeline import upgraded_pipeline`.

Reuses:
- utils.multi_agent (Planner / Worker / Reviewer)
- utils.mcp_helpers (EduMCPClient → enterprise-demo MCP server)
- utils.embedding_backend (SimpleVectorStore + dashscope embedder)
- utils.observability (Langfuse / MockObserver)
- utils.config (env loading)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

# Locate course root so utils/ is importable regardless of cwd
_root = Path(__file__).resolve().parent
for _ in range(5):
    if (_root / "utils").is_dir() and (_root / "data").is_dir():
        break
    _root = _root.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "utils"))
sys.path.insert(0, str(_root / "mcp_server_demo"))

from config import setup
from multi_agent import Message, MessageType, BaseAgent
from mcp_helpers import EduMCPClient
from embedding_backend import SimpleVectorStore
from observability import observe


env = setup()
llm = env.get_llm()
embedder = env.get_embedder()


# ── Knowledge base (would be external in real prod) ──
KNOWLEDGE_DOCS = [
    {"id": "hr_01", "text": "公司年假政策：入职 5 年以下每年 5 天，5-10 年 10 天，10 年以上 15 天。", "category": "hr"},
    {"id": "hr_02", "text": "病假需出示三甲医院证明，全薪连续不超过 30 天。", "category": "hr"},
    {"id": "tech_01", "text": "API 限流：免费版 60 req/min，企业版 6000 req/min。", "category": "tech"},
    {"id": "tech_02", "text": "API 鉴权使用 Bearer Token；token 由 CONSOLE 生成，30 天过期。", "category": "tech"},
    {"id": "tech_03", "text": "出现 429 (Too Many Requests) 时建议指数退避重试。", "category": "tech"},
    {"id": "prod_01", "text": "StarLink 基础版 199 元/月，5 路并发；企业版 1999 元/月，100 路并发。", "category": "product"},
    {"id": "prod_03", "text": "SKU-A100 是 StarLink 入门套件，4999 元。", "category": "product"},
]

vector_store = SimpleVectorStore(embedding_backend=embedder)
vector_store.add_documents(
    [d["text"] for d in KNOWLEDGE_DOCS],
    metadata=[{"id": d["id"], "category": d["category"]} for d in KNOWLEDGE_DOCS],
)


# ── MCP setup ──
from server import build_server as build_mcp_server  # type: ignore
mcp_server = build_mcp_server()
mcp_client = EduMCPClient(user_id="capstone-skill")
mcp_client.connect(mcp_server)


# ── Agents ──
PLANNER_PROMPT = '''你是 **路由 Planner**。
判断用户问题应走哪条路径，输出 JSON：
- 知识/文档/政策类 → {"path": "rag"}
- 订单/库存/通知 → {"path": "mcp"}
- 闲聊/无法判断 → {"path": "direct"}
只输出 JSON。'''

RAG_PROMPT = '基于检索文档简洁答用户问题。文档不够明确说『信息不足』。'
REVIEWER_PROMPT = '审核回答质量：APPROVE 或 REJECT，附一句理由。'

planner = BaseAgent("Planner", llm, PLANNER_PROMPT, temperature=0.0)
rag_worker = BaseAgent("RAGWorker", llm, RAG_PROMPT, temperature=0.1)
reviewer = BaseAgent("Reviewer", llm, REVIEWER_PROMPT, temperature=0.0)


@observe("skill.route")
def route_query(query: str) -> str:
    raw = planner.receive(Message("user", "Planner", MessageType.TASK, query)).payload.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(raw).get("path", "direct")
    except Exception:
        return "direct"


@observe("skill.rag")
def rag_branch(query: str) -> str:
    results = vector_store.search(query, top_k=3)
    if not results:
        return "[Fallback] 知识库未覆盖"
    ctx = "\n".join(f"- {r['document']}" for r in results)
    return rag_worker.receive(
        Message("Planner", "RAGWorker", MessageType.TASK, f"问题: {query}\n\n检索:\n{ctx}")
    ).payload


@observe("skill.mcp")
def mcp_branch(query: str) -> str:
    tools = mcp_client.list_all_tools()
    desc = "; ".join(f"{t['name']}({list(t['parameters']['properties'].keys())})" for t in tools)
    raw = llm.generate(
        f"工具: {desc}\n问题: {query}\n输出 JSON: {{'tool': '...', 'arguments': {{...}}}}",
        temperature=0.0,
    ).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        plan = json.loads(raw)
        result = mcp_client.call(mcp_server.name, plan["tool"], **plan["arguments"])
        return f"调 {plan['tool']}({plan['arguments']}) → {result}"
    except Exception as e:
        return f"[MCP 失败] {e}"


@observe("skill.direct")
def direct_branch(query: str) -> str:
    return llm.generate(f"用一两句话简短回答: {query}", temperature=0.3).strip()


@observe("skill.review")
def review_answer(query: str, answer: str) -> str:
    return reviewer.receive(
        Message("user", "Reviewer", MessageType.REVIEW, f"Q: {query}\nA: {answer}")
    ).payload


@observe("skill.upgraded_pipeline")
def upgraded_pipeline(query: str) -> dict:
    """Main entry — package this whole thing as a Skill, others import this."""
    path = route_query(query)
    if path == "rag":
        answer = rag_branch(query)
    elif path == "mcp":
        answer = mcp_branch(query)
    else:
        answer = direct_branch(query)
    review = review_answer(query, answer)
    return {"path": path, "answer": answer, "review": review}


if __name__ == "__main__":
    # Smoke test
    for q in [
        "入职 7 年有几天年假？",
        "查订单 ORD-001",
        "你好",
    ]:
        print("=" * 60)
        print(f"Q: {q}")
        r = upgraded_pipeline(q)
        print(f"  path={r['path']}")
        print(f"  answer={r['answer'][:120]}")
