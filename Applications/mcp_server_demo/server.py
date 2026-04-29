"""Demo MCP-style server — a real subprocess-runnable example.

This server speaks **JSON-RPC over stdio** — the same transport pattern
official MCP servers use. We don't rely on the `mcp` Python SDK (it needs
Python 3.10+) so we can run on the course's 3.9 conda env. The wire protocol
is a faithful subset (initialize / tools/list / tools/call), enough for
learners to see how a real MCP client/server pair looks.

Three modes:
    python server.py --stdio     # 真起 server，stdio JSON-RPC loop（生产模式）
    python server.py --demo      # 进程内打印 tool 列表（无 transport，用于教学）
    (no args)                    # 自动：装了 mcp 走 FastMCP，否则 stdio

Exposes 3 enterprise-style tools:
- query_order(order_id)        — look up order status
- check_inventory(sku)         — look up stock
- send_notification(user_id, message)  — mock send
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

# Walk up from this file to find the course/repo root (containing utils/ and data/)
# so we work both in Applications/ (repo root 2 levels up) and in enterprise_5days/
# (root 1 level up). Same pattern as skills_demo/capstone_assistant/pipeline.py.
_root = Path(__file__).resolve().parent
for _ in range(5):
    if (_root / "utils").is_dir() and (_root / "data").is_dir():
        break
    _root = _root.parent
sys.path.insert(0, str(_root))
from utils.mcp_helpers import EduMCPServer, ToolDef, MCP_AVAILABLE


# ── Mock backend data ────────────────────────────────────
ORDERS = {
    "ORD-001": {"status": "shipped", "total": 199.0, "customer": "alice"},
    "ORD-002": {"status": "pending", "total": 89.0, "customer": "bob"},
    "ORD-003": {"status": "delivered", "total": 450.0, "customer": "carol"},
}
INVENTORY = {
    "SKU-A100": 35,
    "SKU-B200": 0,
    "SKU-C300": 128,
}
NOTIFICATION_LOG: list[dict] = []


# ── Tool implementations ─────────────────────────────────
def query_order(order_id: str) -> str:
    """Look up order by ID. Returns JSON string."""
    if order_id not in ORDERS:
        return json.dumps({"error": f"Order {order_id} not found"})
    return json.dumps(ORDERS[order_id], ensure_ascii=False)


def check_inventory(sku: str) -> str:
    """Look up stock for a SKU. Returns JSON string."""
    qty = INVENTORY.get(sku)
    if qty is None:
        return json.dumps({"error": f"SKU {sku} not in inventory"})
    return json.dumps({"sku": sku, "quantity": qty, "in_stock": qty > 0})


def send_notification(user_id: str, message: str) -> str:
    """Mock-send a notification. Returns confirmation."""
    NOTIFICATION_LOG.append({"user_id": user_id, "message": message})
    return json.dumps({"status": "sent", "queue_size": len(NOTIFICATION_LOG)})


# ── Build server (same API for real MCP and Edu mock) ────
def build_server():
    server = EduMCPServer(name="enterprise-demo")
    server.add_tool(ToolDef(
        name="query_order",
        description="Look up an order by ID",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
        func=query_order,
    ))
    server.add_tool(ToolDef(
        name="check_inventory",
        description="Check stock quantity for a SKU",
        parameters={
            "type": "object",
            "properties": {"sku": {"type": "string"}},
            "required": ["sku"],
        },
        func=check_inventory,
    ))
    server.add_tool(ToolDef(
        name="send_notification",
        description="Send a notification to a user",
        parameters={
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["user_id", "message"],
        },
        func=send_notification,
    ))
    return server


def run_real_mcp():
    """Run as real MCP server over stdio (production mode)."""
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("enterprise-demo")

    @mcp.tool()
    def query_order_real(order_id: str) -> str:
        """Look up an order by ID"""
        return query_order(order_id)

    @mcp.tool()
    def check_inventory_real(sku: str) -> str:
        """Check stock quantity for a SKU"""
        return check_inventory(sku)

    @mcp.tool()
    def send_notification_real(user_id: str, message: str) -> str:
        """Send a notification to a user"""
        return send_notification(user_id, message)

    mcp.run(transport="stdio")


def run_demo_mode():
    """Run as in-process demo (no transport, just print tool list)."""
    server = build_server()
    print("=" * 60)
    print(f"MCP Demo Server: {server.name}")
    print("=" * 60)
    print(f"\n{len(server.list_tools())} tools available:")
    for t in server.list_tools():
        print(f"  • {t['name']}: {t['description']}")
        print(f"      params: {list(t['parameters']['properties'].keys())}")


# ── Stdio JSON-RPC loop — 真正的 MCP 协议精髓（不依赖 mcp 包） ──
def run_stdio_jsonrpc():
    """Real subprocess server: read JSON-RPC requests from stdin, write responses to stdout.

    Implements a subset of MCP protocol:
    - initialize       → {protocol_version, server_info}
    - tools/list       → {tools: [...]}
    - tools/call       → {content: [{type: "text", text: ...}]}
    - shutdown         → exit cleanly

    This is what every real MCP server does at the wire level.
    Official mcp SDK just wraps this with nicer ergonomics.
    """
    server = build_server()
    # Log to stderr so it doesn't pollute stdout (which carries protocol)
    print(f"[server] {server.name} listening on stdio (JSON-RPC)", file=sys.stderr, flush=True)
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as e:
                resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {e}"}}
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
                continue

            method = req.get("method")
            params = req.get("params", {})
            req_id = req.get("id")
            print(f"[server] ← {method}({params})", file=sys.stderr, flush=True)

            try:
                if method == "initialize":
                    result = {
                        "protocol_version": "2025-11-05-edu",
                        "server_info": {"name": server.name, "version": "0.1.0"},
                        "capabilities": {"tools": {}, "resources": {}},
                    }
                elif method == "tools/list":
                    result = {"tools": server.list_tools()}
                elif method == "tools/call":
                    tool_name = params.get("name")
                    arguments = params.get("arguments", {})
                    out = server.call_tool(tool_name, arguments)
                    result = {"content": [{"type": "text", "text": str(out)}]}
                elif method == "resources/list":
                    result = {"resources": server.list_resources()}
                elif method == "resources/read":
                    uri = params.get("uri")
                    out = server.read_resource(uri)
                    result = {"contents": [{"uri": uri, "mime_type": "text/plain", "text": out}]}
                elif method == "shutdown":
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"ok": True}}
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()
                    print("[server] shutdown requested, exiting", file=sys.stderr, flush=True)
                    return
                else:
                    raise ValueError(f"Unknown method: {method}")
                resp = {"jsonrpc": "2.0", "id": req_id, "result": result}
            except Exception as e:
                resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32603, "message": str(e)}}

            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        except KeyboardInterrupt:
            print("[server] interrupted, exiting", file=sys.stderr, flush=True)
            break


if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo_mode()
    elif "--stdio" in sys.argv:
        # 强制走自实现 JSON-RPC（教学最稳）
        run_stdio_jsonrpc()
    elif MCP_AVAILABLE:
        # 装了官方 mcp 包就用真 SDK
        print("Starting real MCP server on stdio (FastMCP)…", file=sys.stderr)
        run_real_mcp()
    else:
        # 默认走自实现 JSON-RPC stdio loop（生产质感的 demo）
        run_stdio_jsonrpc()
