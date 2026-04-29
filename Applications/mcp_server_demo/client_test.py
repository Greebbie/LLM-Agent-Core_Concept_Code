"""Test client for the demo MCP server.

Three modes:
- Stdio JSON-RPC (DEFAULT): spawns server.py as subprocess, talks JSON-RPC over stdio.
                            This is what real MCP clients do at the wire level.
- Real MCP SDK: only if `mcp` Python pkg installed (Python 3.10+).
- In-process Edu mode: imports server in same process (no transport).

Usage:
    python client_test.py                # auto: stdio JSON-RPC (production-flavor)
    python client_test.py --inproc       # in-process (fastest demo)
    python client_test.py --sdk          # real Anthropic mcp SDK if available
"""
from __future__ import annotations
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.mcp_helpers import EduMCPClient, MCP_AVAILABLE


# ============================================================
# Stdio JSON-RPC client — talks to server.py via subprocess pipes
# ============================================================
class StdioJsonRpcClient:
    """Real subprocess + stdio JSON-RPC client. Same wire pattern as official MCP."""

    def __init__(self, server_script: str | Path):
        self.server_script = str(server_script)
        self.proc: subprocess.Popen | None = None
        self._req_id = 0

    def __enter__(self):
        self.proc = subprocess.Popen(
            [sys.executable, self.server_script, "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,  # line-buffered
        )
        # Give server a moment to start
        time.sleep(0.3)
        return self

    def __exit__(self, *args):
        if self.proc:
            try:
                self.request("shutdown")
            except Exception:
                pass
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def request(self, method: str, params: dict | None = None) -> dict:
        """Send a JSON-RPC request, return the result (or raise on error)."""
        self._req_id += 1
        req = {"jsonrpc": "2.0", "id": self._req_id, "method": method, "params": params or {}}
        line = json.dumps(req, ensure_ascii=False) + "\n"
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(line)
        self.proc.stdin.flush()
        # Read one response line
        assert self.proc.stdout is not None
        resp_line = self.proc.stdout.readline()
        if not resp_line:
            raise RuntimeError("Server closed connection unexpectedly")
        resp = json.loads(resp_line.strip())
        if "error" in resp:
            raise RuntimeError(f"JSON-RPC error: {resp['error']}")
        return resp.get("result", {})


def stdio_test():
    """真起一个独立 server 进程，通过 stdio JSON-RPC 通信。"""
    server_script = Path(__file__).parent / "server.py"
    print("─" * 60)
    print("Stdio JSON-RPC client (real subprocess + protocol)")
    print("─" * 60)
    with StdioJsonRpcClient(server_script) as client:
        # 1. Initialize handshake
        info = client.request("initialize")
        print(f"\n🤝 Initialized: {info['server_info']['name']} v{info['server_info']['version']}")
        print(f"   Protocol: {info['protocol_version']}")

        # 2. List tools
        tools = client.request("tools/list")
        print(f"\n📋 {len(tools['tools'])} tools listed:")
        for t in tools["tools"]:
            print(f"  • {t['name']}: {t['description']}")

        # 3. Call query_order
        print("\n📞 calling query_order(order_id='ORD-001') ...")
        result = client.request("tools/call", {"name": "query_order", "arguments": {"order_id": "ORD-001"}})
        print(f"   ← {result['content'][0]['text']}")

        # 4. Call check_inventory
        print("\n📞 calling check_inventory(sku='SKU-A100') ...")
        result = client.request("tools/call", {"name": "check_inventory", "arguments": {"sku": "SKU-A100"}})
        print(f"   ← {result['content'][0]['text']}")


async def real_mcp_test():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).parent / "server.py")],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("─" * 60)
            print("Real Anthropic MCP SDK client")
            print("─" * 60)

            tools = await session.list_tools()
            print(f"\n📋 {len(tools.tools)} tools listed:")
            for t in tools.tools:
                print(f"  • {t.name}: {t.description}")

            print("\n📞 calling query_order_real(order_id='ORD-001') ...")
            result = await session.call_tool("query_order_real", {"order_id": "ORD-001"})
            print(f"   ← {result.content[0].text if result.content else result}")


def inproc_test():
    from server import build_server
    server = build_server()
    client = EduMCPClient(user_id="demo_user")
    client.connect(server)
    print("─" * 60)
    print("In-process Edu client (fastest, no transport)")
    print("─" * 60)
    tools = client.list_all_tools()
    print(f"\n📋 {len(tools)} tools:")
    for t in tools:
        print(f"  • [{t['server']}] {t['name']}: {t['description']}")
    result = client.call("enterprise-demo", "query_order", order_id="ORD-001")
    print(f"\n📞 query_order(ORD-001) → {result}")


if __name__ == "__main__":
    if "--inproc" in sys.argv:
        inproc_test()
    elif "--sdk" in sys.argv:
        if not MCP_AVAILABLE:
            print("⚠ mcp SDK not installed; falling back to stdio JSON-RPC\n")
            stdio_test()
        else:
            asyncio.run(real_mcp_test())
    else:
        # Default: stdio JSON-RPC (real subprocess + protocol)
        stdio_test()
