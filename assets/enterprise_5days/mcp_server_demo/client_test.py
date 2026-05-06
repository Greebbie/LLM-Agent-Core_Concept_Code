"""Test client for the demo MCP server.

Three modes:
- Stdio JSON-RPC (default): spawns server.py as a subprocess and talks JSON-RPC over stdio.
- Real MCP SDK: used only when the `mcp` package is installed.
- In-process Edu mode: imports the server in the same process for a fast classroom demo.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_root = Path(__file__).resolve().parent
for _ in range(5):
    if (_root / "utils").is_dir() and (_root / "data").is_dir():
        break
    _root = _root.parent
sys.path.insert(0, str(_root))
from utils.mcp_helpers import EduMCPClient, MCP_AVAILABLE


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
            bufsize=1,
        )
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
        """Send one JSON-RPC request and return the result."""
        self._req_id += 1
        req = {"jsonrpc": "2.0", "id": self._req_id, "method": method, "params": params or {}}
        line = json.dumps(req, ensure_ascii=False) + "\n"
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(line)
        self.proc.stdin.flush()
        assert self.proc.stdout is not None
        resp_line = self.proc.stdout.readline()
        if not resp_line:
            raise RuntimeError("Server closed connection unexpectedly")
        resp = json.loads(resp_line.strip())
        if "error" in resp:
            raise RuntimeError(f"JSON-RPC error: {resp['error']}")
        return resp.get("result", {})


def stdio_test() -> None:
    """Start an independent server process and communicate over stdio JSON-RPC."""
    server_script = Path(__file__).parent / "server.py"
    print("-" * 60)
    print("Stdio JSON-RPC client (real subprocess + protocol)")
    print("-" * 60)
    with StdioJsonRpcClient(server_script) as client:
        info = client.request("initialize")
        print(f"\n[OK] Initialized: {info['server_info']['name']} v{info['server_info']['version']}")
        print(f"   Protocol: {info['protocol_version']}")

        tools = client.request("tools/list")
        print(f"\n[TOOLS] {len(tools['tools'])} tools listed:")
        for t in tools["tools"]:
            print(f"  - {t['name']}: {t['description']}")

        print("\n[CALL] query_order(order_id='ORD-001') ...")
        result = client.request("tools/call", {"name": "query_order", "arguments": {"order_id": "ORD-001"}})
        print(f"   <- {result['content'][0]['text']}")

        print("\n[CALL] check_inventory(sku='SKU-A100') ...")
        result = client.request("tools/call", {"name": "check_inventory", "arguments": {"sku": "SKU-A100"}})
        print(f"   <- {result['content'][0]['text']}")


async def real_mcp_test() -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).parent / "server.py")],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("-" * 60)
            print("Real MCP SDK client")
            print("-" * 60)

            tools = await session.list_tools()
            print(f"\n[TOOLS] {len(tools.tools)} tools listed:")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description}")

            print("\n[CALL] query_order_real(order_id='ORD-001') ...")
            result = await session.call_tool("query_order_real", {"order_id": "ORD-001"})
            print(f"   <- {result.content[0].text if result.content else result}")


def inproc_test() -> None:
    from server import build_server

    server = build_server()
    client = EduMCPClient(user_id="demo_user")
    client.connect(server)
    print("-" * 60)
    print("In-process Edu client (fastest, no transport)")
    print("-" * 60)
    tools = client.list_all_tools()
    print(f"\n[TOOLS] {len(tools)} tools:")
    for t in tools:
        print(f"  - [{t['server']}] {t['name']}: {t['description']}")
    result = client.call("enterprise-demo", "query_order", order_id="ORD-001")
    print(f"\n[CALL] query_order(ORD-001) -> {result}")


if __name__ == "__main__":
    if "--inproc" in sys.argv:
        inproc_test()
    elif "--sdk" in sys.argv:
        if not MCP_AVAILABLE:
            print("[WARN] mcp SDK not installed; falling back to stdio JSON-RPC\n")
            stdio_test()
        else:
            asyncio.run(real_mcp_test())
    else:
        stdio_test()
