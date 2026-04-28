"""Patch Day 4 下午 notebook: add a 真 stdio JSON-RPC subprocess demo.

Inserts after the in-process llm_use_mcp demo cell (in section A.4).
Adds:
- 1 markdown explaining MCP protocol essence
- 1 code cell that spawns server.py as subprocess + does JSON-RPC handshake
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_lib import load_nb, save_nb, cell_source, make_md, make_code

PATH = Path("assets/enterprise_5days/instructor/Day4_下午_MCP与Skills.ipynb")


def main():
    nb = load_nb(PATH)
    cells = nb["cells"]

    # Find the cell that does the in-process llm_use_mcp demo (Demo 1: 查订单)
    target_idx = None
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = cell_source(c)
        if "llm_use_mcp" in src and "Demo 2" in src:
            target_idx = i
            break
    if target_idx is None:
        # Fallback: find Exercise 2 (build_basic_server)
        for i, c in enumerate(cells):
            src = cell_source(c)
            if "练习 2" in src and "build_basic_server" in src:
                target_idx = i - 1  # Insert before exercise 2
                break
    if target_idx is None:
        print("⚠ Could not find injection point")
        return

    new_cells = [
        make_md("""---

### A.4.1 · 真起独立 server 进程：subprocess + stdio JSON-RPC（10 min）

上面 `llm_use_mcp` 的 demo 把 server 放在**同一个 Python 进程**里——方便教学，但**不是真实生产方式**。

**真 MCP 协议的本质**：
1. server 是**独立进程**（用任何语言写都行，不止 Python）
2. client 用 **subprocess** 拉起 server
3. 双方走 **JSON-RPC over stdio**（每行一个 JSON 消息）
4. 协议方法：`initialize` / `tools/list` / `tools/call` / `resources/list` / `resources/read` / `shutdown`

下面 demo **真起一个独立 server 进程**（`mcp_server_demo/server.py --stdio`），用真 JSON-RPC 与它通信。

> 💡 这跟 Anthropic 官方 `mcp` Python SDK（需 Python 3.10+）的底层做法**完全一样**，只是我们手写了协议帧，让你看到原貌而不是被 SDK 包起来。Claude Desktop / Cursor / Claude Code 也是这样跟 MCP server 说话。
"""),

        make_code("""# 真起独立 server 进程 + 跑 stdio JSON-RPC 通信
import subprocess, json, time, sys as _sys
from pathlib import Path

server_script = Path("mcp_server_demo/server.py")

# 1. 起 server subprocess
print(f"启动 server: {_sys.executable} {server_script} --stdio")
proc = subprocess.Popen(
    [_sys.executable, str(server_script), "--stdio"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding="utf-8",
    bufsize=1,
)
time.sleep(0.3)  # 让 server 启动

req_id = 0
def rpc_call(method, params=None):
    global req_id
    req_id += 1
    req = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}
    line = json.dumps(req, ensure_ascii=False) + "\\n"
    print(f"  → {method}({params or '{}'})")
    proc.stdin.write(line); proc.stdin.flush()
    resp_line = proc.stdout.readline()
    resp = json.loads(resp_line.strip())
    if "error" in resp:
        print(f"  ← ERROR: {resp['error']}")
        return None
    return resp.get("result", {})

# 2. 协议握手
print("\\n--- Step 1: initialize 协议握手 ---")
info = rpc_call("initialize")
print(f"  ← server={info['server_info']['name']} v{info['server_info']['version']}, protocol={info['protocol_version']}")

# 3. 列出 tools
print("\\n--- Step 2: tools/list 列出能力 ---")
tools = rpc_call("tools/list")
for t in tools["tools"]:
    print(f"  ← {t['name']}: {t['description'][:50]}")

# 4. 真调一个 tool
print("\\n--- Step 3: tools/call 调用 query_order ---")
result = rpc_call("tools/call", {"name": "query_order", "arguments": {"order_id": "ORD-002"}})
print(f"  ← {result['content'][0]['text']}")

# 5. 再调另一个
print("\\n--- Step 4: tools/call 调用 check_inventory ---")
result = rpc_call("tools/call", {"name": "check_inventory", "arguments": {"sku": "SKU-A100"}})
print(f"  ← {result['content'][0]['text']}")

# 6. 关闭
print("\\n--- Step 5: shutdown 关闭 server ---")
rpc_call("shutdown")
proc.terminate()
try:
    proc.wait(timeout=2)
except subprocess.TimeoutExpired:
    proc.kill()

print("\\n💡 看到没？这就是 Claude Desktop / Cursor / Claude Code 跟 MCP server 通信的真实方式。")
print("   每条 JSON-RPC 消息一行，server 是独立进程（任何语言都能写），双方靠 stdio 管道通信。")
"""),
    ]

    cells[target_idx + 1:target_idx + 1] = new_cells
    save_nb(nb, PATH)
    print(f"✓ Patched {PATH}")
    print(f"  Total cells: {len(cells)} (added {len(new_cells)} after [{target_idx}])")


if __name__ == "__main__":
    main()
