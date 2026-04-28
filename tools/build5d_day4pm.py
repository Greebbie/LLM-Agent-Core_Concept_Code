"""Build Day 4 下午 · MCP 协议 (90min) + Skills 协议 (90min) notebook."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_lib import (
    save_nb, make_md, make_code, add_tag,
    make_path_fix_cell, make_lecture_note,
    load_nb, cell_source,
)

OUTPUT = Path("assets/enterprise_5days/instructor/Day4_下午_MCP协议实战.ipynb")


def build():
    cells = []

    cells.append(make_md("""# Day 4 下午 · MCP 协议 (90 min) + Skills 协议 (90 min)

## 学习目标

**A. MCP (Model Context Protocol)** — 90 min
1. 理解 MCP = LLM 工具调用的 USB-C
2. 掌握 Tools / Resources（Prompts 略提）三件套
3. 写一个真实可跑的 MCP server（带权限层）

**B. Anthropic Skills** — 90 min
4. 理解 Skills vs MCP vs Prompts 三协议生态
5. 写一个完整 Skill（SKILL.md + helper.py + reference/）
6. 掌握 Progressive Disclosure & Skills × MCP 集成
7. 生产实践：分发 / 版本 / 安全

## 前置

- Day 3 上午 ReAct Agent 中的 tool 调用基础
- Day 4 上午 Multi-Agent (Agent 之间需要标准化协议)
- 已 `pip install mcp>=0.9` (没装会自动 fallback 到 EduMCPServer)
"""))

    cells.append(make_lecture_note(
        title="""Day 4 下午 · MCP 协议 (90min) + Skills 协议 (90min)""",
        duration_min=180,
        opener="""问：『手机为什么用 USB-C 不用 5 种接口？』 → 类比 MCP。再问：『团队所有人的 Claude 怎么共享你写的工作流？』 → 引出 Skills。""",
        key_points=[
            """**MCP** = Claude 调用外部工具的标准协议 (Tools / Resources / Prompts 三件套)""",
            """**Skills** = 可打包的内化能力（Claude『知道怎么做什么』），与 MCP 配对""",
            """**关键差别**：MCP 解决『能调什么』；Skills 解决『何时做什么 + 怎么做』""",
            """**Progressive Disclosure**：Skills 的杀手特性 — frontmatter 决定何时载入，body+reference 按需加载，不烧 context""",
            """**Skills × MCP**：Skill body 教 Claude 调哪个 MCP tool 完成任务（解耦能力 vs 工具）""",
        ],
        misconceptions=[
            """学员以为 MCP 和 Skills 二选一 → 强调它们是配对，缺一不可""",
            """学员以为 Skills 只是 prompt 模板 → 强调含 helper scripts 和 progressive 加载""",
        ],
        interaction="""现场让学员说一个『部门内部反复使用的 SOP』，把它包成 Skill 草稿（SKILL.md frontmatter）""",
        if_short_on_time="""跳过 Skills × MCP 集成练习 (练习 5)，保 MCP server demo + Skills 写作 + Progressive 主线。""",
    ))

    cells.append(make_path_fix_cell())

    cells.append(make_code("""# 导入：LLM + MCP + Skills helpers
from config import setup
env = setup()
from mcp_helpers import (
    EduMCPServer, EduMCPClient,
    ToolDef, ResourceDef, PromptDef,
    tool_from_function, MCP_AVAILABLE,
)
from skills_helpers import (
    Skill, parse_skill_md, validate_skill,
    discover_skills, match_skill_for_query, load_skill_progressive,
)
import json

llm = env.get_llm()
print(f"✓ LLM 就位")
print(f"✓ MCP SDK 可用: {MCP_AVAILABLE}  (False 走 EduMCPServer 教学模式)")
print(f"✓ Skills helpers 就位")
"""))

    # ============================================================
    # PART A · MCP (90 min)
    # ============================================================
    cells.append(make_md("""---

# PART A · MCP 协议（90 min）

## A.1 · Why MCP（15 min）

### MCP 出现前的乱世

每家 LLM 都有自己的 tool 调用方式：

| 厂商 | 调用方式 |
|---|---|
| OpenAI | `function_call` |
| Anthropic | `tool_use` |
| Google | `function_declarations` |
| Cohere | `connectors` |

**结果**：你要让 LLM 调你公司的 CRM API，要为 N 家 LLM 各写一遍 schema。

### MCP = Model Context Protocol

Anthropic 2024 末推、2025-2026 成事实标准的**跨厂工具调用协议**。

类比 USB-C：写一次 server，所有支持 MCP 的 LLM/IDE 都能用。

### 三件套

```
MCP Server 暴露：
├── Tools      — LLM 可调用的函数（带 JSON schema）
├── Resources  — LLM 可读取的数据源（file / db / api）
└── Prompts    — 可复用的提示模板（带参数）
```

**今天聚焦 Tools + Resources（Prompts 一句话提）。**
"""))

    # ── A.2 Tools ──
    cells.append(make_md("""---

## A.2 · Tools（20 min + 1 练习）

Tool = 一个函数 + description + JSON schema。LLM 看 description 决定调不调；调时按 schema 填参数。
"""))

    cells.append(make_code("""# 用 helper 把普通 Python 函数自动转成 ToolDef
def add(a: int, b: int) -> int:
    \"\"\"Add two integers.\"\"\"
    return a + b

add_tool = tool_from_function(add)

print("ToolDef 自动生成：")
print(f"  name: {add_tool.name}")
print(f"  description: {add_tool.description}")
print(f"  parameters schema: {json.dumps(add_tool.parameters, indent=2)}")
print(f"\\n  call(a=3, b=5) → {add_tool.call(a=3, b=5)}")
"""))

    cells.append(make_code("""# 把多个 Tool 放进 Server，让 LLM 当 client
def multiply(a: int, b: int, label: str = "result") -> str:
    \"\"\"Multiply two integers and return labeled result.\"\"\"
    return f"{label}: {a * b}"

server = EduMCPServer(name="math-helper")
server.add_tool(add_tool)
server.add_tool(tool_from_function(multiply))

# Demo: LLM 看 tools 列表 → 决定调哪个
def llm_calls_tool(query, srv):
    tools = srv.list_tools()
    desc = "\\n".join(f"- {t['name']}({list(t['parameters']['properties'].keys())}): {t['description']}" for t in tools)
    raw = llm.generate(
        f"可用 tools:\\n{desc}\\n\\n用户: {query}\\n\\n输出 JSON: {{\\\"tool\\\": \\\"...\\\", \\\"arguments\\\": {{...}}}}。只输出 JSON。",
        temperature=0,
    ).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    plan = json.loads(raw)
    return f"{plan['tool']}({plan['arguments']}) → {srv.call_tool(plan['tool'], plan['arguments'])}"

print(llm_calls_tool("帮我算 15 加 27", server))
print(llm_calls_tool("把 8 和 9 相乘标记为 'order_total'", server))
"""))

    # ── Exercise 1: Custom tool + schema ──
    cells.append(make_code('''# ============================================================
# 练习 1 | 自定义 Tool + Schema 严格校验
# ============================================================
#
# 【基础】（人人必做，10 min）
#   实现 build_search_tool()：定义 search_employee(name, department=None) 函数
#   并包装成 ToolDef 返回
#
# 【进阶】（技术学员选做，10 min）
#   实现 build_strict_tool(func)：在 tool_from_function 基础上加严格校验：
#   - 缺 required 参数 → ValueError("Missing: ...")
#   - 传未声明参数 → ValueError("Unknown: ...")
# ============================================================

EMPLOYEES = [
    {"name": "张三", "department": "技术部", "level": "senior"},
    {"name": "李四", "department": "市场部", "level": "junior"},
    {"name": "王五", "department": "技术部", "level": "lead"},
]


def build_search_tool():
    """【基础】返回 ToolDef"""
    # ↓↓↓ 【基础】填空（约 6 行）↓↓↓
    def search_employee(name: str, department: str = None):
        results = [e for e in EMPLOYEES if name in e["name"]]
        if department:
            results = [e for e in results if e["department"] == department]
        return json.dumps(results, ensure_ascii=False)
    return tool_from_function(search_employee)
    # ↑↑↑ 【基础】结束 ↑↑↑


def build_strict_tool(func):
    """【进阶】带 unknown 参数校验"""
    # ↓↓↓ 【进阶】填空（约 12 行）↓↓↓
    base = tool_from_function(func)
    declared = set(base.parameters["properties"].keys())
    required = set(base.parameters.get("required", []))
    original = base.func
    def strict(**kwargs):
        missing = required - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing: {sorted(missing)}")
        unknown = set(kwargs.keys()) - declared
        if unknown:
            raise ValueError(f"Unknown: {sorted(unknown)}")
        return original(**kwargs)
    base.func = strict
    return base
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】build_search_tool"); print("=" * 56)
    try:
        tool = build_search_tool()
        assert tool.name == "search_employee"
        result = tool.call(name="张三")
        print(f"  search('张三') → {result}")
        assert "张三" in result
        result = tool.call(name="王", department="技术部")
        print(f"  search('王', dept='技术部') → {result}")
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】build_strict_tool"); print("=" * 56)
    try:
        def divide(a: int, b: int) -> float:
            """Divide a/b"""
            return a / b
        strict = build_strict_tool(divide)
        assert strict.call(a=10, b=2) == 5.0
        print(f"  strict(a=10,b=2) → 5.0 ✓")
        try:
            strict.call(a=10)
        except ValueError as e:
            print(f"  缺参报错: {e} ✓")
        try:
            strict.call(a=10, b=2, c=99)
        except ValueError as e:
            print(f"  未知参数报错: {e} ✓")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''))

    # ── A.3 Resources (demo only) ──
    cells.append(make_md("""---

## A.3 · Resources（15 min · 演示）

Resource = LLM 可**读取的数据源**（不是函数调用）。

| Tool | Resource |
|---|---|
| 函数式（有副作用） | 数据式（只读） |
| 例：`send_email()` | 例：`file:///docs/policy.md` |
"""))

    cells.append(make_code("""# 演示：定义 file resource + dynamic resource
DOCS = {
    "policy_leave.md": "# 请假制度\\n年假：5 年以下 5 天/年；5 年以上 15 天/年。",
    "policy_expense.md": "# 报销制度\\n餐费 ≤ 100 元/餐。差旅一线 500/晚，二三线 350/晚。",
}

server2 = EduMCPServer(name="enterprise-docs")
for fn in DOCS:
    server2.add_resource(ResourceDef(
        uri=f"file:///docs/{fn}",
        name=fn,
        mime_type="text/markdown",
        reader=(lambda f=fn: DOCS[f]),
    ))

# 加一个动态 resource（每次读返回当前时间戳）
import time, random
server2.add_resource(ResourceDef(
    uri="live:///stats",
    name="live_stats",
    mime_type="application/json",
    reader=lambda: json.dumps({"timestamp": time.time(), "active_users": random.randint(50, 200)}),
))

print("可读取的 Resources:")
for r in server2.list_resources():
    print(f"  • {r['uri']}  ({r['mime_type']})")

print(f"\\n读 policy_leave: {server2.read_resource('file:///docs/policy_leave.md')[:60]}...")
print(f"读 live_stats: {server2.read_resource('live:///stats')}")
print("\\n💡 Live Resource 让 LLM 看到的总是『现在』，缓存失效问题自动解决")
"""))

    # ── A.4 Real server + Exercise 2 ──
    cells.append(make_md("""---

## A.4 · 实战：写真实 MCP Server + 权限层（25 min + 1 练习）

我们在 `mcp_server_demo/` 写好了一个**真正可跑的 MCP server**：
- `server.py` — 暴露 3 个企业 tool（订单/库存/通知）
- `client_test.py` — client 测试

下面用 LLM 当大脑试一遍**端到端**流程，并加权限层。
"""))

    cells.append(make_code("""# 复用 mcp_server_demo
import sys
from pathlib import Path
sys.path.insert(0, str(Path('mcp_server_demo')))
from server import build_server as build_demo_server  # type: ignore

demo_server = build_demo_server()
demo_client = EduMCPClient(user_id="alice")
demo_client.connect(demo_server)

print(f"✓ Demo server 就位 ({len(demo_client.list_all_tools())} tools)")
for t in demo_client.list_all_tools():
    print(f"  • {t['name']}: {t['description']}")
"""))

    cells.append(make_code("""# Demo: LLM 用 demo server 完成端到端任务
def llm_use_mcp(query, client):
    tools = client.list_all_tools()
    desc = "\\n".join(f"- [{t['server']}] {t['name']}({list(t['parameters']['properties'].keys())}): {t['description']}" for t in tools)
    raw = llm.generate(
        f"可用工具:\\n{desc}\\n\\n用户: {query}\\n\\n输出 JSON: {{\\\"server\\\": \\\"...\\\", \\\"tool\\\": \\\"...\\\", \\\"arguments\\\": {{...}}}}。只输出 JSON。",
        temperature=0,
    ).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        plan = json.loads(raw)
        result = client.call(plan["server"], plan["tool"], **plan["arguments"])
        answer = llm.generate(
            f"用户问 '{query}'，调 {plan['tool']}({plan['arguments']}) 得到: {result}。请用一句话给最终答复。",
            temperature=0.2,
        )
        return {"plan": plan, "raw": result, "answer": answer.strip()}
    except Exception as e:
        return {"error": str(e), "raw_llm": raw[:200]}


print("=" * 60); print("Demo: 查订单"); print("=" * 60)
print(json.dumps(llm_use_mcp("查 ORD-001", demo_client), ensure_ascii=False, indent=2))

print("\\n" + "=" * 60); print("Demo: 查库存"); print("=" * 60)
print(json.dumps(llm_use_mcp("SKU-A100 还有多少货", demo_client), ensure_ascii=False, indent=2))
"""))

    cells.append(make_code('''# ============================================================
# 练习 2 | MCP Server 加权限层（基于 user_id 控制 tool 可见性）
# ============================================================
#
# 【基础】（人人必做，10 min）
#   build_basic_server()：建含 2 tool (read_orders, get_stats) 的 EduMCPServer
#
# 【进阶】（技术学员选做，15 min）
#   build_server_with_auth()：admin 全开放；viewer 只能调 read_*
#   实现 server.set_auth_check(fn) 校验
# ============================================================

def build_basic_server():
    """【基础】2-tool server"""
    # ↓↓↓ 【基础】填空（约 8 行）↓↓↓
    server = EduMCPServer(name="exercise-server")
    def read_orders():
        """List recent orders"""
        return json.dumps([{"id": "O1", "amt": 100}, {"id": "O2", "amt": 250}])
    def get_stats():
        """Get current stats"""
        return json.dumps({"orders_today": 42, "active_users": 128})
    server.add_tool(tool_from_function(read_orders))
    server.add_tool(tool_from_function(get_stats))
    return server
    # ↑↑↑ 【基础】结束 ↑↑↑


def build_server_with_auth():
    """【进阶】admin 全开放，viewer 只能 read_*"""
    # ↓↓↓ 【进阶】填空（约 16 行）↓↓↓
    server = EduMCPServer(name="auth-server")
    def read_orders():
        """List orders"""
        return json.dumps([{"id": "O1"}])
    def write_order(item: str, qty: int):
        """Create order"""
        return json.dumps({"created": item, "qty": qty})
    def delete_user(user_id: str):
        """Delete user"""
        return json.dumps({"deleted": user_id})
    for fn in [read_orders, write_order, delete_user]:
        server.add_tool(tool_from_function(fn))

    USER_ROLES = {"alice": "admin", "bob": "viewer"}
    def auth_check(user_id, action):
        role = USER_ROLES.get(user_id, "guest")
        if role == "admin":
            return True
        if role == "viewer":
            return action.startswith("read_")
        return False
    server.set_auth_check(auth_check)
    return server
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】build_basic_server"); print("=" * 56)
    try:
        srv = build_basic_server()
        client = EduMCPClient(user_id="any")
        client.connect(srv)
        tools = client.list_all_tools()
        assert len(tools) == 2
        print(f"  Tools: {[t['name'] for t in tools]}")
        result = client.call(srv.name, "read_orders")
        print(f"  read_orders() → {result}")
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】build_server_with_auth"); print("=" * 56)
    try:
        srv = build_server_with_auth()
        admin = EduMCPClient(user_id="alice"); admin.connect(srv)
        viewer = EduMCPClient(user_id="bob"); viewer.connect(srv)
        admin_tools = [t["name"] for t in admin.list_all_tools()]
        viewer_tools = [t["name"] for t in viewer.list_all_tools()]
        print(f"  admin 可见: {admin_tools}")
        print(f"  viewer 可见: {viewer_tools}")
        assert "delete_user" in admin_tools
        assert "delete_user" not in viewer_tools
        try:
            viewer.call(srv.name, "write_order", item="A", qty=1)
            print("  ✗ viewer 不应能调 write_order")
        except PermissionError:
            print("  viewer 调 write_order → 被拒 ✓")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''))

    cells.append(make_md("""### MCP 部分小结

- **Tools** = 函数 + JSON schema；用 `tool_from_function` 自动生成
- **Resources** = LLM 可读数据源；可静态可动态
- **Prompts**（一句话提）= 复用模板，类似 jinja2 但参数化更轻
- **Server + Auth**：生产场景用 `set_auth_check` 按用户限制 tool 可见性

下半场进入 **Skills**——如果说 MCP 解决『LLM 能调什么』，Skills 解决『LLM 知道何时怎么做什么』。
"""))

    # ============================================================
    # PART B · Skills (90 min)
    # ============================================================
    cells.append(make_md("""---

# PART B · Anthropic Skills（90 min）

## B.1 · Why Skills + 三协议对比（15 min）

### 三协议生态

|  | Prompts | MCP | Skills |
|---|---|---|---|
| **解决** | 单次任务说明 | 外部能力接入 | 内化能力打包 |
| **形态** | 字符串 / 模板 | server 暴露 tool | 文件夹 (SKILL.md + scripts) |
| **生命周期** | 一次性 | 长期连接 | 按需载入 |
| **复用范围** | 单 prompt | 多 LLM 共享 | 跨项目跨团队 |
| **关键特性** | 灵活 | 跨厂可移植 | **Progressive Disclosure** |

### 一个 Skill 的样子

```
my_skill/
├── SKILL.md           # 必需：YAML frontmatter + body
├── helper.py          # 可选：Claude 可调用的脚本
└── reference/         # 可选：progressive disclosure 文档
    ├── checklist.md
    └── examples.json
```

`SKILL.md` 头部：
```yaml
---
name: code-review
description: Performs a structured code review... (1-2 句话决定何时被选中)
allowed-tools: [bash, read_file]
version: "0.2"
---

# Code Review Skill
... (body：步骤 / 例子 / 模板)
```

### 关键创新：Progressive Disclosure

**传统 prompt**：把所有指令塞 system prompt → 每次都烧 context
**Skills**：
1. 启动时只读 `description`（几十字）
2. 匹配到再读 `body`（几百字）
3. 需要细节再读 `reference/*.md`（按需）

**省 context = 省钱 + 提速。**
"""))

    # ── B.2 SKILL.md anatomy + Exercise 3 ──
    cells.append(make_md("""---

## B.2 · SKILL.md 解剖 + 写第一个 Skill（25 min + 1 练习）

`skills_demo/code_review/SKILL.md` 是一个完整例子，看一下：
"""))

    cells.append(make_code("""# 解析现成的 code-review skill
fm, body = parse_skill_md("skills_demo/code_review/SKILL.md")
print("Frontmatter:")
for k, v in fm.items():
    print(f"  {k}: {v}")
print(f"\\nBody (前 400 字):\\n{body[:400]}...")

# 也看下 validate
result = validate_skill("skills_demo/code_review")
print(f"\\nvalidate: ok={result['ok']}")
if result['warnings']:
    print(f"  warnings: {result['warnings']}")
"""))

    cells.append(make_code("""# Discovery: 列出 skills_demo/ 里所有 skill
skills = discover_skills("skills_demo")
print(f"发现 {len(skills)} 个 skills:")
for s in skills:
    print(f"  • {s.name} (v{s.version}): {s.description[:80]}...")
    print(f"      helpers: {[p.name for p in s.helper_files]}")
    print(f"      references: {[p.name for p in s.reference_files]}")
"""))

    cells.append(make_code('''# ============================================================
# 练习 3 | 写一个完整的 SKILL.md
# ============================================================
#
# 【基础】（人人必做，10 min）
#   写一个 meeting_notes skill 的 SKILL.md 字符串：
#   - frontmatter: name + description
#   - body: 何时用 + 步骤 (3-5 步)
#   - validate 通过 (ok=True)
#
# 【进阶】（技术学员选做，10 min）
#   实现 score_skill_description(desc, llm)：
#   - 用 LLM 给 description 打分 1-5（清晰度 + 可路由性）
#   - 返回 (score, suggestion)
#   - 演示：好 description 是技术活
# ============================================================
import tempfile, os


def write_meeting_notes_skill():
    """【基础】返回 SKILL.md 完整字符串"""
    # ↓↓↓ 【基础】填空（约 25 行）↓↓↓
    return """---
name: meeting-notes
description: Helps the user summarize a meeting transcript or audio recording into structured notes — action items, decisions, follow-ups. Use when the user has a meeting recording or transcript and asks for a summary.
allowed-tools: [read_file]
version: "0.1"
---

# Meeting Notes Skill

Turn meeting transcripts into structured, actionable notes.

## When to use

- "总结这场会议"
- "从录音里提取 action items"
- "这场会议有什么决定？"

## Workflow

1. **Identify input**: 文件路径 / 直接粘贴的 transcript
2. **Extract**:
   - **决定 (Decisions)**: 拍板的事项
   - **Action items**: 谁、做什么、何时
   - **Follow-ups**: 待跟进 / 待澄清
3. **Format** as markdown:
   ```markdown
   ## Decisions
   - ...

   ## Action Items
   - [ ] @<owner> <task> by <date>

   ## Follow-ups
   - ...
   ```
4. **Verify** with the user before sending out.

## What this skill does NOT

- Translate / transcribe (use a STT skill)
- Send the notes (use a notification skill)
"""
    # ↑↑↑ 【基础】结束 ↑↑↑


def score_skill_description(desc, llm):
    """【进阶】LLM 评分 description 质量"""
    # ↓↓↓ 【进阶】填空（约 14 行）↓↓↓
    prompt = f"""一个好的 Skill description 应该：
1. 一句话清楚说『何时用』(明确的触发条件)
2. 包含动词 + 对象 (做什么)
3. 长度 30-200 字
4. 让 Claude 一眼能与其他 skill 区分

下面这个 description 你给几分 (1-5)？再给一句改进建议。
description: {desc}

输出格式：
SCORE: <1-5>
SUGGESTION: <一句话>"""
    raw = llm.generate(prompt, temperature=0).strip()
    score = 3
    suggestion = ""
    for line in raw.splitlines():
        if line.upper().startswith("SCORE:"):
            try:
                score = int("".join(c for c in line if c.isdigit())[:1])
            except (ValueError, IndexError):
                pass
        elif line.upper().startswith("SUGGESTION:"):
            suggestion = line.split(":", 1)[1].strip()
    return (score, suggestion)
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】write_meeting_notes_skill"); print("=" * 56)
    try:
        skill_md = write_meeting_notes_skill()
        # 写到临时文件夹验证
        with tempfile.TemporaryDirectory() as tmp:
            sd = os.path.join(tmp, "meeting-notes")
            os.makedirs(sd)
            with open(os.path.join(sd, "SKILL.md"), "w", encoding="utf-8") as f:
                f.write(skill_md)
            result = validate_skill(sd)
            print(f"  validate: ok={result['ok']}, warnings={result['warnings']}")
            assert result["ok"], f"validate 不通过：{result['errors']}"
            fm, body = parse_skill_md(os.path.join(sd, "SKILL.md"))
            assert fm["name"] == "meeting-notes"
            assert len(fm["description"]) > 30
            print(f"  name: {fm['name']}  desc 长度: {len(fm['description'])}  body 长度: {len(body)}")
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】score_skill_description"); print("=" * 56)
    try:
        good = "Helps the user write structured meeting notes from transcripts. Extracts action items, decisions, and follow-ups. Use when the user has a meeting recording or transcript."
        bad = "AI 助手"
        score_g, sug_g = score_skill_description(good, llm)
        score_b, sug_b = score_skill_description(bad, llm)
        print(f"  好 desc: score={score_g} | suggestion: {sug_g[:80]}")
        print(f"  差 desc: score={score_b} | suggestion: {sug_b[:80]}")
        assert score_g >= score_b
        print("✅ 进阶通过 — 好 description 评分更高")
    except NotImplementedError:
        print("⏭ 进阶跳过")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''))

    # ── B.3 Helper + Progressive ──
    cells.append(make_md("""---

## B.3 · Helper Scripts + Progressive Disclosure（20 min + 1 练习）

### Helper Scripts

Skills 不是死文档——可以挂 Python 脚本。Claude 会读 `helper.py` 的 docstring，按需调用其中函数。

例：`code_review/helper.py` 提供 `run_checks(target)` 跑 ruff + mypy。Skill body 教 Claude 在第 2 步调用它。

### Progressive Disclosure（杀手特性）

**Naive 做法**：所有 skill 全部内容塞进 system prompt → context 爆掉

**Skills 做法**：3 层载入
1. **Always**: discovery 时只读 `description`（几十字）
2. **On match**: 用户 query 匹配 → 加载 body（几百字）
3. **On demand**: query 涉及细节 → 加载相关 `reference/*.md`（按需）

下面演示 `load_skill_progressive`：
"""))

    cells.append(make_code("""# Progressive disclosure 实战
skills = discover_skills("skills_demo")
code_review = next(s for s in skills if s.name == "code-review")

# 场景 A：query 简单 → 只载入 body，不读 reference/checklist.md
loaded_a = load_skill_progressive(code_review, "review this function: def add(a,b): return a+b", llm)
print("=" * 56)
print("场景 A: '简单函数 review'")
print("=" * 56)
print(f"  loaded references: {[r['name'] for r in loaded_a['references_loaded']]}")
print(f"  estimated tokens: {loaded_a['tokens_estimate']}")

# 场景 B：query 复杂 + 提到团队 → LLM 决定加载 checklist.md
loaded_b = load_skill_progressive(
    code_review,
    "review this PR — 我们团队对错误处理特别敏感，要走完整 checklist",
    llm,
)
print("\\n" + "=" * 56)
print("场景 B: '完整 checklist review'")
print("=" * 56)
print(f"  loaded references: {[r['name'] for r in loaded_b['references_loaded']]}")
print(f"  estimated tokens: {loaded_b['tokens_estimate']}")
print(f"\\n💡 场景 A 省了 ~{loaded_b['tokens_estimate'] - loaded_a['tokens_estimate']} tokens — 简单任务不需要全文档")
"""))

    cells.append(make_code('''# ============================================================
# 练习 4 | validate_skill + LLM 路由 (match_skill_for_query)
# ============================================================
#
# 【基础】（人人必做，10 min）
#   实现 validate_all(skills_dir)：跑 validate_skill 检查所有子目录，返回 dict {name: {ok, errors, warnings}}
#
# 【进阶】（技术学员选做，15 min）
#   实现 audit_routing(test_queries, skills, llm)：
#   - 一组 (query, expected_skill_name) pairs
#   - 用 match_skill_for_query 路由，统计准确率
#   - 错路由 case 列出来供 description 改进
# ============================================================

def validate_all(skills_dir):
    """【基础】批量 validate"""
    # ↓↓↓ 【基础】填空（约 6 行）↓↓↓
    from pathlib import Path as P
    out = {}
    for sub in sorted(P(skills_dir).iterdir()):
        if sub.is_dir() and (sub / "SKILL.md").exists():
            out[sub.name] = validate_skill(sub)
    return out
    # ↑↑↑ 【基础】结束 ↑↑↑


def audit_routing(test_queries, skills, llm):
    """【进阶】路由准确率审计"""
    # ↓↓↓ 【进阶】填空（约 14 行）↓↓↓
    correct = 0
    audit_log = []
    for query, expected in test_queries:
        picked = match_skill_for_query(query, skills, llm)
        picked_name = picked.name if picked else None
        ok = picked_name == expected
        if ok:
            correct += 1
        audit_log.append({
            "query": query,
            "expected": expected,
            "picked": picked_name,
            "ok": ok,
        })
    return {
        "accuracy": correct / len(test_queries) if test_queries else 0,
        "errors": [a for a in audit_log if not a["ok"]],
        "log": audit_log,
    }
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】validate_all('skills_demo')"); print("=" * 56)
    try:
        report = validate_all("skills_demo")
        for name, r in report.items():
            status = "✓" if r["ok"] else "✗"
            print(f"  {status} {name}: errors={r['errors']}, warnings={len(r['warnings'])}")
        assert all(r["ok"] for r in report.values()), "应所有 demo skill 都 valid"
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】audit_routing"); print("=" * 56)
    try:
        skills = discover_skills("skills_demo")
        test_cases = [
            ("review my python code", "code-review"),
            ("查 ORD-005 订单", "db-query"),
            ("入职 5 年年假几天", "enterprise-knowledge-assistant"),
            ("SKU-A100 库存多少", "db-query"),
        ]
        result = audit_routing(test_cases, skills, llm)
        print(f"  路由准确率: {result['accuracy']:.0%}")
        for log in result["log"]:
            mark = "✓" if log["ok"] else "✗"
            print(f"    {mark} '{log['query'][:40]}' → expected={log['expected']}, picked={log['picked']}")
        if result["errors"]:
            print(f"\\n  💡 错路由 case 是改 description 的输入")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''))

    # ── B.4 Skills × MCP integration ──
    cells.append(make_md("""---

## B.4 · Skills × MCP 集成模式（20 min + 1 练习）

```
┌─────────────────┐         ┌──────────────────┐
│  Skills          │         │  MCP             │
│  (能力 / 何时做) │ ─────▶ │  (工具 / 怎么调) │
└─────────────────┘         └──────────────────┘

例：db-query Skill 教 Claude『查订单』，actual API 调用走 enterprise-demo MCP server
```

`skills_demo/db_query/SKILL.md` 的 `allowed-tools` 字段限制了它只能调 3 个 MCP tool：
- `mcp__enterprise-demo__query_order`
- `mcp__enterprise-demo__check_inventory`
- `mcp__enterprise-demo__send_notification`

下面看完整流程。
"""))

    cells.append(make_code("""# 完整流程：用户 query → 路由到 Skill → Skill 调 MCP → 返回
skills = discover_skills("skills_demo")
demo_server = build_demo_server()  # 复用前面的 MCP server
demo_client = EduMCPClient(user_id="demo")
demo_client.connect(demo_server)


def skill_calls_mcp(query):
    \"\"\"端到端：query → 选 skill → 让 Claude 按 skill body 决定调 MCP tool\"\"\"
    # 1. Skill discovery + routing
    skill = match_skill_for_query(query, skills, llm)
    if skill is None:
        return f"[no matching skill] {query}"
    # 2. Load skill body (progressive)
    loaded = load_skill_progressive(skill, query, llm)
    # 3. Skill body 指导 Claude 调哪个 MCP tool
    tools = demo_client.list_all_tools()
    desc = "\\n".join(f"- {t['name']}({list(t['parameters']['properties'].keys())})" for t in tools)
    plan_prompt = f'''Skill: {skill.name}
Skill 指导:
{loaded['body'][:600]}

可用 MCP tools:
{desc}

用户 query: {query}

按 skill 指导决定调哪个 tool。输出 JSON: {{"tool": "...", "arguments": {{...}}}}。'''
    raw = llm.generate(plan_prompt, temperature=0).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        plan = json.loads(raw)
        result = demo_client.call(demo_server.name, plan["tool"], **plan["arguments"])
        return {"skill": skill.name, "tool": plan["tool"], "result": result}
    except Exception as e:
        return {"skill": skill.name, "error": str(e)}


# Demo
print("=" * 60); print("Demo 1: '查 ORD-002'"); print("=" * 60)
print(json.dumps(skill_calls_mcp("查 ORD-002 订单状态"), ensure_ascii=False, indent=2))

print("\\n" + "=" * 60); print("Demo 2: '库存查 SKU-A100'"); print("=" * 60)
print(json.dumps(skill_calls_mcp("SKU-A100 还有多少库存？"), ensure_ascii=False, indent=2))
"""))

    cells.append(make_code('''# ============================================================
# 练习 5 | 写一个新 Skill 调用现有 MCP server
# ============================================================
#
# 【基础】（人人必做，10 min）
#   写 notify_skill_md：返回字符串 — 一个 notify-customer skill
#   - description 提到『何时用：发通知 / 提醒 / 告知』
#   - body 教 Claude 用 send_notification tool
#   - allowed-tools 含 mcp__enterprise-demo__send_notification
#
# 【进阶】（技术学员选做，15 min）
#   写 multi_tool_skill_md：一个 skill 调 ≥ 2 个 MCP tool
#   场景：『发货前自动检查 + 通知』
#   - 先 check_inventory(sku)
#   - 库存够 → query_order(order_id) 拿客户名
#   - 然后 send_notification 给客户
#   body 要清楚教 Claude 这个 3 步流程
# ============================================================

def notify_skill_md():
    """【基础】返回 SKILL.md 字符串"""
    # ↓↓↓ 【基础】填空（约 16 行）↓↓↓
    return """---
name: notify-customer
description: Sends a notification to a customer/user via the enterprise notification system. Use when the user wants to inform, alert, or remind a specific person about something (order shipped, inventory ready, account issue, etc).
allowed-tools: [mcp__enterprise-demo__send_notification]
version: "0.1"
---

# Notify Customer Skill

Send a single notification to a user.

## When to use

- "通知 alice 她的订单到了"
- "发个提醒给 bob"
- "告诉 carol 库存已补"

## Workflow

1. 解析: 谁 (user_id) + 内容 (message)
2. 调 `mcp__enterprise-demo__send_notification(user_id=..., message=...)`
3. 确认发出 + 反馈给用户

## Examples

| 用户说 | tool 调用 |
|---|---|
| "通知 alice 订单到了" | `send_notification(user_id="alice", message="订单到了")` |
| "提醒 bob 续费" | `send_notification(user_id="bob", message="请续费")` |
"""
    # ↑↑↑ 【基础】结束 ↑↑↑


def multi_tool_skill_md():
    """【进阶】3 步发货 skill"""
    # ↓↓↓ 【进阶】填空（约 24 行）↓↓↓
    return """---
name: pre-ship-check
description: Performs the pre-shipment workflow — checks inventory, looks up the customer for an order, then notifies them that the order is ready to ship. Use when the user wants to do a "ready to ship" or "pre-ship" workflow for an existing order.
allowed-tools:
  - mcp__enterprise-demo__check_inventory
  - mcp__enterprise-demo__query_order
  - mcp__enterprise-demo__send_notification
version: "0.1"
---

# Pre-Ship Check Skill

3-step workflow: 检库存 → 查客户 → 通知。

## When to use

- "ORD-001 准备发货"
- "检查 ORD-XXX 是否能发货并通知客户"
- "走发货前检查"

## Workflow (强制 3 步顺序)

1. **Check inventory**: 调 `check_inventory(sku=...)`
   - 库存 = 0 → 中止，告诉用户『缺货，无法发货』
2. **Look up order**: 调 `query_order(order_id=...)` 拿 customer 字段
3. **Notify**: 调 `send_notification(user_id=<customer>, message="您的订单 <id> 即将发货")`

## Error handling

- 任何一步失败 → 不继续后面步骤，回滚之前的 side effects（这里只 send_notification 有 side effect）
- 报告每步 PASS / FAIL 状态

## Output format

```
[1/3] check_inventory(SKU-XXX) → in_stock: <yes/no>
[2/3] query_order(ORD-XXX) → customer: <name>
[3/3] send_notification(<name>, ...) → sent
```
"""
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】notify_skill_md"); print("=" * 56)
    try:
        md = notify_skill_md()
        with tempfile.TemporaryDirectory() as tmp:
            sd = os.path.join(tmp, "notify-customer")
            os.makedirs(sd)
            with open(os.path.join(sd, "SKILL.md"), "w", encoding="utf-8") as f:
                f.write(md)
            result = validate_skill(sd)
            assert result["ok"], result["errors"]
            fm, body = parse_skill_md(os.path.join(sd, "SKILL.md"))
            assert "send_notification" in fm.get("allowed-tools", [])[0]
            assert "通知" in body or "notify" in body.lower()
        print(f"  validate ok + 含 send_notification + body 含通知关键词")
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】multi_tool_skill_md"); print("=" * 56)
    try:
        md = multi_tool_skill_md()
        # 检查 body 含三步流程关键词
        for kw in ["check_inventory", "query_order", "send_notification"]:
            assert kw in md, f"应包含 {kw}"
        # frontmatter 含 3 个 allowed-tools (multiline 形式)
        assert md.count("mcp__enterprise-demo__") >= 3, "应有 3 个 MCP tool"
        print("  ✓ body 含三步流程")
        print("  ✓ allowed-tools 含 3 个 MCP tools")
        print("✅ 进阶通过 — 复合 skill 适合多步业务流程")
    except NotImplementedError:
        print("⏭ 进阶跳过")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''))

    # ── B.5 生产实践 ──
    cells.append(make_md("""---

## B.5 · 生产实践 + 总结（10 min）

### 分发

| 方式 | 适合 |
|---|---|
| `~/.claude/skills/` 个人本地 | 个人助理 / 试验 |
| Git repo 团队共享 | 团队复用，version control |
| Claude.ai 上传 | 企业用户跨设备同步 |
| Anthropic Skill Marketplace（2026 起） | 公开发布 |

### 版本管理

`SKILL.md` frontmatter 加 `version: "1.0"`。改 description 或 body 时手动 bump。

```
my-skill/
├── SKILL.md   (version: 1.0)
└── CHANGELOG.md
    - v1.0: initial
    - v1.1: tightened description, added reference/sql_examples
```

### 安全

- helper.py 跑哪个用户/进程权限？默认是 Claude 进程权限——**别在 helper 里干 sudo / rm -rf**
- `allowed-tools` 字段限定 MCP tool 子集，违规会被 Claude 拒绝
- reference/ 不要塞密钥 / PII

### 测试一个 Skill 的好坏

1. **`description` 路由测试**：N 个 query，观察 LLM 选中率（练习 4 进阶）
2. **`body` 指令清晰度**：让 Claude 跑同 query 5 次，输出一致吗？
3. **`reference/` 进度披露**：哪些 reference 真的被加载？没加载的可能是写得太冷门

### 与 MCP 的协同

- Skills 描述『何时做 + 步骤』；MCP 提供『可调的 tool』
- 一个 skill 通过 `allowed-tools` 锁定它能用的 MCP tool 子集
- 同一组 MCP tool 可被多个 skill 复用（不重写）

### 何时不要用 Skill

- 一次性任务（写 prompt 够用）
- 高度动态的指令（每次都不一样）
- 与 LLM 模型强耦合（换模型就坏）
"""))

    cells.append(make_md("""---

## Day 4 下午 总结

| 协议 | 解决 | 形态 | 学到了 |
|---|---|---|---|
| **Prompts** | 单次任务 | 字符串 | (Day 3 已学) |
| **MCP** | 外部能力 | server 暴露 tool | Tools/Resources/真实 server + 权限 |
| **Skills** | 内化能力 | 文件夹 (SKILL.md+scripts) | 写法 + Progressive Disclosure + 配 MCP |

### 为何今天把 MCP + Skills 放一起

它们是 Anthropic 2026 战略的**两个互补支柱**：
- MCP = 把能力接进来（外部 API → LLM 可调）
- Skills = 把能力打包出去（团队 SOP → 可复用）

明天 Day 5 上午 **Agentic RAG**，下午 **LLMOps + 升级 Capstone**。Day 5 下午**最后 30 min** 我们会把整个升级 Capstone 打包成一个 `enterprise-knowledge-assistant` Skill，演示『5 天合体 → 一个文件夹可复用』。

### 推荐资料

- MCP 官方: https://modelcontextprotocol.io
- Anthropic Skills 文档（搜『Claude Skills』）
- 现成 community MCP servers: github / gmail / slack / postgres
- `skills_demo/` 三个完整可改的 skill 例子
"""))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    save_nb(nb, OUTPUT)

    # Tag exercises
    nb2 = load_nb(OUTPUT)
    n_tagged = 0
    for c in nb2["cells"]:
        if c["cell_type"] != "code":
            continue
        src = cell_source(c)
        first = src.strip().split("\n")[0]
        if first.startswith("# =====") and "练习" in src[:200]:
            add_tag(c, "fillin")
            add_tag(c, "batch5")
            n_tagged += 1
    save_nb(nb2, OUTPUT)

    print(f"✓ Built {OUTPUT}")
    print(f"  Total cells: {len(cells)} | tagged {n_tagged} fillin")


if __name__ == "__main__":
    build()
