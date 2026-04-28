"""Patch Day 4 下午 notebook: append a concrete '用 Skill 实操' demo block.

Adds before B.5 (生产实践) section:
- 1 markdown intro: 实操对比 (无 skill vs 有 skill)
- 1 code cell: code_review skill 端到端跑通 + helper.py 调用
- 1 code cell: naive review (无 skill) vs skill-driven review 输出对比
- 1 markdown: 何时 skill 真正值得
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_lib import load_nb, save_nb, cell_source, set_cell_source, make_md, make_code

PATH = Path("assets/enterprise_5days/instructor/Day4_下午_MCP协议实战.ipynb")


def main():
    nb = load_nb(PATH)
    cells = nb["cells"]

    # Find B.5 markdown (生产实践)
    b5_idx = None
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown" and "B.5" in cell_source(c) and "生产实践" in cell_source(c):
            b5_idx = i
            break
    if b5_idx is None:
        print("⚠ Could not find B.5 markdown; appending at end before 总结")
        for i, c in enumerate(cells):
            if c["cell_type"] == "markdown" and "Day 4 下午 总结" in cell_source(c):
                b5_idx = i
                break
    if b5_idx is None:
        b5_idx = len(cells)

    new_cells = [
        make_md("""---

## B.4.5 · 实操：用 code_review Skill 审查一段含 bug 的代码（15 min）

到这里学员可能仍觉得 Skills "概念懂了，但具体能给我做什么？"

下面用 `code_review/` skill 跑一个**真实任务**：

```
任务情景: 我有一段 Python 代码，疑似有 bug + 风格问题。
        想要 Claude 按团队 checklist 帮我审一遍。

无 skill 怎么办: 写个 prompt"review this code" → 输出格式不稳，
                每个 reviewer 可能给不同维度的反馈。

有 skill 怎么办: discover code-review skill → 自动按 SKILL.md
                的 5 步 workflow + checklist.md 出结构化报告。
```
"""),

        make_code('''# 实操 1: 用 code_review skill 端到端审查
import sys, json
from pathlib import Path

# 待审的代码（含若干典型问题）
buggy_code = """
def calc_discount(price, discount):
    # 没参数校验、负价格不报错、整数除法可能出问题
    final = price - price * discount
    return final

def process_orders(orders):
    # 异常被吞、空列表会报错
    total = 0
    try:
        for o in orders:
            total = total + o['amount']
    except:
        pass
    return total / len(orders)

def get_user(user_id):
    # SQL 注入、明文密码
    sql = "SELECT * FROM users WHERE id = " + str(user_id)
    return execute(sql)
"""

# Step 1: discover + 路由到 code-review
skills = discover_skills("skills_demo")
picked = match_skill_for_query("review this Python code for bugs and style issues", skills, llm)
print(f"Step 1 路由: {picked.name if picked else '(none)'}")

# Step 2: progressive load — 简单 query 不需 reference
loaded = load_skill_progressive(picked, "review this Python code", llm)
print(f"Step 2 加载: body {len(loaded['body'])} 字符 + {len(loaded['references_loaded'])} 个 reference")

# Step 3: 让 LLM 按 skill body 跑审查 workflow
review_prompt = f"""你是 code-review skill 的执行体。严格按下面的 workflow 审查代码。

[SKILL body]
{loaded['body']}

[要审的代码]
```python
{buggy_code}
```

按 workflow 第 4 步的 Output format 输出结构化报告。
"""
report = llm.generate(review_prompt, temperature=0.1)
print("\\n" + "=" * 60)
print("Step 3 输出（按 SKILL workflow 的结构化报告）:")
print("=" * 60)
print(report)
'''),

        make_code('''# 实操 2: 对比 — 无 skill (naive) vs 有 skill 输出质量
# 同一段 buggy_code，看裸 prompt 与 skill-driven 的差异

naive_prompt = f"Review this code:\\n```python\\n{buggy_code}\\n```"
naive_report = llm.generate(naive_prompt, temperature=0.1)

print("=" * 60)
print("【无 Skill】裸 prompt 'Review this code'")
print("=" * 60)
print(naive_report[:600])
print(f"\\n... 总长 {len(naive_report)} 字符")

print("\\n" + "=" * 60)
print("【有 Skill】code-review skill 按 workflow 跑")
print("=" * 60)
print(report[:600])
print(f"\\n... 总长 {len(report)} 字符")

# 量化对比
def has_section(text, keyword):
    return keyword in text

print("\\n" + "=" * 60)
print("结构化指标对比:")
print("=" * 60)
print(f"{'指标':<30} {'裸 prompt':<10} {'Skill 驱动':<10}")
print("-" * 60)
for marker in ["🔴 Blocking", "🟡 Should fix", "🟢 Nice", "files reviewed", "summary"]:
    lower = marker.lower()
    n_n = "✓" if (marker in naive_report or lower in naive_report.lower()) else "✗"
    n_s = "✓" if (marker in report or lower in report.lower()) else "✗"
    print(f"  含『{marker}』{'':<{30-len(marker)-2}} {n_n:<10} {n_s:<10}")

print("""
💡 关键观察:
  - Skill 驱动的输出**总有相同结构**（团队跨人复用、可比对）
  - 裸 prompt 每次格式可能漂移（不同 reviewer 不同角度）
  - Skill 的 workflow 自带 checklist，更不容易漏（如安全/测试维度）
  - Token 略多但质量明显更稳——对企业流程化场景值得
"""[1:])
'''),

        make_md("""### 何时 Skill 真正值得？

| 场景 | 推荐 |
|---|---|
| 一次性 / ad-hoc 任务 | 直接 prompt，不用 skill |
| **团队反复跑同一类工作流**（code review / 周报 / 会议纪要 / SOP）| ✅ Skill |
| 需要**结构化输出**给下游系统消费 | ✅ Skill |
| 需要**可观测 / 可审计**（每次走相同步骤）| ✅ Skill |
| 任务需要**多个 helper script 协作** | ✅ Skill |
| Prompt > 200 字 + 含步骤 + 含示例 | ✅ Skill（已经够复杂了） |

**rule of thumb**: 同样的指令你写给同事第 3 遍 → 该封 Skill 了。

### 用 Skill 你刚才完成了什么

你刚刚把一个**团队 Code Review SOP** 用 1 个 SKILL.md + 1 个 helper.py + 1 个 checklist.md 表达出来。任何团队成员（或他们的 Claude / Cursor / Claude Code）只要 `import` 这个 skill 文件夹就能复用——**不需要重写 prompt，不需要培训新人**。
"""),
    ]

    cells[b5_idx:b5_idx] = new_cells
    save_nb(nb, PATH)
    print(f"✓ Patched {PATH}")
    print(f"  Total cells: {len(cells)} (added {len(new_cells)})")
    print(f"  Inserted before B.5 at idx {b5_idx}")


if __name__ == "__main__":
    main()
