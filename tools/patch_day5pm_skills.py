"""Patch Day 5 下午 notebook: append Capstone-as-Skill section (30 min).

Runs after build5d_day5pm.py. Adds:
- 1 markdown: 块 3 简介
- 1 markdown: discover + match
- 2 code: skill demo + load_progressive
- 1 Checkpoint 4 (基础 + 进阶)
- Updated 总结
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_lib import (
    load_nb, save_nb, cell_source, set_cell_source,
    make_md, make_code, add_tag,
)

PATH = Path("assets/enterprise_5days/instructor/Day5_下午_LLMOps与生产Capstone.ipynb")


def main():
    nb = load_nb(PATH)
    cells = nb["cells"]

    # Find the 总结 markdown (last) — insert new section BEFORE it, then update 总结
    summary_idx = None
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown" and "Day 5 下午 总结" in cell_source(c):
            summary_idx = i
            break
    if summary_idx is None:
        # Append at end
        summary_idx = len(cells)

    # Build new cells
    new_cells = [
        make_md("""---

## 块 3 · Capstone → Skill 打包（30 min）

我们已经把 5 天所有内容合体跑出一个升级 Capstone。最后 30 min**演示如何把它打包成一个 Skill**，让其他团队（或你自己未来的项目）一键复用。

### Capstone Skill 文件夹

`skills_demo/capstone_assistant/` 已经打包好了：

```
capstone_assistant/
├── SKILL.md                # 何时调 + workflow（决定 Claude 选不选它）
├── pipeline.py             # upgraded_pipeline() 主入口（从本 notebook 抽出）
├── eval.py                 # batch_eval 复用
└── reference/
    ├── architecture.md     # 4 大组件 deep dive (按需载入)
    └── eval_cases.jsonl    # 10 个评测用例
```

### 演示：discover + route + invoke
"""),

        make_code("""# Step 1: discover skills
import sys
from pathlib import Path
sys.path.insert(0, "utils")
from skills_helpers import discover_skills, match_skill_for_query, load_skill_progressive, validate_skill

skills = discover_skills("skills_demo")
print(f"发现 {len(skills)} 个 skills:")
for s in skills:
    print(f"  • {s.name}: {s.description[:80]}...")
    print(f"      version={s.version}, helpers={[p.name for p in s.helper_files]}")
"""),

        make_code("""# Step 2: 用户来一个企业相关 query → LLM 路由到 capstone-assistant skill
test_queries = [
    "入职 7 年有几天年假？",
    "查 ORD-002 订单状态",
    "code review 这个函数",
]

for q in test_queries:
    picked = match_skill_for_query(q, skills, llm)
    print(f"  query: {q[:30]}")
    print(f"    → routed to: {picked.name if picked else '(none)'}")
"""),

        make_code("""# Step 3: 加载并调用 capstone_assistant.pipeline.upgraded_pipeline
# (注意：pipeline.py 已自动 setup env + 建 vector store + 起 mcp server)

import importlib.util
spec = importlib.util.spec_from_file_location(
    "capstone_pipeline",
    "skills_demo/capstone_assistant/pipeline.py",
)
capstone_module = importlib.util.module_from_spec(spec)

print("加载 capstone_assistant skill 的 pipeline.py...")
print("(首次会 setup env + 建 vector store + 启 mcp server，需 5-10s)")
spec.loader.exec_module(capstone_module)

# 直接调用打包后的 pipeline
result = capstone_module.upgraded_pipeline("入职 8 年有几天年假？")
print(f"\\n打包后 Capstone 调用结果:")
print(f"  path: {result['path']}")
print(f"  answer: {result['answer'][:120]}")
print(f"  review: {result['review'][:80]}")
print("\\n💡 这就是 Skill 的价值：5 天的工程整合，别人一行 import 就能用。")
"""),

        # Checkpoint 4
        make_code('''# ============================================================
# Checkpoint 4 | Skill 收尾验证 (基础 + 进阶)
# ============================================================
#
# 【基础】（人人必做，5 min）
#   实现 capstone_skill_health_check()：
#   - validate capstone_assistant 文件夹
#   - 确认 pipeline.py 可被 import
#   - 确认 SKILL.md 含 enterprise-knowledge-assistant name
#
# 【进阶】（技术学员选做，10 min）
#   写一个改进版 description，让 capstone-assistant 在『内部知识查询』场景被路由概率更高，
#   在『闲聊』场景概率更低。
#   - 改 SKILL.md description 字段（in-place 写回文件）
#   - 用 audit_routing 跑前后对比
# ============================================================

def capstone_skill_health_check():
    """【基础】校验 capstone_assistant skill 健康度"""
    # ↓↓↓ 【基础】填空（约 8 行）↓↓↓
    skill_dir = "skills_demo/capstone_assistant"
    val = validate_skill(skill_dir)
    fm, body = parse_skill_md(f"{skill_dir}/SKILL.md")
    return {
        "validate": val,
        "name": fm.get("name"),
        "description_len": len(fm.get("description", "")),
        "body_len": len(body),
        "has_pipeline": Path(f"{skill_dir}/pipeline.py").exists(),
    }
    # ↑↑↑ 【基础】结束 ↑↑↑


def improve_capstone_description():
    """【进阶】改 description 让路由更准；返回 before/after audit"""
    # ↓↓↓ 【进阶】填空（约 22 行）↓↓↓
    from skills_helpers import discover_skills, match_skill_for_query
    skill_path = Path("skills_demo/capstone_assistant/SKILL.md")
    original_text = skill_path.read_text(encoding="utf-8")

    # 测试用例
    cases = [
        ("入职 5 年年假几天", "enterprise-knowledge-assistant"),  # 应路由到这
        ("查 ORD-XXX 订单", "enterprise-knowledge-assistant"),    # 应路由到这
        ("review my python code", "code-review"),                  # 不应路由到这
        ("发个通知给 alice", "db-query"),                          # 不应路由到这（注：db_query 也含 send_notification）
    ]

    def audit():
        skills = discover_skills("skills_demo")
        correct = 0
        log = []
        for q, expected in cases:
            picked = match_skill_for_query(q, skills, llm)
            picked_name = picked.name if picked else None
            ok = picked_name == expected
            log.append({"q": q, "expected": expected, "picked": picked_name, "ok": ok})
            if ok:
                correct += 1
        return correct / len(cases), log

    before_acc, before_log = audit()

    # 改 description（更窄、更具体）
    new_desc = "Answers internal company questions about HR policy, technical APIs, product info, and routes order/inventory queries to MCP tools. Use when the user asks specifically about company internals (policy / products / orders / API rate limits). Do NOT use for code review, generic chitchat, or external information."
    new_text = original_text
    # 简单替换 description 行（YAML 单行）
    import re as _re
    new_text = _re.sub(
        r"^description:.*$",
        f"description: {new_desc}",
        new_text,
        count=1,
        flags=_re.MULTILINE,
    )
    skill_path.write_text(new_text, encoding="utf-8")

    after_acc, after_log = audit()

    # 还原
    skill_path.write_text(original_text, encoding="utf-8")

    return {
        "before_accuracy": before_acc,
        "after_accuracy": after_acc,
        "before_log": before_log,
        "after_log": after_log,
        "new_description": new_desc[:120] + "...",
    }
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】capstone_skill_health_check"); print("=" * 56)
    try:
        health = capstone_skill_health_check()
        print(f"  validate: ok={health['validate']['ok']}")
        print(f"  name: {health['name']}")
        print(f"  desc 长度: {health['description_len']} 字符")
        print(f"  body 长度: {health['body_len']} 字符")
        print(f"  has pipeline.py: {health['has_pipeline']}")
        assert health["validate"]["ok"]
        assert health["name"] == "enterprise-knowledge-assistant"
        assert health["has_pipeline"]
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】improve_capstone_description (前后路由对比)"); print("=" * 56)
    try:
        result = improve_capstone_description()
        print(f"  Before accuracy: {result['before_accuracy']:.0%}")
        print(f"  After  accuracy: {result['after_accuracy']:.0%}")
        print(f"\\n  改进版 description: {result['new_description']}")
        print(f"\\n  Before vs After 详细:")
        for b, a in zip(result["before_log"], result["after_log"]):
            change = ""
            if b["picked"] != a["picked"]:
                change = f" (路由从 {b['picked']} → {a['picked']})"
            print(f"    '{b['q'][:30]}': {'✓' if b['ok'] else '✗'} → {'✓' if a['ok'] else '✗'}{change}")
        print("\\n  💡 description 是 Skill 路由的唯一线索 — 写好它是 Skills 工程的核心活")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

# Local imports for the cell
from skills_helpers import parse_skill_md  # type: ignore

verify()
'''),
    ]

    # Tag the new exercise cell
    add_tag(new_cells[-1], "fillin")
    add_tag(new_cells[-1], "batch5")

    # Insert before summary
    cells[summary_idx:summary_idx] = new_cells

    # Update 总结 cell to mention Skill packaging
    if summary_idx + len(new_cells) < len(cells):
        summary_cell = cells[summary_idx + len(new_cells)]
        if "Day 5 下午 总结" in cell_source(summary_cell):
            new_summary = """---

## Day 5 下午 总结

### 你学到了
1. **LLMOps 3 支柱**：trace / metric / alert
2. **Langfuse @observe + span**：一行装饰器搞定可观测
3. **trace_id 跨 Agent 传递**：父子 span 让你看『哪个 Agent 哪步慢』
4. **升级 Capstone**：5 天合体——Multi-Agent + MCP + Agentic RAG + LLMOps
5. **Capstone → Skill 打包**：把工程整合封成一个文件夹，团队复用
6. **评测 + 反思**：用 eval dataset 量化质量、用 trace 找性能瓶颈、用 description 优化路由

### 5 天回顾

| Day | 主题 |
|---|---|
| 1 | 文本→向量 + Transformer |
| 2 | 预训练 + SFT + LoRA + DPO + 评测 |
| 3 | RAG + Agent + 小 Capstone |
| 4 | Multi-Agent + **MCP 协议 + Skills 协议** |
| 5 | Agentic RAG + LLMOps + **升级 Capstone (打包成 Skill)** |

### 下一步建议（出培训后）

1. **回公司挑 1 个真实场景**做迷你 Capstone，最后**封成一个 Skill** 给团队
2. **接入真 Langfuse / OTEL** 看 trace 优化
3. **用 MCP** 把内部 API 暴露给 Claude / 团队工具
4. **2026 还可以学**：Vision LLM / Voice Agent / Reasoning Models / Computer Use / Safety
5. **持续读论文**：Anthropic / OpenAI cookbook，LangChain / LlamaIndex production guide

### 复盘环节

每位学员说一个『下周回公司能用上的一个改造点』 → 5 天的最大收获 ✓
"""
            set_cell_source(summary_cell, new_summary)

    save_nb(nb, PATH)
    print(f"✓ Patched {PATH}")
    print(f"  Total cells: {len(cells)} (added {len(new_cells)})")
    n_fill = sum(
        1 for c in cells
        if c["cell_type"] == "code"
        and "fillin" in c.get("metadata", {}).get("tags", [])
    )
    print(f"  Total fillin: {n_fill}")


if __name__ == "__main__":
    main()
