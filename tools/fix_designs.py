"""Phase 3 design fixes — 5 targeted patches across 4 notebooks.

1. Day4 上午 ex 4 dual_agent_with_failover — primary 真注入失败
2. Day5 上午 CRAG threshold 调 + demo case 选更难的
3. Day4 下午 实操 2 — naive vs skill 换更尖锐对比 (跑 3 次看格式漂移)
4. Day4 下午 ex 3 score_skill_description — 强制评分谱度
5. Day5 下午 Capstone tokens 自动估算 (改 utils/observability.py)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_lib import load_nb, save_nb, cell_source, set_cell_source


# ============================================================
# Fix 5 first: utils/observability.py — auto token tracking in @observe
# ============================================================
def fix_observability_tokens():
    path = Path("assets/enterprise_5days/utils/observability.py")
    text = path.read_text(encoding="utf-8")
    old_decorator = '''def observe(name: Optional[str] = None) -> Callable:
    """Decorator to auto-trace a function call."""
    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if _BACKEND == "langfuse" and _LANGFUSE_CLIENT is not None:
                # Use real Langfuse — minimal usage
                trace = _LANGFUSE_CLIENT.trace(name=span_name, input={"args": str(args)[:500], "kwargs": str(kwargs)[:500]})
                try:
                    out = fn(*args, **kwargs)
                    trace.update(output=str(out)[:500])
                    return out
                except Exception as e:
                    trace.update(level="ERROR", status_message=str(e)[:200])
                    raise
            else:
                span = observer.start_span(span_name, input={"args": str(args)[:200], "kwargs": str(kwargs)[:200]})
                try:
                    result = fn(*args, **kwargs)
                    observer.end_span(span, output=str(result)[:300])
                    return result
                except Exception as e:
                    observer.end_span(span, output=f"[ERROR] {e}", error=str(e))
                    raise
        return wrapper
    return decorator'''

    new_decorator = '''def observe(name: Optional[str] = None) -> Callable:
    """Decorator to auto-trace a function call.

    Auto-estimates token usage from string lengths of args/result (rough but
    consistent — divide-by-4 is a common ASCII heuristic; Chinese closer to /2,
    so we use /3 as compromise). For exact counts integrate tiktoken in prod.
    """
    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            args_str = " ".join(str(a) for a in args) + " " + " ".join(f"{k}={v}" for k, v in kwargs.items())
            input_tokens = len(args_str) // 3  # rough estimate (Chinese-friendly)
            if _BACKEND == "langfuse" and _LANGFUSE_CLIENT is not None:
                trace = _LANGFUSE_CLIENT.trace(name=span_name, input={"args": str(args)[:500], "kwargs": str(kwargs)[:500]})
                try:
                    out = fn(*args, **kwargs)
                    output_tokens = len(str(out)) // 3
                    trace.update(output=str(out)[:500])
                    # Real Langfuse usage tracking would go here via trace.usage
                    return out
                except Exception as e:
                    trace.update(level="ERROR", status_message=str(e)[:200])
                    raise
            else:
                span = observer.start_span(span_name, input={"args": str(args)[:200], "kwargs": str(kwargs)[:200]})
                try:
                    result = fn(*args, **kwargs)
                    output_tokens = len(str(result)) // 3
                    # Auto-record tokens (Chinese ≈ 3 chars/token rough)
                    observer.record_tokens(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        cost_usd=(input_tokens + output_tokens) * 1e-6,  # mock pricing
                    )
                    observer.end_span(span, output=str(result)[:300], tokens=input_tokens + output_tokens)
                    return result
                except Exception as e:
                    observer.end_span(span, output=f"[ERROR] {e}", error=str(e))
                    raise
        return wrapper
    return decorator'''

    if old_decorator in text:
        text = text.replace(old_decorator, new_decorator)
        path.write_text(text, encoding="utf-8")
        print("  ✓ observability.py: @observe now auto-tracks tokens")
    else:
        print("  ⚠ observability.py: pattern not found, skipping")


# ============================================================
# Fix 1: Day4 上午 ex 4 dual_agent_with_failover — primary 真注入失败
# ============================================================
def fix_day4am_failover():
    path = Path("assets/enterprise_5days/instructor/Day4_上午_Multi-Agent协作.ipynb")
    nb = load_nb(path)
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = cell_source(c)
        if "dual_agent_with_failover" not in src or "TechExpert" not in src:
            continue
        # Replace the verify() block to inject a guaranteed-failing primary
        if "result = dual_agent_with_failover(" in src and "primary_agent_name=\"TechExpert\"" in src:
            old_verify = """    print("=" * 56); print("【进阶】dual_agent_with_failover"); print("=" * 56)
    try:
        # 用 Triage + TechExpert 当主备 (实际场景：主用 qwen-plus，备用 qwen-turbo)
        result = dual_agent_with_failover(
            query="API 怎么用？",
            primary_agent_name="TechExpert",
            backup_agent_name="Triage",
            agents_dict=agents_handoff,
        )
        print(f"  使用 agent: {result['used_agent']}  |  是否切换: {result['switched']}")
        print(f"  回复 (前 100 字): {result['result'][:100]}")
        assert "result" in result and "used_agent" in result
        print("✅ 进阶通过")"""
            new_verify = """    print("=" * 56); print("【进阶】dual_agent_with_failover (强制注入 primary 失败)"); print("=" * 56)
    try:
        # 真演示 failover：构造一个必失败的 primary agent
        from utils.multi_agent import BaseAgent as _BA
        class AlwaysFailAgent(_BA):
            def receive(self, msg):
                raise RuntimeError("Primary agent simulated failure (timeout/rate-limit)")
        broken_primary = AlwaysFailAgent("BrokenPrimary", llm, "无所谓 prompt")
        agents_with_broken = {**agents_handoff, "BrokenPrimary": broken_primary}
        result = dual_agent_with_failover(
            query="API 怎么用？",
            primary_agent_name="BrokenPrimary",  # 这个一定失败
            backup_agent_name="TechExpert",       # 真 LLM 备用
            agents_dict=agents_with_broken,
            max_failures=2,
        )
        print(f"  使用 agent: {result['used_agent']}  |  是否切换: {result['switched']}")
        print(f"  回复 (前 100 字): {result['result'][:100]}")
        assert result['switched'] == True, "应该切到 backup"
        assert result['used_agent'] == "TechExpert", "backup 应该是 TechExpert"
        print("✅ 进阶通过 — primary 失败被检测到，自动切到 backup")"""
            new_src = src.replace(old_verify, new_verify)
            if new_src != src:
                set_cell_source(c, new_src)
                print("  ✓ Day4_上午 ex 4: 注入 BrokenPrimary 真演示 failover")
                break
    save_nb(nb, path)


# ============================================================
# Fix 2: Day5 上午 CRAG threshold + demo case
# ============================================================
def fix_day5am_crag():
    path = Path("assets/enterprise_5days/instructor/Day5_上午_Agentic_RAG.ipynb")
    nb = load_nb(path)
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = cell_source(c)
        if "def crag(" not in src:
            continue
        # Tune thresholds: high 0.55 (was 0.7), low 0.25 (was 0.3) — demo cases will hit better tiers
        new_src = src.replace("def crag(query, k=3, high=0.7, low=0.3):",
                              "def crag(query, k=3, high=0.55, low=0.25):")
        # Replace demo queries with ones that show all 3 tiers more clearly
        new_src = new_src.replace(
            'for q in ["入职 5 年有几天年假?", "API 限流是多少?", "公司有没有团建活动?"]:',
            'for q in ["StarLink 基础版价格", "API 限流是多少", "员工健身房会员怎么报销"]:',
        )
        if new_src != src:
            set_cell_source(c, new_src)
            print("  ✓ Day5_上午 CRAG: 阈值调到 0.55/0.25, demo cases 换更分明")
            break
    save_nb(nb, path)


# ============================================================
# Fix 3: Day4 下午 naive vs skill 换更尖锐对比（跑 3 次看格式稳定性）
# ============================================================
def fix_day4pm_naive_vs_skill():
    path = Path("assets/enterprise_5days/instructor/Day4_下午_MCP与Skills.ipynb")
    nb = load_nb(path)
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = cell_source(c)
        if "实操 2: 对比" not in src or "naive_report" not in src:
            continue
        # Replace the entire 实操 2 with a sharper comparison
        new_src = '''# 实操 2: 对比 — naive 跑 3 次看格式漂移 vs skill 跑 3 次看格式稳定
# 关键洞察：单次跑 naive 也能给好答案；但**多次跑**naive 输出格式漂移大，
# 不能给下游系统消费。Skill 强制走 SKILL.md workflow 输出可重复。

def has_critical_section(text):
    """检查是否含『阻塞级问题』标识（任何形式）"""
    return any(m in text for m in ["🔴", "Blocking", "Critical", "严重", "BLOCKER"])

def has_should_fix(text):
    return any(m in text for m in ["🟡", "Should fix", "应改", "SHOULD"])

def has_nice(text):
    return any(m in text for m in ["🟢", "Nice", "建议", "NIT"])

def count_bullet_sections(text):
    """大致数 markdown 主标题数量"""
    return sum(1 for line in text.split("\\n") if line.strip().startswith(("##", "###")))


print("=" * 70)
print("跑 3 次同代码 — 对比 naive 与 skill-driven 的格式稳定性")
print("=" * 70)

naive_results = []
skill_results = []
for i in range(3):
    print(f"\\n第 {i+1} 次...")
    # naive
    n = llm.generate(f"Review this code:\\n```python\\n{buggy_code}\\n```", temperature=0.3)
    naive_results.append(n)
    # skill (复用上面 Step 3 的 review_prompt)
    s = llm.generate(review_prompt, temperature=0.3)
    skill_results.append(s)

# 量化指标
print("\\n" + "=" * 70)
print(f"{'指标':<30} {'naive 3 次':<15} {'skill 3 次':<15}")
print("=" * 70)
metrics = [
    ("含『🔴 Blocking』标识", has_critical_section),
    ("含『🟡 Should fix』标识", has_should_fix),
    ("含『🟢 Nice』标识", has_nice),
]
for label, fn in metrics:
    n_n = sum(fn(r) for r in naive_results)
    n_s = sum(fn(r) for r in skill_results)
    print(f"  {label:<28} {n_n:>3}/3 次          {n_s:>3}/3 次")

# 长度方差（格式稳定性的代理指标）
import statistics
naive_lens = [len(r) for r in naive_results]
skill_lens = [len(r) for r in skill_results]
print(f"  长度均值                       {statistics.mean(naive_lens):>6.0f}        {statistics.mean(skill_lens):>6.0f}")
print(f"  长度标准差（越小越稳定）       {statistics.stdev(naive_lens):>6.0f}        {statistics.stdev(skill_lens):>6.0f}")
print(f"  章节数（## 标题）均值          {statistics.mean([count_bullet_sections(r) for r in naive_results]):>6.1f}        {statistics.mean([count_bullet_sections(r) for r in skill_results]):>6.1f}")

print("""
💡 关键观察:
  - 单次看 naive 也写得不错——但同样问题跑 3 次，格式漂移大（章节数 / 标题命名 / 三档分类不稳）
  - skill-driven 强制走 SKILL.md 的 workflow 第 4 步 Output format → 三档分类 + 章节稳定
  - 工业场景下下游系统（dashboard / 自动化流转）需要稳定的结构化输出 → skill 才靠谱
"""[1:])
'''
        if new_src != src:
            set_cell_source(c, new_src)
            print("  ✓ Day4_下午 实操 2: 改为 3 次跑统计格式稳定性对比")
            break
    save_nb(nb, path)


# ============================================================
# Fix 4: Day4 下午 score_skill_description 强谱度
# ============================================================
def fix_day4pm_score_skill():
    path = Path("assets/enterprise_5days/instructor/Day4_下午_MCP与Skills.ipynb")
    nb = load_nb(path)
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = cell_source(c)
        if "def score_skill_description" not in src:
            continue
        # Strengthen the prompt to force broader scoring
        old_prompt = """    prompt = f\"\"\"一个好的 Skill description 应该：
1. 一句话清楚说『何时用』(明确的触发条件)
2. 包含动词 + 对象 (做什么)
3. 长度 30-200 字
4. 让 Claude 一眼能与其他 skill 区分

下面这个 description 你给几分 (1-5)？再给一句改进建议。
description: {desc}

输出格式：
SCORE: <1-5>
SUGGESTION: <一句话>\"\"\""""
        new_prompt = """    prompt = f\"\"\"评分一个 Skill description 的质量 1-5 分。**严格使用全谱**：
- **5 分**：完美 — 清楚触发条件 + 动词宾语 + 30-200 字 + 与其他 skill 明显区分
- **4 分**：良 — 满足 3 项
- **3 分**：及格 — 满足 2 项
- **2 分**：差 — 只满足 1 项（如缺触发条件 / 太短 / 太模糊）
- **1 分**：极差 — 几乎不可用（< 10 字 / 完全模糊 / 无具体动词）

绝对禁止全部 3 分中庸。差的就给 1-2，好的就给 4-5。

description: {desc}

输出格式（严格 2 行）：
SCORE: <数字>
SUGGESTION: <一句话改进建议>\"\"\""""
        if old_prompt in src:
            new_src = src.replace(old_prompt, new_prompt)
            set_cell_source(c, new_src)
            print("  ✓ Day4_下午 ex 3: score prompt 强制全谱，禁止中庸")
            break
    save_nb(nb, path)


def main():
    print("Phase 3 设计修复:")
    print()
    fix_observability_tokens()
    fix_day4am_failover()
    fix_day5am_crag()
    fix_day4pm_naive_vs_skill()
    fix_day4pm_score_skill()
    print("\n✓ Phase 3 done")


if __name__ == "__main__":
    main()
