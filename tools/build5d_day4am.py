"""Build Day 4 上午 · Multi-Agent 协作 notebook for enterprise_5days.

Run from repo root:
    python tools/build5d_day4am.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_lib import (
    save_nb, make_md, make_code,
    make_path_fix_cell, make_lecture_note,
)

OUTPUT = Path("assets/enterprise_5days/instructor/Day4_上午_Multi-Agent协作.ipynb")


def build():
    cells = []

    # ── Title + 学习目标 ──
    cells.append(make_md("""# Day 4 上午 · Multi-Agent 协作（3h）

## 学习目标

1. 理解**单 Agent 的天花板**——为什么 ReAct + Tool 还不够
2. 掌握 3 大协作模式：**Hierarchical / Debate / Handoff**
3. 学会**任务分解 + 状态管理**：让多 Agent 系统不混乱
4. 实现**失败处理**：Circuit Breaker / 重试 / 降级
5. Mini-Project：3 Agent (PM / 工程师 / 测试) 协作写需求文档

## 前置

- Day 3 上午的 ReAct Agent 与 Tool 调用基础
- Day 3 下午的小 Capstone（理解单 Agent + RAG 的局限）
"""))

    # ── 📋 讲课提示 ──
    cells.append(make_lecture_note(
        title="""Day 4 上午 · Multi-Agent 协作（3h）""",
        duration_min=180,
        opener="""问：『一个聪明人能做完一切吗？为什么大公司要分团队？』 → 引出多 Agent 的必要性：分工 + 互查 + 平行。""",
        key_points=[
            """**单 Agent 三大瓶颈**：错误累积 / 上下文过载 / 专业模糊""",
            """**Hierarchical** = 公司层级（Planner 拆解 → Worker 执行 → Reviewer 把关）""",
            """**Debate** = 多人投标（多 Agent 各给方案，Judge 选最优；适合主观判断）""",
            """**Handoff** = 客服转接（A 处理不了显式 transfer 给 B；OpenAI Swarm 模式）""",
            """生产场景**必加**：trace 跟踪 / 失败重试 / Circuit Breaker / Token 预算""",
        ],
        misconceptions=[
            """学员以为多 Agent = 多个 ReAct 串联 → 强调『协议 + 状态』才是核心""",
            """学员以为 Hierarchical 一定优于 Debate → 强调主观/创意题 Debate 更稳""",
        ],
        interaction="""让学员说自己公司一个『需要多角色协作』的真实流程，现场拆成 3 Agent 设计。""",
        if_short_on_time="""跳过 CircuitBreaker 进阶（练习 4），保留 3 个协作模式 + Mini-Project 即可。""",
    ))

    # ── 路径就位 ──
    cells.append(make_path_fix_cell())

    # ── 导入 ──
    cells.append(make_code("""# 导入：LLM 后端 + 多 Agent 调度器
from utils.config import env
from utils.multi_agent import (
    Message, MessageType, BaseAgent, Orchestrator, CircuitBreaker,
)

llm = env.get_llm()
print(f"✓ LLM 就位：{type(llm).__name__}")
"""))

    # ── Part 1: 为什么单 Agent 不够 ──
    cells.append(make_md("""---

## Part 1 · 为什么单 Agent 不够（15 min）

### 三大瓶颈

| 瓶颈 | 表现 | 案例 |
|---|---|---|
| **错误累积** | ReAct 越长，每步错误概率累乘 | 10 步链每步 95% 正确 → 整体 60% |
| **上下文过载** | 长任务塞爆 prompt，关键信息被淹没 | 客服+技术+合规一起塞，模型抓不住重点 |
| **专业模糊** | 一个 Agent 既要写代码又要审代码 → 自审通常不严 | 让 LLM 同时『生成』和『验证』效果差 |

### 多 Agent 的核心思想

**分工 + 互查 + 协议**。每个 Agent 有：
- 一个**清晰职责**（system prompt）
- 一个**工具子集**（限制能干啥）
- 与其他 Agent 的**消息协议**（结构化通信）

### 单 Agent vs Multi-Agent 对比演示
"""))

    cells.append(make_code("""# 单 Agent 试图同时做 4 件事：拆需求 → 写代码 → 审代码 → 测试
single_agent_prompt = '''你需要完成以下任务，所有步骤一个人做完：
1. 把『一个 Python 函数计算两个日期之间相差的工作日数（跳过周末）』拆成步骤
2. 写出代码
3. 审查代码找问题
4. 写 3 个 test case

请输出全部 4 个步骤的结果。'''

print("=" * 60)
print("单 Agent 输出（一锅端）")
print("=" * 60)
single_out = llm.generate(single_agent_prompt, temperature=0.3)
print(single_out[:1500])
print(f"\\n... 总长 {len(single_out)} 字符")
"""))

    cells.append(make_md("""上面的输出会有几个典型问题：
- **审查不严**：Agent 自己写完自己审，几乎不会找出问题
- **测试覆盖不足**：和代码出自同一思维，盲点重合
- **格式混乱**：4 件事挤在一段，回头不好引用

下面我们用 **Hierarchical 多 Agent** 模式重做同一题。
"""))

    # ── Pattern 1: Hierarchical ──
    cells.append(make_md("""---

## Part 2 · 协作模式 1：Hierarchical（公司层级）

```
        Planner
       /   |   \\
   Worker1 Worker2 Worker3   (并行执行)
       \\   |   /
        Reviewer  ← 不通过则回流到 Planner 改进
```

**适用场景**：任务可拆解成清晰子任务（写文档 / 写代码 / 数据分析报告）。

**关键设计**：
- Planner 不执行细节，只产出『计划』
- Worker 拿到子任务**不与其他 Worker 通信**（避免乱）
- Reviewer 是**独立第三方**，不参与执行
- Reviewer 拒绝 → Planner 收到反馈重做（最多 N 次）
"""))

    cells.append(make_code("""# 定义 3 类专业 Agent
PLANNER_PROMPT = '''你是一个 **任务规划师 (Planner)**。
拿到任务后，把它拆成 2-4 个**清晰可独立完成**的子任务，每个子任务一行。
只输出子任务列表，不要解释。

格式：
1. <子任务 1>
2. <子任务 2>
...'''

WORKER_PROMPT = '''你是一个 **执行者 (Worker)**。
拿到一个具体子任务，给出简洁的解决方案（代码 / 文本 / 分析）。
不要超出子任务范围。'''

REVIEWER_PROMPT = '''你是一个 **质量审查者 (Reviewer)**。
检查所有 Worker 的输出，判断整体质量。

如果通过：第一行写『APPROVE』，后面写简短理由。
如果不通过：第一行写『REJECT』，后面列出 1-3 个改进点。

严格但建设性。'''

planner = BaseAgent("Planner", llm, PLANNER_PROMPT, temperature=0.2)
worker_a = BaseAgent("WorkerA", llm, WORKER_PROMPT, temperature=0.4)
worker_b = BaseAgent("WorkerB", llm, WORKER_PROMPT, temperature=0.4)
reviewer = BaseAgent("Reviewer", llm, REVIEWER_PROMPT, temperature=0.1)

agents = {a.name: a for a in [planner, worker_a, worker_b, reviewer]}
orch = Orchestrator(agents, trace=lambda m: print(f"  📨 {m}"))

print("✓ 4 Agent 就位 (Planner / WorkerA / WorkerB / Reviewer)")
"""))

    cells.append(make_code("""# 演示：Hierarchical 模式跑同一题（写工作日相差函数）
task = "写一个 Python 函数，计算两个日期之间相差的工作日数（跳过周末）。要求：含函数实现 + 至少 2 个测试用例。"

print("=" * 60)
print("Hierarchical 模式")
print("=" * 60)
result = orch.run_hierarchical(
    task=task,
    planner="Planner",
    workers=["WorkerA", "WorkerB"],
    reviewer="Reviewer",
    max_revisions=1,
)
print("\\n" + "=" * 60)
print(f"📋 最终评审：{result['review']}")
print(f"🔄 修改轮数：{result['revisions']}")
print(f"📦 输出：\\n{result['final'][:800]}...")
"""))

    cells.append(make_md("""**对比观察：**
- 单 Agent 版本：1 次 LLM 调用，输出杂糅
- Hierarchical 版本：4-7 次 LLM 调用（贵 4x），但**每个 Agent 只做一件事**，Reviewer 是独立第三方
- 工业场景中，Hierarchical 的输出质量通常比单 Agent 高 30-50%（按人工评分）
"""))

    # ── Exercise 1: Hierarchical ──
    cells.append(make_code('''# ============================================================
# 练习 1 | Hierarchical with Reflux：Reviewer 拒绝后回流 Planner
# ============================================================
#
# 【基础】（人人必做，10 min）
#   实现 hierarchical_simple(task)：用 Planner + 2 Worker + Reviewer 跑一遍，max_revisions=0
#   提示：直接调 orch.run_hierarchical(...)
#
# 【进阶】（技术学员选做，15 min）
#   实现 hierarchical_with_reflux(task, max_attempts=3)：
#   - 每轮 Reviewer 拒绝 → 把反馈塞回 Planner，Planner 重新拆解
#   - 拒绝/通过都记录到 history list
#   - 返回 {final, attempts, history}
#   提示：Orchestrator.run_hierarchical 已经支持 max_revisions，但你要自己写循环看 history
# ============================================================

def hierarchical_simple(task):
    """【基础】单轮 Hierarchical"""
    # ↓↓↓ 【基础】填空（约 5 行）↓↓↓
    return orch.run_hierarchical(
        task=task,
        planner="Planner",
        workers=["WorkerA", "WorkerB"],
        reviewer="Reviewer",
        max_revisions=0,
    )
    # ↑↑↑ 【基础】结束 ↑↑↑


def hierarchical_with_reflux(task, max_attempts=3):
    """【进阶】Reviewer 拒绝后回流，记录每轮"""
    # ↓↓↓ 【进阶】填空（约 12 行）↓↓↓
    history = []
    current_task = task
    for attempt in range(max_attempts):
        result = orch.run_hierarchical(
            task=current_task,
            planner="Planner",
            workers=["WorkerA", "WorkerB"],
            reviewer="Reviewer",
            max_revisions=0,
        )
        history.append({"attempt": attempt + 1, "review": result["review"][:120]})
        if "approve" in result["review"].lower() or "通过" in result["review"]:
            return {"final": result["final"], "attempts": attempt + 1, "history": history, "status": "approved"}
        # 把反馈拼回 task 让 Planner 重做
        current_task = f"{task}\\n\\n[上轮 Reviewer 反馈]：{result['review'][:200]}"
    return {"final": result["final"], "attempts": max_attempts, "history": history, "status": "max_attempts"}
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】hierarchical_simple"); print("=" * 56)
    try:
        result = hierarchical_simple("写一个 Python 函数：判断字符串是否为回文（忽略大小写和空格）")
        assert "final" in result and len(result["final"]) > 50
        print(f"  最终输出长度: {len(result['final'])} 字符")
        print(f"  Reviewer 评语: {result['review'][:100]}")
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】hierarchical_with_reflux"); print("=" * 56)
    try:
        result = hierarchical_with_reflux("设计一个反爬虫系统的 3 层防御策略", max_attempts=2)
        print(f"  尝试轮数: {result['attempts']}/2")
        print(f"  状态: {result['status']}")
        for h in result["history"]:
            print(f"    Attempt {h['attempt']}: {h['review'][:80]}...")
        assert "history" in result and len(result["history"]) >= 1
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''))

    # ── Pattern 2: Debate ──
    cells.append(make_md("""---

## Part 3 · 协作模式 2：Debate（多人投标）

```
       问题
      /    \\
 DebaterA   DebaterB   (并行各给方案)
      \\    /
       Judge → 选最优 / 加权聚合
```

**适用场景**：主观题 / 创意题 / 决策题（不像 Hierarchical 那种可清晰拆解的执行题）。

**为什么 Debate 比单 Agent 更好**：
- 不同 temperature / system prompt 的 Debater 给**真实多样**的方案
- Judge 看到 ≥2 个方案对比，更容易识别哪个更好
- 类似人类『头脑风暴 + 选择』的流程

**进阶**：不只让 Judge 选 1 个，而是给每个方案打 confidence 分，加权聚合。
"""))

    cells.append(make_code("""# 定义 Debater + Judge
DEBATER_A_PROMPT = '''你是一个 **保守派分析师**。倾向于稳健、低风险的方案。
'''
DEBATER_B_PROMPT = '''你是一个 **激进派创新者**。倾向于大胆、高回报的方案。
'''
JUDGE_PROMPT = '''你是 **决策评审**。
看完所有方案后，选出 **最适合企业实际落地** 的那个，给出明确选择 + 一句理由。
格式：
最佳方案：<方案名/标号>
理由：<一句话>
'''

debater_a = BaseAgent("Conservative", llm, DEBATER_A_PROMPT, temperature=0.5)
debater_b = BaseAgent("Innovative", llm, DEBATER_B_PROMPT, temperature=0.9)
judge = BaseAgent("Judge", llm, JUDGE_PROMPT, temperature=0.1)

agents_debate = {a.name: a for a in [debater_a, debater_b, judge]}
orch_debate = Orchestrator(agents_debate, trace=lambda m: print(f"  📨 {m}"))

print("✓ Debate 团队就位 (Conservative / Innovative / Judge)")
"""))

    cells.append(make_code("""# Demo: 让两位辩手对同一商业问题给方案，Judge 选
print("=" * 60)
print("Debate 模式 - 商业决策题")
print("=" * 60)
question = "我们公司要上线 AI 客服。先用 RAG 还是先做 SFT 微调？给一个明确的选择 + 3 句理由。"
result = orch_debate.run_debate(
    question=question,
    debaters=["Conservative", "Innovative"],
    judge="Judge",
)
print("\\n" + "=" * 60)
for d, ans in result["answers"].items():
    print(f"\\n【{d}】\\n{ans[:300]}\\n")
print(f"\\n📋 Judge 决定：\\n{result['judgment']}")
"""))

    # ── Exercise 2: Debate ──
    cells.append(make_code('''# ============================================================
# 练习 2 | Debate + Confidence Weighted Judge
# ============================================================
#
# 【基础】（人人必做，10 min）
#   实现 simple_debate(question)：2 Debater + 1 Judge，调 orch_debate.run_debate
#
# 【进阶】（技术学员选做，15 min）
#   实现 weighted_debate(question, n_judges=3)：
#   - 同一 Judge prompt，跑 3 次（temperature=0 但 model 有微小差异）
#   - 统计每个 debater 被选次数，多数票决定
#   - 同时返回 confidence = 多数票占比 (e.g. 2/3 = 0.67)
# ============================================================

def simple_debate(question):
    """【基础】单次 Debate"""
    # ↓↓↓ 【基础】填空（约 5 行）↓↓↓
    return orch_debate.run_debate(
        question=question,
        debaters=["Conservative", "Innovative"],
        judge="Judge",
    )
    # ↑↑↑ 【基础】结束 ↑↑↑


def weighted_debate(question, n_judges=3):
    """【进阶】多 Judge 投票 + confidence"""
    # ↓↓↓ 【进阶】填空（约 14 行）↓↓↓
    from collections import Counter
    # 先跑一次拿到答案
    result = orch_debate.run_debate(question, ["Conservative", "Innovative"], "Judge")
    answers = result["answers"]
    # 多次让 Judge 投票
    votes = []
    for i in range(n_judges):
        ans_text = "\\n\\n".join(f"[{d}] {a}" for d, a in answers.items())
        judge_prompt = f"""问题：{question}

各方答案：
{ans_text}

请只输出最佳方案的发起人姓名（Conservative 或 Innovative），无需理由。"""
        vote_raw = llm.generate(judge_prompt, temperature=0.0).strip()
        # 简单匹配名字
        for name in answers.keys():
            if name.lower() in vote_raw.lower():
                votes.append(name); break
    counts = Counter(votes)
    if not counts:
        return {"winner": None, "confidence": 0.0, "votes": votes, "answers": answers}
    winner, n = counts.most_common(1)[0]
    return {"winner": winner, "confidence": n / n_judges, "votes": votes, "answers": answers}
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】simple_debate"); print("=" * 56)
    try:
        result = simple_debate("企业 AI 项目应该自研模型还是用 API？")
        assert "answers" in result and len(result["answers"]) == 2
        print(f"  收到 {len(result['answers'])} 个答案 + 1 个 judge 决定")
        print(f"  Judge: {result['judgment'][:120]}")
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】weighted_debate (3 judges)"); print("=" * 56)
    try:
        result = weighted_debate("RAG vs Fine-tuning 选哪个？", n_judges=3)
        print(f"  胜者: {result['winner']}  |  Confidence: {result['confidence']:.0%}")
        print(f"  各 judge 投票: {result['votes']}")
        assert result["winner"] in ["Conservative", "Innovative"] or result["winner"] is None
        assert 0 <= result["confidence"] <= 1
        print(f"\\n  💡 高 confidence (>0.8) 表示 judge 强烈一致；低 confidence 说明案例本身有争议")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''))

    # ── Pattern 3: Handoff ──
    cells.append(make_md("""---

## Part 4 · 协作模式 3：Handoff（OpenAI Swarm 模式）

```
用户 → AgentA：『你处理不了，我转给 AgentB』
        AgentA  ----[HANDOFF: AgentB]---->  AgentB → 处理完成
```

**适用场景**：客服场景、能力不重叠的专业 Agent（前线接待 → 技术专家 → 计费专家）。

**关键设计**：
- 每个 Agent 知道自己『不能』做什么
- 显式输出 `HANDOFF: <agent_name>` 触发转接
- Handoff 时**整个 context 跟着传**（不是只传 task）

**vs Hierarchical**：
- Hierarchical = 自上而下分派（同时多 Worker）
- Handoff = 序列转接（一次只一个 Agent 在干活）
"""))

    cells.append(make_code("""# 3 类专业 Agent，各自只能处理特定问题
TRIAGE_PROMPT = '''你是 **客服前台 (Triage)**。
判断用户问题类型：
- 技术问题 → HANDOFF: TechExpert
- 计费问题 → HANDOFF: BillingExpert
- 简单问候 → 直接回答

如果要 HANDOFF，回复格式：
HANDOFF: <AgentName>
（一句话说明为什么转给他）'''

TECH_PROMPT = '''你是 **技术专家**。只回答技术细节问题（API / 部署 / bug）。
如果是计费 → HANDOFF: BillingExpert
如果是闲聊 → HANDOFF: Triage'''

BILLING_PROMPT = '''你是 **计费专家**。只回答账单 / 价格 / 退款问题。
如果是技术 → HANDOFF: TechExpert'''

triage = BaseAgent("Triage", llm, TRIAGE_PROMPT, temperature=0.1)
tech = BaseAgent("TechExpert", llm, TECH_PROMPT, temperature=0.2)
billing = BaseAgent("BillingExpert", llm, BILLING_PROMPT, temperature=0.2)

agents_handoff = {a.name: a for a in [triage, tech, billing]}
orch_handoff = Orchestrator(agents_handoff, trace=lambda m: print(f"  📨 {m}"))

print("✓ Handoff 团队就位 (Triage / TechExpert / BillingExpert)")
"""))

    cells.append(make_code("""# Demo: 客户问技术问题，Triage 应转给 TechExpert
print("=" * 60)
print("Handoff 模式 - 客服转接")
print("=" * 60)
result = orch_handoff.run_handoff(
    task="你好，我们的 API 调用一直返回 429 错误，怎么办？",
    start_agent="Triage",
    max_hops=3,
)
print("\\n" + "=" * 60)
print(f"🔄 路径: {' → '.join(result['path'])}")
print(f"⏱  跳数: {result['hops']}")
print(f"💬 最终回复: {result['final'][:300]}")
"""))

    # ── Exercise 3: Handoff ──
    cells.append(make_code('''# ============================================================
# 练习 3 | Handoff with Capability Matching
# ============================================================
#
# 【基础】（人人必做，10 min）
#   实现 simple_handoff(query)：让 Triage 自动判断转给谁
#
# 【进阶】（技术学员选做，15 min）
#   实现 capability_router(query, agent_capabilities)：
#   - agent_capabilities = {"TechExpert": ["api", "部署", "bug"], ...}
#   - 不依赖 LLM，**用关键词匹配**预先选 agent
#   - 然后直接发给该 agent，省掉 Triage 一跳
#   - 返回 (selected_agent, response)
#   - 这是工业上『用规则路由 + LLM 兜底』的混合策略
# ============================================================

def simple_handoff(query):
    """【基础】用 Triage 起步"""
    # ↓↓↓ 【基础】填空（约 4 行）↓↓↓
    return orch_handoff.run_handoff(
        task=query,
        start_agent="Triage",
        max_hops=3,
    )
    # ↑↑↑ 【基础】结束 ↑↑↑


def capability_router(query, agent_capabilities):
    """【进阶】关键词路由跳过 Triage"""
    # ↓↓↓ 【进阶】填空（约 10 行）↓↓↓
    query_lower = query.lower()
    scores = {}
    for agent_name, keywords in agent_capabilities.items():
        scores[agent_name] = sum(1 for kw in keywords if kw.lower() in query_lower)
    if not scores or max(scores.values()) == 0:
        # 兜底回 Triage
        return ("Triage", orch_handoff.agents["Triage"].receive(
            Message("user", "Triage", MessageType.TASK, query)
        ).payload)
    selected = max(scores, key=scores.get)
    response = orch_handoff.agents[selected].receive(
        Message("user", selected, MessageType.TASK, query)
    ).payload
    return (selected, response)
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】simple_handoff"); print("=" * 56)
    try:
        # 测试技术问题
        result = simple_handoff("我的 API 怎么处理 rate limit？")
        assert "path" in result and len(result["path"]) >= 1
        print(f"  路径: {' → '.join(result['path'])}")
        print(f"  跳数: {result['hops']}")
        print(f"  最终: {result['final'][:120]}")
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】capability_router (规则路由)"); print("=" * 56)
    try:
        caps = {
            "TechExpert": ["api", "部署", "bug", "错误", "代码"],
            "BillingExpert": ["账单", "价格", "退款", "费用", "发票"],
        }
        # 技术问题
        agent1, resp1 = capability_router("API 调用 timeout 怎么办？", caps)
        # 计费问题
        agent2, resp2 = capability_router("我想要发票退款", caps)
        print(f"  '技术 query' → 路由到: {agent1}")
        print(f"  '计费 query' → 路由到: {agent2}")
        assert agent1 == "TechExpert"
        assert agent2 == "BillingExpert"
        print(f"\\n  💡 关键词路由比 LLM Triage 快 5x，准确率 80%+；剩下 20% 让 LLM 兜底即可")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''))

    # ── Part 5: 失败处理 ──
    cells.append(make_md("""---

## Part 5 · 失败处理：Circuit Breaker（30 min）

生产环境中 Agent 会遇到：
- LLM API timeout / rate limit / 5xx 错误
- Tool 调用失败（数据库 down、API 接口变化）
- 模型输出不可用（hallucination / format 错乱）

**Circuit Breaker** 是经典的容错模式（来自 Netflix Hystrix）：

| 状态 | 行为 |
|---|---|
| **Closed** | 正常调用，记录失败次数 |
| **Open** | 失败次数到阈值 → 直接拒绝调用，避免雪崩 |
| **Half-Open** | 等 reset_timeout → 试一次，成功则关闭，失败则继续 Open |

`utils/multi_agent.py` 已实现 `CircuitBreaker` 类。
"""))

    cells.append(make_code("""# Demo: CircuitBreaker 保护一个不稳定 Agent
import random

def flaky_llm_call(query):
    if random.random() < 0.7:  # 70% 失败率
        raise RuntimeError("LLM API timeout")
    return f"OK: {query}"

cb = CircuitBreaker(failure_threshold=3, reset_timeout_s=2.0)
print("=" * 60)
print("CircuitBreaker 演示 (failure_threshold=3, reset=2s)")
print("=" * 60)
random.seed(42)
for i in range(8):
    try:
        result = cb.call(flaky_llm_call, f"query_{i}")
        print(f"  [{i}] ✓ {result}  | state={cb.state}")
    except RuntimeError as e:
        print(f"  [{i}] ✗ {e}  | state={cb.state}")
import time as _t
print("\\n等 2 秒让 breaker 复位...")
_t.sleep(2.1)
try:
    result = cb.call(lambda q: f"OK after reset: {q}", "recovery_test")
    print(f"  Recovery: ✓ {result}  | state={cb.state}")
except RuntimeError as e:
    print(f"  Recovery: ✗ {e}  | state={cb.state}")
"""))

    # ── Exercise 4: Circuit Breaker ──
    cells.append(make_code('''# ============================================================
# 练习 4 | 失败注入 + Circuit Breaker + 备用 Agent 切换
# ============================================================
#
# 【基础】（人人必做，10 min）
#   实现 protected_call(unreliable_fn, *args)：用 CircuitBreaker 包装函数调用，捕获异常返回 None
#
# 【进阶】（技术学员选做，15 min）
#   实现 dual_agent_with_failover(query, primary_agent, backup_agent)：
#   - primary_agent 失败 N 次 → 自动切到 backup_agent
#   - 备用 Agent 也失败 → 返回 fallback message
#   - 返回 {result, used_agent, switched}
# ============================================================

def protected_call(unreliable_fn, *args, threshold=3):
    """【基础】包装函数 + 异常捕获"""
    # ↓↓↓ 【基础】填空（约 6 行）↓↓↓
    cb = CircuitBreaker(failure_threshold=threshold, reset_timeout_s=1.0)
    try:
        return cb.call(unreliable_fn, *args)
    except Exception as e:
        return None
    # ↑↑↑ 【基础】结束 ↑↑↑


def dual_agent_with_failover(query, primary_agent_name, backup_agent_name, agents_dict, max_failures=2):
    """【进阶】primary 失败自动切 backup"""
    # ↓↓↓ 【进阶】填空（约 16 行）↓↓↓
    cb = CircuitBreaker(failure_threshold=max_failures, reset_timeout_s=1.0)
    switched = False
    primary = agents_dict[primary_agent_name]
    backup = agents_dict[backup_agent_name]
    used = primary_agent_name
    try:
        msg = Message("user", primary_agent_name, MessageType.TASK, query)
        result = cb.call(primary.receive, msg)
        return {"result": result.payload, "used_agent": used, "switched": False}
    except Exception as primary_err:
        # 切到 backup
        switched = True
        used = backup_agent_name
        try:
            msg = Message("user", backup_agent_name, MessageType.TASK, query)
            result = backup.receive(msg)
            return {"result": result.payload, "used_agent": used, "switched": True}
        except Exception as backup_err:
            return {"result": "[FALLBACK] 所有 agent 都不可用，请稍后再试", "used_agent": "fallback", "switched": True}
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】protected_call"); print("=" * 56)
    try:
        # 跑一个 100% 成功的函数
        ok = protected_call(lambda x: x * 2, 21)
        assert ok == 42
        # 跑一个 100% 失败的函数
        def always_fail(x):
            raise RuntimeError("always fail")
        bad = protected_call(always_fail, 1, threshold=2)
        assert bad is None
        print(f"  ✓ 成功调用返回 {ok}")
        print(f"  ✓ 失败调用返回 None (而非崩溃)")
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】dual_agent_with_failover"); print("=" * 56)
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
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''))

    # ── Mini-Project ──
    cells.append(make_md("""---

## Mini-Project · 3 Agent 协作写需求文档（30 min）

**情景**：产品 PM 提一个需求，工程师分析可行性，QA 写测试要点。最终输出一份小型 PRD。

```
用户提需求
    ↓
PM Agent (拆需求 + 用户故事)
    ↓
EngineerAgent (技术方案 + 接口)
    ↓
QAAgent (测试要点 + 边界情况)
    ↓
组装 → 输出 PRD
```

这个 Mini-Project 是 Day 5 下午**升级 Capstone** 的预热——那时这 3 个 Agent 会作为基础组件之一。
"""))

    cells.append(make_code("""# 3 个专业角色
PM_PROMPT = '''你是 **产品经理 (PM)**。把用户需求拆成：
1. 用户故事 (1-2 个，As a... I want... So that...)
2. 验收标准 (3 条以内，可测量)
只输出这 2 部分。'''

ENG_PROMPT = '''你是 **资深工程师**。看 PM 给的用户故事和验收标准后输出：
1. 技术方案 (一段话, ≤ 100 字)
2. API 接口 (1 个，POST /xxx，含 request / response 字段)
只输出这 2 部分。'''

QA_PROMPT = '''你是 **QA 测试**。看完 PM + 工程师方案后输出：
1. 关键测试点 (3 条)
2. 边界情况 (2 条)
只输出这 2 部分。'''

pm = BaseAgent("PM", llm, PM_PROMPT, temperature=0.3)
eng = BaseAgent("Engineer", llm, ENG_PROMPT, temperature=0.2)
qa = BaseAgent("QA", llm, QA_PROMPT, temperature=0.2)


def run_prd_workflow(raw_requirement: str) -> str:
    \"\"\"3 Agent 顺序协作产出 PRD\"\"\"
    # PM 拆需求
    pm_msg = Message("user", "PM", MessageType.TASK, raw_requirement)
    pm_out = pm.receive(pm_msg).payload
    # Engineer 设计
    eng_msg = Message("PM", "Engineer", MessageType.TASK,
                       f"原需求：{raw_requirement}\\n\\nPM 拆解：{pm_out}")
    eng_out = eng.receive(eng_msg).payload
    # QA 测试
    qa_msg = Message("Engineer", "QA", MessageType.TASK,
                      f"原需求：{raw_requirement}\\n\\nPM:\\n{pm_out}\\n\\nEng:\\n{eng_out}")
    qa_out = qa.receive(qa_msg).payload
    # 组装
    prd = f\"\"\"# 产品需求文档 (PRD)

## 原始需求
{raw_requirement}

## 产品 (来自 PM Agent)
{pm_out}

## 技术方案 (来自 Engineer Agent)
{eng_out}

## 测试方案 (来自 QA Agent)
{qa_out}
\"\"\"
    return prd


# 演示：写一个『企业内部知识问答机器人』PRD
demo_req = "我们想给员工做一个内部知识问答机器人，能回答公司制度、HR 政策。手机微信里能用。"
print("=" * 60)
print("Mini-Project: 3 Agent 写 PRD")
print("=" * 60)
prd = run_prd_workflow(demo_req)
print(prd)
"""))

    # ── 总结 ──
    cells.append(make_md("""---

## Day 4 上午 总结

### 你学到了
1. **3 大协作模式**：Hierarchical（层级 + Reviewer 把关）/ Debate（多方案 + Judge 决定）/ Handoff（序列转接）
2. **失败处理**：Circuit Breaker 防雪崩 + 主备 Agent 切换
3. **状态管理**：Orchestrator 维护 transcript，每条消息可追溯

### 何时用哪个模式

| 场景 | 推荐模式 |
|---|---|
| 任务能拆成清晰子任务（写代码、写文档、做分析） | **Hierarchical** |
| 主观判断 / 创意 / 多方案权衡（决策、文案、设计） | **Debate** |
| 客服 / 多专家系统 / 能力不重叠的 Agent | **Handoff** |
| 单 Agent 够用、追求最低延迟 | 不要硬上 Multi-Agent |

### 与下午 MCP 的衔接

下午我们学 **MCP 协议**——把这些 Agent 调用的 tool（查订单、查库存、读文档）规范化为**跨厂可移植的协议**。今天 Mini-Project 里的 3 个 Agent，下午会用 MCP server 暴露统一工具集。

### 推荐资料
- OpenAI Swarm（Handoff 模式参考实现）
- LangGraph（Hierarchical + State 模式）
- AutoGen / CrewAI（更高层抽象）
"""))

    # ── Save ──
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
    print(f"✓ Built {OUTPUT}")
    print(f"  Total cells: {len(cells)}")
    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"  Markdown: {n_md} | Code: {n_code}")


if __name__ == "__main__":
    build()
