"""Apply v2 transformations to all 7 instructor notebooks.

Run from repo root:
    python tools/transform_v2.py

Convention: every natural-language string with potentially-embedded quotes
uses triple-double-quoted strings to avoid the nested-quote trap.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_lib import (
    load_nb, save_nb, cell_source, set_cell_source,
    make_md, make_code, add_tag,
    insert_after, insert_before, delete_cells,
    replace_in_source, replace_regex_in_source,
    make_path_fix_cell, make_lecture_note,
    autotag_exercises, scan_cost_cells,
)

INSTRUCTOR = Path("assets/enterprise_ver2/instructor")


# ============================================================ common pass
def normalize_paths(nb: dict) -> int:
    """Insert path-fix cell + rewrite font paths + drop legacy sys.path inserts."""
    has_path_fix = any(
        "课程根目录" in cell_source(c)
        for c in nb["cells"] if c["cell_type"] == "code"
    )
    if not has_path_fix:
        insert_idx = 0
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "markdown":
                insert_idx = i
                break
        insert_after(nb, insert_idx, make_path_fix_cell())

    n_changed = 0
    for c in nb["cells"]:
        if replace_regex_in_source(c, r'os\.path\.join\(\s*"\.\."\s*,\s*"fonts"', 'os.path.join("fonts"'):
            n_changed += 1
        if replace_in_source(c, "../fonts/", "fonts/"):
            n_changed += 1
        if replace_in_source(c, '"../fonts"', '"fonts"'):
            n_changed += 1

    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = cell_source(c)
        if 'sys.path.insert(0, "utils")' in src and "课程根目录" not in src:
            new_src = src
            for legacy in [
                'sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("__file__")), "utils"))\n',
                'sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("__file__")), "utils"))',
                'sys.path.insert(0, "utils")\n',
                'sys.path.insert(0, "utils")',
            ]:
                new_src = new_src.replace(legacy, "")
            set_cell_source(c, new_src)
            n_changed += 1

    return n_changed


# ============================================================ Day 0
def transform_day0(nb: dict) -> dict:
    normalize_paths(nb)

    note = make_lecture_note(
        title="""Day 0 · 环境配置（开班前自助 20 min）""",
        duration_min=20,
        opener="""开课前发出来让学员自己跑通即可。开课当天前 5 分钟现场抽查 1-2 位学员的 Step 5 LLM 测试 ✅。""",
        key_points=[
            """**conda activate llmc** —— 强调使用统一的 conda 环境，避免依赖打架""",
            """API Key 不要提交 git；推荐永久配环境变量（Step 4 末尾）""",
            """字体路径使用相对路径自动定位（学员从 instructor/ 或 student/ 都能跑）""",
        ],
        misconceptions=[
            """学员经常以为 `pip install` 在 base 环境就行 → 强调先 `conda activate llmc`""",
            """学员看到 ⚠️ GPU 不可用就放弃 → 强调 Day 1-2 大部分内容 CPU 也能跑""",
        ],
        interaction="""Step 5 跑通后，让学员把 LLM 回的第一句『自我介绍』贴在群里，即可作为签到。""",
        if_short_on_time="""若开课前未发，直接 5 分钟讲师演示一遍 Step 1→Step 7，告诉学员后续课中再补。""",
    )
    insert_after(nb, 0, note)

    # Add a conda llmc verification step between Step 1 and Step 2
    step1_idx = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code" and "Python 版本符合要求" in cell_source(c):
            step1_idx = i
            break
    if step1_idx is not None:
        llmc_md = make_md("""---
## Step 1.5：确认 conda 环境

本课程统一推荐使用 conda 环境 `llmc`（已配 PyTorch / transformers / peft / dashscope / sentence-transformers）。

如果你还没建过：
```bash
conda create -n llmc python=3.10 -y
conda activate llmc
pip install -r requirements.txt
```

**确认 Jupyter 用的是 llmc 内核：** Kernel 菜单 → Change Kernel → 选 `llmc`。

运行下面的 cell 自检：""")
        llmc_code = make_code('''import sys, platform
print(f"Python 解释器：{sys.executable}")
print(f"Python 版本：{platform.python_version()}")
is_isolated = "envs" in sys.executable or "llmc" in sys.executable.lower()
if is_isolated:
    print("✅ 看起来在 conda 隔离环境里（推荐 llmc）")
else:
    print("⚠️ 似乎在系统 Python 里跑。建议 `conda activate llmc` 后重启 Jupyter。")
    print("   不切环境也能跑，但容易遇到依赖版本冲突。")
''')
        insert_after(nb, step1_idx, llmc_md, llmc_code)

    # Update file structure diagram in Step 8
    for c in nb["cells"]:
        if c["cell_type"] != "markdown":
            continue
        src = cell_source(c)
        if "课程文件结构" in src and "enterprise/" in src and "enterprise_ver2" not in src:
            new_src = src.replace(
                "enterprise/                          ← 你现在在这里",
                "enterprise_ver2/                     ← 课程根目录\n├── instructor/                      ← 讲师版（你现在在这里）\n│   └── Day0~Day3_下午.ipynb\n├── student/                         ← 学员版（同 7 本，留填空）",
            )
            # Drop the legacy individual notebook lines
            for legacy in [
                "├── Day0_环境配置与测试.ipynb         ← 本文件\n",
                "├── Day1_上午_从文本到向量.ipynb\n",
                "├── Day1_下午_Transformer架构.ipynb\n",
                "├── Day2_上午_预训练与SFT.ipynb\n",
                "├── Day2_下午_LoRA对齐与评测.ipynb\n",
                "├── Day3_上午_RAG与Agent实战.ipynb\n",
                "├── Day3_下午_企业知识助手Capstone.ipynb\n",
            ]:
                new_src = new_src.replace(legacy, "")
            set_cell_source(c, new_src)
            break

    return nb


# ============================================================ Day 1 上午
def transform_day1_am(nb: dict) -> dict:
    normalize_paths(nb)

    # 删除 Token 成本计算器段
    indices_to_delete = []
    for i, c in enumerate(nb["cells"]):
        src = cell_source(c)
        if any(p in src for p in [
            "企业 Token 成本计算器", "企业Token成本计算器", "Token 成本计算器",
        ]):
            indices_to_delete.append(i)
            for j in range(i + 1, min(i + 5, len(nb["cells"]))):
                nxt = cell_source(nb["cells"][j])
                if any(k in nxt for k in [
                    "DAILY_QUERIES", "PRICE_PER_1K_TOKENS", "AVG_CHARS_PER_QUERY",
                    "monthly_savings", "月省", "节省¥",
                ]):
                    indices_to_delete.append(j)

    if indices_to_delete:
        first_idx = min(indices_to_delete)
        replacement = make_md("""---

> 💡 **小常识（已替代原 Token 单价估算练习）：** 中文文本约 1.3-1.6 个字符 / token（取决于 tokenizer），比英文略贵。选模型时记得用 `tiktoken` 或厂商工具实测自己语料的 token 数。**本课程不展开商业 ROI 估算**——把时间留给原理与实操。""")
        delete_cells(nb, indices_to_delete)
        insert_before(nb, first_idx, replacement)

    note = make_lecture_note(
        title="""Day 1 上午 · 从文本到向量（3h）""",
        duration_min=180,
        opener="""问：『如果只让你用 5 个数字描述今天的天气，你会怎么选？』 → 引出 embedding = 用一组数字表示语义。""",
        key_points=[
            """**训练循环 4 步是 LLM 一切的基础**：Forward → Loss → Backward → Step""",
            """Tokenizer 决定模型的『识字能力』；中文 token 比英文贵 (~1.5×)，但不算金额，强调比例感""",
            """Embedding 本质是『查表』(lookup)，不是矩阵乘法；用 `nn.Embedding` 演示""",
            """Prompt Engineering 是『用对话引导思考链』，不是『魔法咒语』""",
        ],
        misconceptions=[
            """学员把 nn.Embedding 当 nn.Linear → 用同样输入演示输出对比""",
            """Few-shot 不是越多越好 → 演示 0/2/8/16 shot 的边际收益递减""",
        ],
        interaction="""让 1-2 位学员当场给 1 个公司术语，现场跑 embedding + cosine 相似度查最相似的 3 个词。""",
        if_short_on_time="""跳过 PCA 可视化（最后 10 分钟）；prompt 部分只跑 zero/few-shot 对比，CoT 留作课后阅读。""",
    )
    insert_after(nb, 0, note)
    autotag_exercises(nb)
    return nb


# ============================================================ Day 1 下午
def transform_day1_pm(nb: dict) -> dict:
    normalize_paths(nb)
    note = make_lecture_note(
        title="""Day 1 下午 · Transformer 架构（3.5h）""",
        duration_min=210,
        opener="""问：『你在图书馆找一本书，怎么找？』 → 引出 QKV：Query 是问题，Key 是书脊标签，Value 是书内容。""",
        key_points=[
            """**Self-Attention = Q·K^T 算相似度 + softmax 归一 + 加权 V**，三步循环往复""",
            """Causal Mask 是 GPT 解码的灵魂：阻止偷看未来词""",
            """Multi-Head 不是更多参数，是**让模型同时学多种关系**（语法/语义/位置）""",
            """Residual + LayerNorm 是深网络能训起来的两个救命发明""",
            """采样策略（greedy / top-k / top-p / temperature）决定生成『风格』""",
        ],
        misconceptions=[
            """把 Multi-Head 想成『多次跑同一个 Attention』 → 强调每个 head 学的是不同的投影""",
            """把 LayerNorm 想成 BatchNorm → 强调归一化的维度不一样""",
        ],
        interaction="""用 5×5 注意力热图让学员肉眼判断『句子里每个词最关注哪个词』，引出 attention 可解释性。""",
        if_short_on_time="""跳过 LayerNorm 推导（4.2 节后半），直接用 PyTorch 自带 nn.LayerNorm；位置编码只讲 sinusoidal，不展开 RoPE。""",
    )
    insert_after(nb, 0, note)
    autotag_exercises(nb)
    return nb


# ============================================================ Day 2 上午
def transform_day2_am(nb: dict) -> dict:
    normalize_paths(nb)

    # Find SFT Part 3 start; cut it and stage for Day 2 PM
    sft_start = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "markdown" and "Part 3" in cell_source(c) and "SFT" in cell_source(c):
            sft_start = i
            break

    if sft_start is not None:
        sft_cells = nb["cells"][sft_start:]
        nb["cells"] = nb["cells"][:sft_start]
        nb["cells"].append(make_md("""---

## 上午小结

我们完成了**预训练完整闭环**：数据 → 词表 → 模型 → 训练循环 → 生成对比。

下午我们将基于今天上午的 Base Model 思路，进入 **SFT 指令微调** + **LoRA 高效微调** + **DPO 对齐** + **评测**。

**☕ 午休 1 小时**"""))
        staging = Path("tools/_day2_sft_staging.json")
        with open(staging, "w", encoding="utf-8") as f:
            json.dump(sft_cells, f, ensure_ascii=False, indent=1)
        print(f"  ↪ Day2_上午: 切出 {len(sft_cells)} 个 SFT cells → {staging}")

    note = make_lecture_note(
        title="""Day 2 上午 · 预训练完整闭环（3h）""",
        duration_min=180,
        opener="""问：『婴儿学说话靠什么？』 → 大量听 → 引出预训练 = 让模型大量『听』文本，学会『下一个字最可能是什么』。""",
        key_points=[
            """**预训练 = Next Token Prediction**，loss = CrossEntropy on shifted tokens""",
            """字符级 vs BPE：选择决定词表大小与上下文长度的 trade-off""",
            """训练前后生成对比 → 让学员**亲眼看见**模型从乱码到通顺的过程""",
            """Loss 曲线诊断：上升=学习率太大；震荡=batch 太小；过快收敛=过拟合""",
        ],
        misconceptions=[
            """学员以为预训练 = 教模型回答问题 → 强调 base model 不会聊天，只会续写""",
            """学员认为越大数据越好 → 强调质量 > 数量，垃圾进垃圾出""",
        ],
        interaction="""让学员把训练数据换成自己公司业务文本（10 行即可），跑 50 epoch 看生成结果是否带上业务术语。""",
        if_short_on_time="""保留预训练全流程，跳过『调超参』练习；下午会再讲一次完整训练。""",
    )
    insert_after(nb, 0, note)
    autotag_exercises(nb)
    return nb


# ============================================================ Day 2 下午
def transform_day2_pm(nb: dict) -> dict:
    normalize_paths(nb)

    # Delete cost calculator (cell ~52) and 商业决策分析 (Ex 6, ~54)
    indices_to_delete = []
    for i, c in enumerate(nb["cells"]):
        src = cell_source(c)
        if "成本计算器" in src and ("API 调用" in src or "自部署" in src):
            indices_to_delete.append(i)
        if "练习 6" in src and ("商业决策" in src or "商业 决策" in src):
            indices_to_delete.append(i)
            # Sweep all consecutive trailing <details> hint cells (有时连写 2-3 个)
            j = i + 1
            while j < len(nb["cells"]):
                nxt_src = cell_source(nb["cells"][j])
                if "<details>" in nxt_src or nxt_src.strip().startswith("<details>"):
                    indices_to_delete.append(j)
                    j += 1
                else:
                    break

    if indices_to_delete:
        delete_cells(nb, indices_to_delete)
        print(f"  ↪ Day2_下午: 删除 {len(indices_to_delete)} 个成本/商业话术 cells")

    # Receive SFT cells from Day 2 AM staging
    staging = Path("tools/_day2_sft_staging.json")
    if staging.exists():
        with open(staging, "r", encoding="utf-8") as f:
            sft_cells = json.load(f)
        receive_md = make_md("""---

## 块 1 · 承接上午 — SFT 动手实战（60 min）

上午我们完成了从零预训练（Base Model）。下午第一段进入 **SFT (Supervised Fine-Tuning) 指令微调**：教模型『听话』。

> 💡 **回顾：** Base Model 会『续写』，不会『对话』。SFT 用 ChatML 格式 + Loss Masking 教它在合适位置生成回复。""")
        insert_pos = 1
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code" and "课程根目录" in cell_source(c):
                insert_pos = i + 1
                break
        nb["cells"][insert_pos:insert_pos] = [receive_md] + sft_cells
        print(f"  ↪ Day2_下午: 插入 1 个 receive markdown + {len(sft_cells)} 个 SFT cells (来自上午)")
        staging.unlink()

    note = make_lecture_note(
        title="""Day 2 下午 · SFT 动手 + LoRA + DPO + 评测（3h）""",
        duration_min=180,
        opener="""问：『上午跑出来的模型像三岁小孩在乱说话；现在我们要教它当客服。』 → 引出 SFT。""",
        key_points=[
            """**SFT 不是预训练的延续，是『行为塑造』**：用 ChatML + Loss Masking 只惩罚回复部分""",
            """**LoRA 核心：** 冻结原权重 + 加低秩旁路 (BA)，可训参数降到 1% 以下""",
            """**rank 选择：** 小任务 r=4-8，复杂任务 r=16-32；不是越大越好""",
            """**DPO 跳过 RM：** 直接用偏好对优化策略 → 比 PPO 简单稳定""",
            """**评测三件套：** PPL（语言流畅度）+ 准确率（任务能力）+ LLM-as-Judge（主观质量）""",
        ],
        misconceptions=[
            """学员以为 LoRA 等于『训练得更少更差』 → 强调多数任务 LoRA 与全参微调几乎无差""",
            """学员以为 DPO 比 SFT 强 → 强调 DPO 是 SFT 之后的对齐步骤，不是替代""",
        ],
        interaction="""让学员观察 LoRA 训练时 GPU 显存占用 vs 全参微调 → 直观感受参数效率。""",
        if_short_on_time="""DPO 部分（块 3 前 30 min）只讲公式与一次推理对比，跳过完整训练；评测保留 PPL + 一个 Judge 例子。""",
    )
    insert_after(nb, 0, note)
    autotag_exercises(nb)
    return nb


# ============================================================ Day 3 上午
def transform_day3_am(nb: dict) -> dict:
    normalize_paths(nb)
    note = make_lecture_note(
        title="""Day 3 上午 · RAG + Agent 实战（3h）""",
        duration_min=180,
        opener="""问：『如果你公司明天上线 AI 客服，怎么让它知道你公司去年的销售数据？』 → 微调？太慢且贵。**RAG**：让它『现场查』。""",
        key_points=[
            """**RAG 三步：** 切块 → 向量检索 → 生成时把检索到的内容喂回给 LLM""",
            """**ReAct = Reasoning + Acting**：模型在 Thought→Action→Observation 循环中调工具""",
            """**Code Agent：** 让 LLM 写 Python 代码，沙箱执行后把结果当 Observation""",
            """**RAG 失败 4 种：** OOD、模糊查询、跨文档、对抗 prompt → 必须有 confidence 阈值兜底""",
        ],
        misconceptions=[
            """学员以为 RAG 一定比微调好 → 给出决策矩阵：知识更新频率 + 数据敏感度 + 任务复杂度""",
            """学员以为 Agent 是『更聪明的 chatbot』 → 强调 Agent = LLM + 工具 + 循环""",
        ],
        interaction="""让学员上传一份自己部门的 FAQ（5-10 条），现场跑通 RAG 问答 + 失败模式分析。""",
        if_short_on_time="""Code Agent 部分只演示一个例子（员工数据分析），跳过 mini-project；保留 RAG + ReAct 主线。""",
    )
    insert_after(nb, 0, note)
    autotag_exercises(nb)
    return nb


# ============================================================ Day 3 下午
def transform_day3_pm(nb: dict) -> dict:
    normalize_paths(nb)

    # 删除 "## 3. 成本优化" markdown
    indices_to_delete = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "markdown" and "## 3. 成本优化" in cell_source(c):
            indices_to_delete.append(i)
    if indices_to_delete:
        delete_cells(nb, indices_to_delete)
        print(f"  ↪ Day3_下午: 删除 {len(indices_to_delete)} 个成本优化 markdown")

    # 合并松散的 # === 分隔 cells
    merged = 0
    new_cells = []
    skip = False
    for i, c in enumerate(nb["cells"]):
        if skip:
            skip = False
            continue
        if c["cell_type"] == "code":
            src = cell_source(c).strip()
            if src.startswith("# ====") and len(src) < 100 and src.count("\n") < 4:
                if i + 1 < len(nb["cells"]) and nb["cells"][i + 1]["cell_type"] == "code":
                    next_src = cell_source(nb["cells"][i + 1])
                    set_cell_source(nb["cells"][i + 1], src + "\n" + next_src)
                    merged += 1
                    new_cells.append(nb["cells"][i + 1])
                    skip = True
                    continue
        new_cells.append(c)
    nb["cells"] = new_cells
    if merged:
        print(f"  ↪ Day3_下午: 合并 {merged} 个松散 # === 分隔 cells")

    note = make_lecture_note(
        title="""Day 3 下午 · Capstone 企业知识助手（3h）""",
        duration_min=180,
        opener="""『3 天的所有积木今天合体。』 → 现场让一组学员说他们部门的真实需求，把 Capstone 用来对照。""",
        key_points=[
            """**Capstone 拼装顺序：** 文档 → 切块 → 向量库 → 检索 → Agent 路由 → 工具调用 → 评测""",
            """**LLM-as-Judge 评测：** 让大模型扮演人类打分，便宜且可大批量""",
            """**部署考量：** 网关、可观测性、安全合规 — 一句话提及，不展开金额""",
            """**复盘：** 让每位学员说 1 个明天回公司能立刻试的小项目""",
        ],
        misconceptions=[
            """学员以为 Capstone 上线就行 → 强调『只是 demo，生产还需 7 大件』（监控/灰度/缓存/限流/审计/SLA/A-B）""",
        ],
        interaction="""分组：技术学员搭后端，业务学员定义评测集 + 写 5 个真实问题。最后 30 分钟现场互测。""",
        if_short_on_time="""跳过部署方案 markdown 段，把时间还给『现场互测 + 复盘』环节。""",
    )
    insert_after(nb, 0, note)
    autotag_exercises(nb)
    return nb


# ============================================================ main
TRANSFORMS = {
    "Day0_环境配置与测试.ipynb": transform_day0,
    "Day1_上午_从文本到向量.ipynb": transform_day1_am,
    "Day1_下午_Transformer架构.ipynb": transform_day1_pm,
    "Day2_上午_预训练与SFT.ipynb": transform_day2_am,
    "Day2_下午_LoRA对齐与评测.ipynb": transform_day2_pm,
    "Day3_上午_RAG与Agent实战.ipynb": transform_day3_am,
    "Day3_下午_企业知识助手Capstone.ipynb": transform_day3_pm,
}


def main():
    order = [
        "Day0_环境配置与测试.ipynb",
        "Day1_上午_从文本到向量.ipynb",
        "Day1_下午_Transformer架构.ipynb",
        "Day2_上午_预训练与SFT.ipynb",
        "Day2_下午_LoRA对齐与评测.ipynb",
        "Day3_上午_RAG与Agent实战.ipynb",
        "Day3_下午_企业知识助手Capstone.ipynb",
    ]
    for fn in order:
        path = INSTRUCTOR / fn
        print(f"\n=== {fn} ===")
        nb = load_nb(path)
        before_cells = len(nb["cells"])
        cost = scan_cost_cells(nb)
        if cost:
            print(f"  ⚠ 预扫到 {len(cost)} 个成本/金额相关 cells:")
            for idx, ct, head in cost[:6]:
                print(f"     [{idx}] {ct} {head}")
        TRANSFORMS[fn](nb)
        after_cells = len(nb["cells"])
        save_nb(nb, path)
        cost_after = scan_cost_cells(nb)
        n_fillin = sum(
            1 for c in nb["cells"]
            if c["cell_type"] == "code"
            and "fillin" in c.get("metadata", {}).get("tags", [])
        )
        n_lecture = sum(
            1 for c in nb["cells"]
            if c["cell_type"] == "markdown"
            and "instructor_only" in c.get("metadata", {}).get("tags", [])
        )
        print(f"  ✓ cells: {before_cells} → {after_cells}")
        print(f"  ✓ 成本相关残留: {len(cost_after)}")
        print(f"  ✓ fillin tags: {n_fillin}  |  instructor_only tags: {n_lecture}")


if __name__ == "__main__":
    main()
