"""Rebalance Day 2: move SFT block from PM back to AM (restore v1's natural split).

Result:
  Day2_上午 = 预训练 + SFT (~60 cells)
  Day2_下午 = LoRA + DPO + 评测 (~58 cells)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_lib import load_nb, save_nb, cell_source, set_cell_source, make_md, make_lecture_note

AM_PATH = Path("assets/enterprise_ver2/instructor/Day2_上午_预训练与SFT.ipynb")
PM_PATH = Path("assets/enterprise_ver2/instructor/Day2_下午_LoRA对齐与评测.ipynb")


def main():
    pm = load_nb(PM_PATH)
    am = load_nb(AM_PATH)

    # ---- Locate SFT block in PM ----
    receive_idx = None
    for i, c in enumerate(pm["cells"]):
        if c["cell_type"] == "markdown" and "块 1 · 承接上午" in cell_source(c):
            receive_idx = i
            break
    if receive_idx is None:
        print("⚠ 找不到 receive_md，可能已经撤销过了")
        return

    # The next markdown that begins LoRA section starts the boundary
    # Find: cell with original v1's "import os, warnings, logging" (LoRA setup)
    lora_start_idx = None
    for i in range(receive_idx + 1, len(pm["cells"])):
        c = pm["cells"][i]
        if c["cell_type"] == "code":
            src = cell_source(c)
            if "import os, warnings, logging" in src and "import torch" not in src:
                lora_start_idx = i
                break
    if lora_start_idx is None:
        print("⚠ 找不到 LoRA section 起始点")
        return

    print(f"PM: receive_md=[{receive_idx}], LoRA starts=[{lora_start_idx}]")
    sft_block = pm["cells"][receive_idx + 1 : lora_start_idx]   # exclude receive_md, keep SFT
    print(f"SFT block: {len(sft_block)} cells will move back to AM")

    # Remove receive_md + SFT block from PM
    pm["cells"] = pm["cells"][:receive_idx] + pm["cells"][lora_start_idx:]

    # ---- Insert into AM before the 上午小结 ----
    summary_idx = None
    for i, c in enumerate(am["cells"]):
        if c["cell_type"] == "markdown" and "上午小结" in cell_source(c):
            summary_idx = i
            break
    if summary_idx is None:
        print("⚠ AM: 找不到 上午小结，append 到末尾")
        summary_idx = len(am["cells"])

    # Insert a transition markdown
    transition = make_md(
        "---\n\n"
        "## Part 3 · SFT 指令微调（接续预训练，~75 min）\n\n"
        "上午前半我们完成了**预训练完整闭环**——模型从乱码到通顺。\n"
        "但 base model 只会**续写文本**，不会**对话**。下半场进入 **SFT (Supervised Fine-Tuning)**：\n"
        "用 instruction-response 数据 + ChatML 格式 + Loss Masking 教模型『听话』。\n"
    )
    am["cells"][summary_idx:summary_idx] = [transition] + sft_block

    # Rewrite 上午小结 to reflect new scope
    new_summary_idx = summary_idx + 1 + len(sft_block)
    new_summary = (
        "---\n\n"
        "## 上午小结\n\n"
        "上半场：**预训练完整闭环**（数据 → 词表 → 模型 → 训练循环 → 生成对比）。\n\n"
        "下半场：**SFT 指令微调**（ChatML + Loss Masking + 训练 + 评估）。\n\n"
        "下午我们进入 **LoRA 高效微调** + **DPO 对齐** + **评测三件套**。\n\n"
        "**☕ 午休 1 小时**\n"
    )
    set_cell_source(am["cells"][new_summary_idx], new_summary)

    # ---- Update lecture notes for both ----
    # AM lecture: include SFT
    new_am_note = make_lecture_note(
        title="""Day 2 上午 · 预训练 + SFT（3h，加 30 min 缓冲）""",
        duration_min=180,
        opener="""问：『婴儿学说话靠什么？』 → 大量听 → 引出预训练 = 模型大量『听』文本，学会『下一个字最可能是什么』。""",
        key_points=[
            """**Part 1 预训练 (90 min):** Next Token Prediction、字符级词表、训练循环 4 步、loss 曲线诊断""",
            """**Part 3 SFT (75 min):** Base model 不会聊天 → 用 ChatML + Loss Masking 教对话能力""",
            """**关键差别：** 预训练学『语言规律』(无监督)；SFT 学『对话行为』(有指令-回答监督)""",
            """两者都是 next token prediction，但 loss 计算 mask 的位置不同""",
        ],
        misconceptions=[
            """学员以为预训练 = 教回答问题 → 强调 base model 只会续写""",
            """学员以为 SFT 是『更深的预训练』 → 强调 SFT 改变的是『输入-输出格式』，不是知识""",
        ],
        interaction="""把训练数据换成自己公司业务文本（10 行），跑 50 epoch 看输出是否出现业务术语；SFT 同样让学员现场加 1-2 条 instruction-response。""",
        if_short_on_time="""跳过预训练第 5 个练习『调超参』；SFT 只演示训练，不让所有人手动跑（保留观察 + 思考题）。""",
    )
    # Replace the existing 📋 in AM (cell [1])
    for i, c in enumerate(am["cells"]):
        if c["cell_type"] == "markdown" and "📋" in cell_source(c) and "instructor_only" in c.get("metadata", {}).get("tags", []):
            am["cells"][i] = new_am_note
            break

    # PM lecture: drop SFT scope
    new_pm_note = make_lecture_note(
        title="""Day 2 下午 · LoRA + DPO + 评测三件套（3h）""",
        duration_min=180,
        opener="""问：『上午我们用 SFT 教会模型对话，但 SFT 要训练全部参数，70B 模型要多少显卡？』 → 引出 LoRA：只训 1% 参数。""",
        key_points=[
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
        if_short_on_time="""DPO 部分只讲公式与一次推理对比，跳过完整训练；评测保留 PPL + 一个 Judge 例子。""",
    )
    for i, c in enumerate(pm["cells"]):
        if c["cell_type"] == "markdown" and "📋" in cell_source(c) and "instructor_only" in c.get("metadata", {}).get("tags", []):
            pm["cells"][i] = new_pm_note
            break

    save_nb(am, AM_PATH)
    save_nb(pm, PM_PATH)
    print(f"\n✓ AM cells: {len(am['cells'])}  |  PM cells: {len(pm['cells'])}")


if __name__ == "__main__":
    main()
