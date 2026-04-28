"""Fill in answer code for the 5 LoRA/DPO/eval exercises in Day2_下午.

These exercises use `pass` placeholder between `# ↓↓↓ 你的代码` and `# ↑↑↑ 你的代码`.
We replace the `pass` with the canonical answer derived from inline hints.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_lib import load_nb, save_nb, cell_source, set_cell_source

PATH = Path("assets/enterprise_ver2/instructor/Day2_下午_LoRA对齐与评测.ipynb")

# (substring to find within cell source, replacement_for_pass)
ANSWERS = [
    # 练习 1：实现 LoRA 前向传播
    (
        "练习 1：实现 LoRA 前向传播",
        """    base = W(x)
    lora_out = x @ lora_A @ lora_B
    return base + lora_out * scaling""",
    ),
    # 练习 2：对比不同 rank 的效果
    (
        "练习 2：对比不同 rank 的效果",
        """    lora = LoRALinear(64, 64, rank=r, alpha=r * 2)
    optimizer = torch.optim.Adam(lora.parameters(), lr=0.01)
    for _ in range(100):
        loss = F.mse_loss(lora(X_train), Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    results[r] = loss.item()""",
    ),
    # 练习 3：计算 GPU 显存需求
    (
        "练习 3：计算 GPU 显存需求",
        """    trainable = num_params_billion * trainable_ratio
    opt_bytes = 8 if optimizer == "adam" else 4
    model_gb = num_params_billion * precision_bytes
    gradient_gb = trainable * 2  # FP16 grads
    optimizer_gb = trainable * opt_bytes
    activation_gb = model_gb * 0.5
    total = model_gb + gradient_gb + optimizer_gb + activation_gb
    return {
        "model": model_gb,
        "gradient": gradient_gb,
        "optimizer": optimizer_gb,
        "activation": activation_gb,
        "total": total,
    }""",
    ),
    # 练习 5：自己计算评测指标 — EM
    (
        "compute_exact_match",
        """    matches = sum(1 for p, r in zip(preds, refs) if p == r)
    return matches / len(preds) if preds else 0.0""",
    ),
    # 练习 5：自己计算评测指标 — F1
    (
        "compute_f1_token_overlap",
        """    pred_chars = set(pred)
    ref_chars = set(ref)
    common = pred_chars & ref_chars
    if not pred_chars or not ref_chars or not common:
        return 0.0
    precision = len(common) / len(pred_chars)
    recall = len(common) / len(ref_chars)
    return 2 * precision * recall / (precision + recall)""",
    ),
]


def fill_pass_in_block(src: str, replacement: str) -> tuple[str, bool]:
    """Replace the `pass` (any indent) between ↓↓↓ and ↑↑↑ markers with replacement."""
    lines = src.splitlines(keepends=True)
    out_lines = []
    in_block = False
    replaced = False
    for ln in lines:
        if "↓↓↓" in ln and "你的代码" in ln:
            in_block = True
            out_lines.append(ln)
            continue
        if "↑↑↑" in ln and "你的代码" in ln:
            in_block = False
            out_lines.append(ln)
            continue
        if in_block and not replaced:
            stripped = ln.strip()
            if stripped == "pass":
                out_lines.append(replacement.rstrip("\n") + "\n")
                replaced = True
                continue
            # Skip hint comment lines? No — keep them for student readability
        out_lines.append(ln)
    return "".join(out_lines), replaced


def main():
    nb = load_nb(PATH)
    print(f"Day2_下午: {len(nb['cells'])} cells")
    n_filled = 0
    for marker, answer_code in ANSWERS:
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] != "code":
                continue
            src = cell_source(c)
            if marker not in src:
                continue
            new_src, replaced = fill_pass_in_block(src, answer_code)
            if replaced:
                set_cell_source(c, new_src)
                print(f"  ✓ filled '{marker}' in cell [{i}]")
                n_filled += 1
            else:
                print(f"  ⚠ '{marker}' found in cell [{i}] but no `pass` to replace")
            break

    # 练习 4 (DPO) — special handling: 2 separate ↓↓↓ blocks in same cell
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = cell_source(c)
        if "练习 4：对比 DPO 训练前后的输出" not in src:
            continue
        # Block 1: 生成对比 (replace `pass` after `# 用 generate_response` hint)
        lines = src.splitlines(keepends=True)
        out, in_block, blk = [], False, 0
        for ln in lines:
            if "↓↓↓" in ln and "你的代码" in ln:
                in_block = True
                blk += 1
                out.append(ln)
                continue
            if "↑↑↑" in ln and "你的代码" in ln:
                in_block = False
                out.append(ln)
                continue
            if in_block and ln.strip() == "pass":
                if blk == 1:
                    out.append("    ref_resp = generate_response(reference_model, q)\n")
                    out.append("    dpo_resp = generate_response(policy_model, q)\n")
                    out.append('    print(f"Q: {q}")\n')
                    out.append('    print(f"  Ref: {ref_resp}")\n')
                    out.append('    print(f"  DPO: {dpo_resp}\\n")\n')
                elif blk == 2:
                    out.append("    correct, total = 0, 0\n")
                    out.append("    model.eval()\n")
                    out.append("    with torch.no_grad():\n")
                    out.append("        for sample in dataset:\n")
                    out.append("            c_ids = sample['chosen_input_ids'].unsqueeze(0).to(device)\n")
                    out.append("            c_mask = sample['chosen_attention_mask'].unsqueeze(0).to(device)\n")
                    out.append("            r_ids = sample['rejected_input_ids'].unsqueeze(0).to(device)\n")
                    out.append("            r_mask = sample['rejected_attention_mask'].unsqueeze(0).to(device)\n")
                    out.append("            pl = torch.tensor([sample['prompt_length']]).to(device)\n")
                    out.append("            c_logp = compute_log_probs(model, c_ids, c_mask, pl).item()\n")
                    out.append("            r_logp = compute_log_probs(model, r_ids, r_mask, pl).item()\n")
                    out.append("            if c_logp > r_logp:\n")
                    out.append("                correct += 1\n")
                    out.append("            total += 1\n")
                    out.append("    return correct / total if total > 0 else 0.0\n")
                continue
            out.append(ln)
        set_cell_source(c, "".join(out))
        print(f"  ✓ filled '练习 4：对比 DPO' in cell [{i}] (2 sub-blocks)")
        n_filled += 1
        break

    save_nb(nb, PATH)
    print(f"\nTotal filled: {n_filled}")


if __name__ == "__main__":
    main()
