"""Batch 5 rewrite for Day2_下午: 5 exercises (LoRA + DPO + 评测)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_lib import load_nb, save_nb, cell_source, set_cell_source, add_tag

PATH = Path("assets/enterprise_ver2/instructor/Day2_下午_LoRA对齐与评测.ipynb")


EX1 = '''# ============================================================
# 练习 1 | LoRA 前向传播：函数版 → nn.Module 版
# ============================================================
#
# 【基础】(人人必做，5 min)
#   公式: y = W(x) + F.linear(F.linear(x, lora_A), lora_B) * scaling
#   提示：用 F.linear 自动处理转置
#
# 【进阶】(技术学员选做，10 min)
#   实现 LoRALinear nn.Module，含初始化（A 高斯，B 零）+ scaling = alpha/rank
#   注意：B 初始化为 0 才能保证训练开始时旁路输出为 0（不破坏原模型）
# ============================================================

def lora_forward(x, W, lora_A, lora_B, scaling):
    """【基础】LoRA 前向传播：y = Wx + BAx * scaling"""
    # ↓↓↓ 【基础】填空（约 3 行）↓↓↓
    base = W(x)
    lora_out = F.linear(F.linear(x, lora_A), lora_B)
    return base + lora_out * scaling
    # ↑↑↑ 【基础】结束 ↑↑↑


class LoRALinear(nn.Module):
    """【进阶】完整的 LoRA 模块：包装一个 nn.Linear，旁路 (BA) 可训练"""
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        # ↓↓↓ 【进阶】填空（约 7 行）↓↓↓
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.W.weight.requires_grad = False  # 冻结原权重
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))  # B 初始化为 0
        self.scaling = alpha / rank
        # ↑↑↑ 【进阶】结束 ↑↑↑

    def forward(self, x):
        return self.W(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


def verify():
    print("=" * 56); print("【基础】lora_forward (函数版)"); print("=" * 56)
    try:
        scaling = 16.0 / 8
        W = nn.Linear(64, 64, bias=False)
        x = torch.randn(4, 64)
        rank = 8
        in_features, out_features = 64, 64
        lora_A = torch.randn(rank, in_features) * 0.01
        lora_B = torch.zeros(out_features, rank)
        y = lora_forward(x, W, lora_A, lora_B, scaling)
        base = W(x)
        # B 初始化为 0 时，y 应等于 base
        assert torch.allclose(y, base, atol=1e-6), "B=0 时 y 应等于 base"
        # 给 B 赋值后，y 应不等于 base
        lora_B = torch.randn(out_features, rank) * 0.1
        y2 = lora_forward(x, W, lora_A, lora_B, scaling)
        assert not torch.allclose(y2, base)
        print(f"  y shape: {y.shape}  ✓")
        print(f"  B=0 时 y == base ✓ (LoRA 不影响原输出)")
        print(f"  B≠0 时 y ≠ base ✓ (旁路生效)")
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】LoRALinear nn.Module"); print("=" * 56)
    try:
        layer = LoRALinear(64, 64, rank=8, alpha=16)
        x = torch.randn(4, 64)
        y = layer(x)
        assert y.shape == (4, 64)
        # B 初始化为 0 → 初始时 y == W(x)
        assert torch.allclose(y, layer.W(x), atol=1e-6)
        # 检查可训练参数 = lora_A + lora_B (W 冻结)
        trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        total = sum(p.numel() for p in layer.parameters())
        ratio = trainable / total
        print(f"  可训练参数: {trainable:,} / 总参数: {total:,} = {ratio:.1%}")
        print(f"  ✓ 旁路 BA = {trainable} 参数；原 W = {total - trainable} 参数（冻结）")
        print(f"  ✓ scaling = alpha/rank = 16/8 = {layer.scaling}")
        assert ratio < 0.3
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


EX2 = '''# ============================================================
# 练习 2 | LoRA Rank 对比 + Adapter 切换
# ============================================================
#
# 【基础】(人人必做，10 min)
#   测试 rank ∈ {4, 16, 32} 的最终 MSE，看 rank 越大拟合是否越好
#
# 【进阶】(技术学员选做，15 min)
#   实现 train_two_adapters：在同一个 base 上挂两个 LoRA adapter
#   分别训不同任务 (Y1 / Y2)，再切换前向用哪个 adapter
#   这是工业界『一个 base 模型 + N 个任务 LoRA』的核心模式
# ============================================================

torch.manual_seed(42)
X_train = torch.randn(200, 64)
W_true = torch.randn(64, 64) * 0.1
noise = torch.randn(200, 64) * 0.01
Y_train = X_train @ W_true + noise


def train_one_rank(rank, n_steps=100, lr=0.01):
    """【基础】训一个 rank，返回最终 loss"""
    # ↓↓↓ 【基础】填空（约 7 行）↓↓↓
    lora = LoRALinear(64, 64, rank=rank, alpha=rank * 2)
    optimizer = torch.optim.Adam(lora.parameters(), lr=lr)
    for _ in range(n_steps):
        loss = F.mse_loss(lora(X_train), Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()
    # ↑↑↑ 【基础】结束 ↑↑↑


def train_two_adapters(W_true_2, n_steps=80, rank=8, lr=0.01):
    """【进阶】同一 base 上挂两个 adapter，分别学 (X→Y1) 与 (X→Y2)
    Returns: (adapter1, adapter2, base_W)
    """
    # ↓↓↓ 【进阶】填空（约 14 行）↓↓↓
    Y2_train = X_train @ W_true_2 + torch.randn(200, 64) * 0.01
    base = nn.Linear(64, 64, bias=False)
    base.weight.requires_grad = False
    # adapter 1
    a1_A = nn.Parameter(torch.randn(rank, 64) * 0.01)
    a1_B = nn.Parameter(torch.zeros(64, rank))
    opt1 = torch.optim.Adam([a1_A, a1_B], lr=lr)
    for _ in range(n_steps):
        out = base(X_train) + F.linear(F.linear(X_train, a1_A), a1_B) * 2
        loss = F.mse_loss(out, Y_train)
        opt1.zero_grad(); loss.backward(); opt1.step()
    # adapter 2
    a2_A = nn.Parameter(torch.randn(rank, 64) * 0.01)
    a2_B = nn.Parameter(torch.zeros(64, rank))
    opt2 = torch.optim.Adam([a2_A, a2_B], lr=lr)
    for _ in range(n_steps):
        out = base(X_train) + F.linear(F.linear(X_train, a2_A), a2_B) * 2
        loss = F.mse_loss(out, Y2_train)
        opt2.zero_grad(); loss.backward(); opt2.step()
    return (a1_A, a1_B), (a2_A, a2_B), base
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】不同 rank 的拟合对比"); print("=" * 56)
    try:
        results = {r: train_one_rank(r) for r in [4, 16, 32]}
        for r, l in results.items():
            print(f"  rank={r:>2}  →  final MSE = {l:.6f}")
        # rank 越大通常 loss 越低（不严格）
        assert len(results) == 3
        try:
            import matplotlib.pyplot as plt
            plt.bar([f'r={r}' for r in [4,16,32]], [results[r] for r in [4,16,32]],
                    color=['#3498db','#2ecc71','#e67e22'])
            plt.ylabel('Final MSE'); plt.title('LoRA Rank vs Fitting Ability')
            plt.show()
        except Exception:
            pass
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】两个 adapter 共享同一 base"); print("=" * 56)
    try:
        W_true_2 = torch.randn(64, 64) * 0.2  # 不同的目标
        (A1, B1), (A2, B2), base = train_two_adapters(W_true_2, n_steps=80)
        # 切换 adapter 1 → 应该接近 Y_train
        out1 = base(X_train) + F.linear(F.linear(X_train, A1), B1) * 2
        loss1 = F.mse_loss(out1, Y_train).item()
        # 切换 adapter 2 → 应该接近 Y2_train (用 W_true_2)
        Y2 = X_train @ W_true_2
        out2 = base(X_train) + F.linear(F.linear(X_train, A2), B2) * 2
        loss2 = F.mse_loss(out2, Y2).item()
        print(f"  Adapter 1 在任务 1 的 loss: {loss1:.4f}")
        print(f"  Adapter 2 在任务 2 的 loss: {loss2:.4f}")
        print(f"  💡 同一 base，切换不同 adapter → 不同任务行为")
        assert loss1 < 0.5 and loss2 < 0.5
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


EX3 = '''# ============================================================
# 练习 3 | GPU 显存估算 + 训练吞吐估算
# ============================================================
#
# 【基础】(人人必做，10 min)
#   公式：模型 + 梯度 + 优化器 + 激活值 ≈ 总显存
#
# 【进阶】(技术学员选做，10 min)
#   实现 estimate_throughput(...)：估算每秒处理 token 数
#   公式：FLOPS_per_token = 6 × params × seq_len（前+反向）
#   throughput = GPU_TFLOPS / FLOPS_per_token
# ============================================================

def estimate_gpu_memory(num_params_billion, precision_bytes, trainable_ratio, optimizer="adam"):
    """【基础】估算 GPU 显存 (GB)"""
    # ↓↓↓ 【基础】填空（约 8 行）↓↓↓
    trainable = num_params_billion * trainable_ratio
    opt_bytes = 8 if optimizer == "adam" else 4
    model_gb = num_params_billion * precision_bytes
    gradient_gb = trainable * 2
    optimizer_gb = trainable * opt_bytes
    activation_gb = model_gb * 0.5
    total = model_gb + gradient_gb + optimizer_gb + activation_gb
    return {"model": model_gb, "gradient": gradient_gb, "optimizer": optimizer_gb,
            "activation": activation_gb, "total": total}
    # ↑↑↑ 【基础】结束 ↑↑↑


def estimate_throughput(num_params_billion, seq_len, gpu_tflops):
    """【进阶】估算训练吞吐量 (tokens/s)
    每个 token 训练前向+反向约 6 × params × seq_len FLOPS (经验式)
    """
    # ↓↓↓ 【进阶】填空（约 3 行）↓↓↓
    flops_per_token = 6 * num_params_billion * 1e9 * seq_len
    gpu_flops = gpu_tflops * 1e12
    return gpu_flops / flops_per_token
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】estimate_gpu_memory"); print("=" * 56)
    try:
        scenarios = [
            ("7B Full FP16",   7,  2,   1.0,    "adam"),
            ("7B LoRA FP16",   7,  2,   0.01,   "adam"),
            ("7B QLoRA INT4",  7,  0.5, 0.01,   "adam"),
            ("13B QLoRA INT4", 13, 0.5, 0.005,  "adam"),
        ]
        print(f"  {'Scenario':<18} {'Model':>7} {'Grad':>7} {'Optim':>7} {'Act':>7} {'Total':>7}")
        for name, p, prec, ratio, opt in scenarios:
            r = estimate_gpu_memory(p, prec, ratio, opt)
            print(f"  {name:<18} {r['model']:>6.1f}G {r['gradient']:>6.1f}G "
                  f"{r['optimizer']:>6.1f}G {r['activation']:>6.1f}G {r['total']:>6.1f}G")
        ok = estimate_gpu_memory(7, 2, 1.0)
        assert abs(ok['model'] - 14.0) < 0.1
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】训练吞吐量估算"); print("=" * 56)
    try:
        # A100 FP16 ≈ 312 TFLOPS； H100 ≈ 989 TFLOPS
        cases = [
            ("7B @ A100 FP16, seq=2k", 7, 2048, 312),
            ("7B @ H100 FP16, seq=2k", 7, 2048, 989),
            ("13B @ A100 FP16, seq=4k", 13, 4096, 312),
        ]
        for name, p, seq, tflops in cases:
            tps = estimate_throughput(p, seq, tflops)
            print(f"  {name:<28}  →  {tps:>10,.0f} tokens/s")
        a100 = estimate_throughput(7, 2048, 312)
        assert a100 > 1000
        print("  💡 实际数字略低于估算 (内存带宽 / 通信开销)，但量级吻合")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


EX4 = '''# ============================================================
# 练习 4 | DPO 评测：偏好准确率 + Reward Margin
# ============================================================
#
# 【基础】(人人必做，10 min)
#   生成对比 + 偏好准确率：模型是否给 chosen 更高 log-prob？
#
# 【进阶】(技术学员选做，10 min)
#   计算 reward_margin = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
#   这是 DPO loss 的核心信号；> 0 表示策略相对参考模型更倾向 chosen
# ============================================================

TEST_QUESTIONS = [
    "中国的首都是哪里？",
    "什么是机器学习？",
    "日本的首都是哪里？",
    "什么是深度学习？",
    "HTTP是什么？",
]


def show_generations(model_ref, model_policy, questions):
    """【基础】对比 ref vs DPO 在测试问题上的生成"""
    # ↓↓↓ 【基础】填空（约 4 行）↓↓↓
    for q in questions:
        ref = generate_response(model_ref, q)
        dpo = generate_response(model_policy, q)
        print(f"Q: {q}\\n  Ref: {ref}\\n  DPO: {dpo}\\n")
    # ↑↑↑ 【基础】结束 ↑↑↑


def eval_pref_acc(model, dataset):
    """【基础】偏好准确率：模型对 chosen 给出的 log-prob > rejected 的比例"""
    # ↓↓↓ 【基础】填空（约 12 行）↓↓↓
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for sample in dataset:
            c_ids = sample['chosen_input_ids'].unsqueeze(0).to(device)
            c_mask = sample['chosen_attention_mask'].unsqueeze(0).to(device)
            r_ids = sample['rejected_input_ids'].unsqueeze(0).to(device)
            r_mask = sample['rejected_attention_mask'].unsqueeze(0).to(device)
            pl = torch.tensor([sample['prompt_length']]).to(device)
            c_logp = compute_log_probs(model, c_ids, c_mask, pl).item()
            r_logp = compute_log_probs(model, r_ids, r_mask, pl).item()
            if c_logp > r_logp: correct += 1
            total += 1
    return correct / total if total > 0 else 0.0
    # ↑↑↑ 【基础】结束 ↑↑↑


def reward_margin(policy_model, ref_model, dataset):
    """【进阶】平均 reward margin (DPO loss 的核心信号)
    margin = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    """
    # ↓↓↓ 【进阶】填空（约 16 行）↓↓↓
    margins = []
    policy_model.eval(); ref_model.eval()
    with torch.no_grad():
        for sample in dataset:
            c_ids = sample['chosen_input_ids'].unsqueeze(0).to(device)
            c_mask = sample['chosen_attention_mask'].unsqueeze(0).to(device)
            r_ids = sample['rejected_input_ids'].unsqueeze(0).to(device)
            r_mask = sample['rejected_attention_mask'].unsqueeze(0).to(device)
            pl = torch.tensor([sample['prompt_length']]).to(device)
            pol_c = compute_log_probs(policy_model, c_ids, c_mask, pl).item()
            pol_r = compute_log_probs(policy_model, r_ids, r_mask, pl).item()
            ref_c = compute_log_probs(ref_model, c_ids, c_mask, pl).item()
            ref_r = compute_log_probs(ref_model, r_ids, r_mask, pl).item()
            margins.append((pol_c - ref_c) - (pol_r - ref_r))
    return sum(margins) / len(margins) if margins else 0.0
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】生成对比 + 偏好准确率"); print("=" * 56)
    try:
        show_generations(reference_model, policy_model, TEST_QUESTIONS[:2])
        ref_acc = eval_pref_acc(reference_model, dpo_dataset)
        dpo_acc = eval_pref_acc(policy_model, dpo_dataset)
        print(f"\\n  偏好准确率: Ref = {ref_acc:.0%}  |  DPO = {dpo_acc:.0%}")
        assert 0 <= ref_acc <= 1 and 0 <= dpo_acc <= 1
        print("✅ 基础通过 — DPO acc 通常 > Ref acc\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】Reward Margin (DPO 训练信号)"); print("=" * 56)
    try:
        margin = reward_margin(policy_model, reference_model, dpo_dataset)
        print(f"  平均 reward margin = {margin:+.4f}")
        print(f"  💡 margin > 0 表示 DPO 训练有效（策略更偏向 chosen）")
        print(f"     margin 越大，模型对 chosen / rejected 的区分越强")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


EX5 = '''# ============================================================
# 练习 5 | 评测指标：EM/F1 → BLEU/ROUGE
# ============================================================
#
# 【基础】(人人必做，10 min)
#   EM (精确匹配) + F1 (字符级)
#
# 【进阶】(技术学员选做，10 min)
#   实现 char_bleu(pred, ref, n=2)：bigram BLEU 简化版
#   返回 modified precision (clip count) — BLEU 的核心组件
# ============================================================

eval_pairs = [
    ("产品质量问题", "产品质量"),
    ("物流配送", "物流配送"),
    ("售后服务相关", "售后服务"),
    ("关于价格的争议", "价格争议"),
    ("账户问题", "账户问题"),
    ("产品质量", "产品质量"),
    ("物流延迟问题", "物流配送"),
    ("售后", "售后服务"),
    ("价格争议", "价格争议"),
    ("账户安全问题", "账户问题"),
]
predictions = [p[0] for p in eval_pairs]
references = [p[1] for p in eval_pairs]


def compute_exact_match(preds, refs):
    """【基础】精确匹配率：pred == ref 的比例"""
    # ↓↓↓ 【基础】填空（约 2 行）↓↓↓
    matches = sum(1 for p, r in zip(preds, refs) if p == r)
    return matches / len(preds) if preds else 0.0
    # ↑↑↑ 【基础】结束 ↑↑↑


def compute_f1_token_overlap(pred, ref):
    """【基础】字符集 F1：交集 / (precision + recall)"""
    # ↓↓↓ 【基础】填空（约 6 行）↓↓↓
    pred_chars = set(pred); ref_chars = set(ref)
    common = pred_chars & ref_chars
    if not pred_chars or not ref_chars or not common:
        return 0.0
    p = len(common) / len(pred_chars)
    r = len(common) / len(ref_chars)
    return 2 * p * r / (p + r)
    # ↑↑↑ 【基础】结束 ↑↑↑


def char_bleu_bigram(pred, ref):
    """【进阶】字符级 BLEU-2 的 modified precision (含 clip count)
    取 pred 中所有 bigram，每个最多按 ref 中出现次数计数 (避免重复词刷分)
    """
    # ↓↓↓ 【进阶】填空（约 12 行）↓↓↓
    if len(pred) < 2 or len(ref) < 2:
        return 0.0
    pred_bigrams = [pred[i:i+2] for i in range(len(pred)-1)]
    ref_bigrams = [ref[i:i+2] for i in range(len(ref)-1)]
    from collections import Counter
    pred_counts = Counter(pred_bigrams)
    ref_counts = Counter(ref_bigrams)
    # clip：每个 pred bigram 计数不超过 ref 中的次数
    clipped = 0
    for bg, c in pred_counts.items():
        clipped += min(c, ref_counts.get(bg, 0))
    return clipped / len(pred_bigrams)
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】EM + F1"); print("=" * 56)
    try:
        em = compute_exact_match(predictions, references)
        f1s = [compute_f1_token_overlap(p, r) for p, r in zip(predictions, references)]
        avg_f1 = sum(f1s) / len(f1s)
        print(f"  EM = {em:.0%}  |  avg F1 = {avg_f1:.0%}")
        # 对比表
        print(f"\\n  {'#':<3} {'pred':<14} {'ref':<10} {'EM':>4} {'F1':>6}")
        for i, (p, r, f1) in enumerate(zip(predictions, references, f1s)):
            em_i = 1 if p == r else 0
            print(f"  {i+1:<3} {p:<14} {r:<10} {em_i:>4} {f1:>6.2f}")
        em_unit = compute_exact_match(["A","B","C"], ["A","B","D"])
        assert abs(em_unit - 2/3) < 1e-3
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】char-bigram BLEU (modified precision)"); print("=" * 56)
    try:
        b_perfect = char_bleu_bigram("产品质量", "产品质量")
        b_partial = char_bleu_bigram("产品质量问题", "产品质量")
        b_zero = char_bleu_bigram("完全无关", "产品质量")
        print(f"  完全相同   '产品质量' vs '产品质量'         BLEU-2 = {b_perfect:.3f}")
        print(f"  部分重叠   '产品质量问题' vs '产品质量'     BLEU-2 = {b_partial:.3f}")
        print(f"  完全不同   '完全无关' vs '产品质量'         BLEU-2 = {b_zero:.3f}")
        assert b_perfect == 1.0 and b_zero == 0.0
        # 平均 BLEU 跟 F1 走向相似但不一样
        avg_bleu = sum(char_bleu_bigram(p, r) for p, r in zip(predictions, references)) / len(predictions)
        print(f"\\n  整套 eval 数据的 avg BLEU-2 = {avg_bleu:.3f}")
        print(f"  💡 BLEU 关注 n-gram 顺序，F1 只关心字符集——两者从不同角度衡量")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


REWRITES = {
    "练习 1：实现 LoRA 前向传播": EX1,
    "练习 2：对比不同 rank 的效果": EX2,
    "练习 3：计算 GPU 显存需求": EX3,
    "练习 4：对比 DPO 训练前后的输出": EX4,
    "练习 5：自己计算评测指标": EX5,
}


def main():
    nb = load_nb(PATH)
    n_replaced = 0
    for marker, new_src in REWRITES.items():
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            src = cell_source(c)
            first = src.strip().split("\n")[0]
            if marker in first:
                set_cell_source(c, new_src)
                add_tag(c, "fillin")
                add_tag(c, "batch5")
                c["outputs"] = []
                c["execution_count"] = None
                print(f"  ✓ 重写: {marker}")
                n_replaced += 1
                break
        else:
            print(f"  ⚠ 未找到: {marker}")
    save_nb(nb, PATH)
    print(f"\nTotal rewritten: {n_replaced}/{len(REWRITES)}")


if __name__ == "__main__":
    main()
