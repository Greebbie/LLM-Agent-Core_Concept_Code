"""Batch 5 rewrite for Day2_上午: 6 exercises (2 pretrain + 4 SFT)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_lib import load_nb, save_nb, cell_source, set_cell_source, add_tag

PATH = Path("assets/enterprise_ver2/instructor/Day2_上午_预训练与SFT.ipynb")


EX1 = '''# ============================================================
# 练习 1 | 数据质量对模型质量的影响
# ============================================================
#
# 【基础】(人人必做，10 min)
#   构造低质量数据（重复 + 单字符）→ 训练同结构模型 → 观察生成结果
#   提示：noisy_text 用大量 "这是重复文本。\\n" + "啊"
#
# 【进阶】(技术学员选做，10 min)
#   对比 3 种数据：good (原语料) / noisy (重复) / mixed (50% 各)
#   返回各自的最终 loss + 各跑 5 个 token 看生成质量差异
# ============================================================

# ──── 【基础】构造 noisy 数据 + 训练 ────
def make_noisy_text():
    """【基础】构造低质量训练数据"""
    # ↓↓↓ 【基础】填空（1 行）↓↓↓
    return "这是重复文本。\\n" * 300 + "啊" * 200
    # ↑↑↑ 【基础】结束 ↑↑↑


def train_noisy_step(model, x, y, optimizer):
    """【基础】训练一步"""
    # ↓↓↓ 【基础】填空（4 行）↓↓↓
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
    # ↑↑↑ 【基础】结束 ↑↑↑


# ──── 【进阶】3 种数据对比 ────
def make_mixed_text(good_text):
    """【进阶】混合一半好数据 + 一半 noisy 数据"""
    # ↓↓↓ 【进阶】填空（约 3 行）↓↓↓
    half_lines = good_text.splitlines()[: len(good_text.splitlines()) // 2]
    good_half = "\\n".join(half_lines)
    return good_half + "\\n" + ("这是重复文本。\\n" * 150)
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】训练 noisy 模型 30 轮"); print("=" * 56)
    try:
        noisy_text = make_noisy_text()
        assert len(noisy_text) > 1000
        noisy_dataset = TextDataset(noisy_text, block_size)
        noisy_loader = DataLoader(noisy_dataset, batch_size=batch_size, shuffle=True)
        noisy_model = MiniGPT(config).to(device)
        noisy_opt = torch.optim.AdamW(noisy_model.parameters(), lr=3e-4)
        losses = []
        for epoch in range(30):
            for x, y in noisy_loader:
                x, y = x.to(device), y.to(device)
                loss = train_noisy_step(noisy_model, x, y, noisy_opt)
                losses.append(loss.item())
        print(f"  noisy 数据训练完成 | 最终 loss = {losses[-1]:.4f}")
        # 生成对比
        with torch.no_grad():
            idx = torch.zeros((1, 1), dtype=torch.long).to(device)
            out = noisy_model.generate(idx, max_new_tokens=30)
            text = ''.join([itos[i.item()] for i in out[0]])
        print(f"  生成 (noisy 模型): {text[:60]!r}")
        print("✅ 基础通过 — 模型在 noisy 数据上 loss 下降但生成显著退化\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】3 种数据 (good / noisy / mixed) 对比"); print("=" * 56)
    try:
        results = {}
        for name, txt in [("good", text_data := text), ("noisy", noisy_text), ("mixed", make_mixed_text(text))]:
            ds = TextDataset(txt, block_size)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
            m = MiniGPT(config).to(device)
            opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
            ll = []
            for _ in range(20):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    _, loss = m(x, y); opt.zero_grad(); loss.backward(); opt.step()
                    ll.append(loss.item())
            results[name] = ll[-1]
        print(f"  最终 loss 对比: {results}")
        print(f"  💡 noisy 数据 loss 看起来低（重复模式好学），但生成质量最差 — loss 不等于质量")
        assert "good" in results and "noisy" in results and "mixed" in results
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


EX2 = '''# ============================================================
# 练习 2 | 过拟合诊断 + Early Stopping
# ============================================================
#
# 【基础】(人人必做，10 min)
#   用极小数据集训练 → 跟踪 train_loss + val_loss → 看到过拟合 gap
#
# 【进阶】(技术学员选做，10 min)
#   实现 EarlyStopper：val_loss 连续 patience 轮没下降就停
#   返回 (early_stop_epoch, best_val_loss, best_train_loss_at_that_epoch)
# ============================================================

tiny_text = "\\n".join(text.splitlines()[:10])
tiny_val_text = "\\n".join(text.splitlines()[10:20])


def basic_train_step(model, x, y, optimizer):
    """【基础】一步训练"""
    # ↓↓↓ 【基础】填空（4 行）↓↓↓
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
    # ↑↑↑ 【基础】结束 ↑↑↑


class EarlyStopper:
    """【进阶】当 val_loss 连续 patience 轮没改善就停"""
    def __init__(self, patience=3, min_delta=1e-4):
        # ↓↓↓ 【进阶】填空（约 3 行）↓↓↓
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.counter = 0
        self.best_epoch = 0
        # ↑↑↑ 【进阶】结束 ↑↑↑

    def __call__(self, val_loss, epoch):
        """返回 True 表示应该停止"""
        # ↓↓↓ 【进阶】填空（约 6 行）↓↓↓
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
        # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】跑 50 轮，观察 train/val loss gap"); print("=" * 56)
    try:
        tiny_ds = TextDataset(tiny_text, block_size)
        tiny_val_ds = TextDataset(tiny_val_text, block_size)
        tiny_loader = DataLoader(tiny_ds, batch_size=batch_size, shuffle=True)
        tiny_val_loader = DataLoader(tiny_val_ds, batch_size=batch_size)
        tiny_model = MiniGPT(config).to(device)
        tiny_opt = torch.optim.AdamW(tiny_model.parameters(), lr=3e-4)
        train_losses, val_losses = [], []
        for epoch in range(50):
            tiny_model.train()
            t = []
            for x, y in tiny_loader:
                x, y = x.to(device), y.to(device)
                loss = basic_train_step(tiny_model, x, y, tiny_opt)
                t.append(loss.item())
            tiny_model.eval()
            with torch.no_grad():
                v = []
                for x, y in tiny_val_loader:
                    x, y = x.to(device), y.to(device)
                    _, lv = tiny_model(x, y); v.append(lv.item())
            train_losses.append(sum(t)/len(t)); val_losses.append(sum(v)/len(v) if v else 0)
        gap = val_losses[-1] - train_losses[-1]
        print(f"  最后 epoch: train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}  gap={gap:.4f}")
        assert gap > 0, "过拟合 gap 应 > 0 (val 比 train 高)"
        print("✅ 基础通过 — 看到典型过拟合 (val > train)\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】EarlyStopper"); print("=" * 56)
    try:
        stopper = EarlyStopper(patience=3, min_delta=1e-3)
        # 模拟 val_loss: 先降后涨
        sim_val = [3.0, 2.5, 2.0, 1.8, 1.85, 1.9, 1.95, 2.0]
        stop_epoch = None
        for ep, vl in enumerate(sim_val):
            if stopper(vl, ep):
                stop_epoch = ep; break
        print(f"  模拟 val_loss = {sim_val}")
        print(f"  early stop @ epoch={stop_epoch}, best_epoch={stopper.best_epoch}, best_val={stopper.best:.3f}")
        assert stop_epoch is not None and stop_epoch <= 6
        assert stopper.best_epoch == 3  # 1.8 是最低点
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


EX3 = '''# ============================================================
# 练习 3 | SFT Loss Masking：单轮 → 多轮对话
# ============================================================
#
# 【基础】(人人必做，5 min)
#   create_sft_labels: prompt 部分设 -100 (不计 loss)，response 部分保留 token
#
# 【进阶】(技术学员选做，10 min)
#   create_multiturn_labels: 多轮对话 (user/assistant 交替)，只 mask user 部分
#   接收 list[(role, token_ids)]，返回 input_ids + labels (user 部分 -100)
# ============================================================

def create_sft_labels(input_ids, assistant_start_idx):
    """【基础】单轮 SFT：prompt 全 -100，response 保留 token"""
    # ↓↓↓ 【基础】填空（1 行）↓↓↓
    labels = [-100] * assistant_start_idx + input_ids[assistant_start_idx:]
    # ↑↑↑ 【基础】结束 ↑↑↑
    return labels


def create_multiturn_labels(turns):
    """【进阶】多轮对话 mask
    Args:
        turns: list of (role, token_ids), role ∈ {'user', 'assistant'}
    Returns:
        (input_ids, labels) — labels 中 user 部分为 -100
    """
    # ↓↓↓ 【进阶】填空（约 6 行）↓↓↓
    input_ids, labels = [], []
    for role, ids in turns:
        input_ids.extend(ids)
        if role == 'assistant':
            labels.extend(ids)
        else:
            labels.extend([-100] * len(ids))
    return input_ids, labels
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】单轮 SFT mask"); print("=" * 56)
    try:
        ids = [101, 102, 103, 104, 105]
        labels = create_sft_labels(ids, assistant_start_idx=2)
        print(f"  input_ids = {ids}")
        print(f"  labels    = {labels}")
        assert labels == [-100, -100, 103, 104, 105]
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】多轮对话 mask"); print("=" * 56)
    try:
        turns = [
            ('user', [10, 11, 12]),
            ('assistant', [20, 21]),
            ('user', [13, 14]),
            ('assistant', [22, 23, 24]),
        ]
        input_ids, labels = create_multiturn_labels(turns)
        print(f"  input_ids = {input_ids}")
        print(f"  labels    = {labels}")
        # user1 (3) + assistant1 (2) + user2 (2) + assistant2 (3) = 10 tokens
        assert input_ids == [10, 11, 12, 20, 21, 13, 14, 22, 23, 24]
        assert labels == [-100, -100, -100, 20, 21, -100, -100, 22, 23, 24]
        print("  ✓ user 部分全 -100，assistant 部分保留")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


EX4 = '''# ============================================================
# 练习 4 | SFT 错误分析：分类准确率 + 混淆模式
# ============================================================
#
# 【基础】(人人必做，10 min)
#   按类别统计准确率 + 找出最常见的混淆 (true_label, predicted_label)
#
# 【进阶】(技术学员选做，10 min)
#   找出 base + sft 都错的"硬样本"，按它们的真实 label 分布做归因
#   （这告诉你下一轮 SFT 应该补哪些类别的训练数据）
# ============================================================
from collections import defaultdict, Counter


def analyze_errors(results, all_labels):
    """【基础】返回 (per_category_acc dict, top_confusions list)"""
    # ↓↓↓ 【基础】填空（约 8 行）↓↓↓
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    confusion = Counter()
    for r in results:
        cat_total[r["true"]] += 1
        if r["true"] == r["pred"]:
            cat_correct[r["true"]] += 1
        else:
            confusion[(r["true"], r["pred"])] += 1
    cat_acc = {l: cat_correct[l] / max(cat_total[l], 1) * 100 for l in all_labels}
    return cat_acc, confusion.most_common()
    # ↑↑↑ 【基础】结束 ↑↑↑


def find_hard_samples(base_results, sft_results):
    """【进阶】找 base + sft 都错的样本，按真实 label 统计 → 哪些类别最难"""
    # ↓↓↓ 【进阶】填空（约 6 行）↓↓↓
    hard_by_label = Counter()
    for b, s in zip(base_results, sft_results):
        if b["true"] != b["pred"] and s["true"] != s["pred"]:
            hard_by_label[b["true"]] += 1
    return hard_by_label
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】per-category accuracy + confusion"); print("=" * 56)
    try:
        cat_acc, top_conf = analyze_errors(sft_results, all_labels)
        print("  per-category accuracy:")
        for l, a in sorted(cat_acc.items(), key=lambda x: -x[1]):
            print(f"    {l:<10} {a:5.1f}%")
        print(f"  top 混淆: {top_conf[:3]}")
        assert all(0 <= a <= 100 for a in cat_acc.values())
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】Hard samples → 下轮训练补什么"); print("=" * 56)
    try:
        hard = find_hard_samples(base_results, sft_results)
        print("  base+SFT 都错的样本，按真实类别分布:")
        for label, n in hard.most_common():
            print(f"    {label:<10} {n} 个")
        print("  💡 这些类别在下一轮 SFT 应优先增样")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


EX5 = '''# ============================================================
# 练习 5 | LR 调参 + Cosine Schedule
# ============================================================
#
# 【基础】(人人必做，10 min)
#   测试 3 个 lr 在小步数 (50 步) 的最终 loss
#   提示：复用现有 train_loader_sft，每个 lr 重新初始化模型
#
# 【进阶】(技术学员选做，10 min)
#   实现 cosine_lr_schedule(step, total, base_lr, warmup=10)：
#   warmup 阶段线性升到 base_lr，之后 cosine 衰减到 0
# ============================================================
import math, warnings

QUICK_STEPS = 50
lr_candidates = [1e-5, 5e-5, 2e-4]


def train_with_lr(test_lr, n_steps=QUICK_STEPS):
    """【基础】用指定 lr 训练 n_steps 步，返回最终 loss"""
    # ↓↓↓ 【基础】填空（约 12 行）↓↓↓
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    m.config.pad_token_id = tokenizer.pad_token_id
    opt = torch.optim.AdamW(m.parameters(), lr=test_lr, weight_decay=0.01)
    m.train()
    last_loss = None
    step = 0
    for b in train_loader_sft:
        if step >= n_steps: break
        ids = b["input_ids"].to(device); mk = b["attention_mask"].to(device); y = b["labels"].to(device)
        out = m(input_ids=ids, attention_mask=mk, labels=y)
        opt.zero_grad(); out.loss.backward(); opt.step()
        last_loss = out.loss.item()
        step += 1
    del m
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return last_loss
    # ↑↑↑ 【基础】结束 ↑↑↑


def cosine_lr_schedule(step, total, base_lr, warmup=10):
    """【进阶】warmup + cosine 衰减"""
    # ↓↓↓ 【进阶】填空（约 4 行）↓↓↓
    if step < warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】3 个 lr 在 50 步的对比"); print("=" * 56)
    try:
        results = {}
        for lr in lr_candidates:
            results[lr] = train_with_lr(lr, n_steps=QUICK_STEPS)
            print(f"  lr={lr:>7.0e}  →  final loss = {results[lr]:.4f}")
        best_lr = min(results, key=results.get)
        print(f"  最佳: lr={best_lr:.0e}  loss={results[best_lr]:.4f}")
        assert all(isinstance(v, float) for v in results.values())
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】Cosine schedule (warmup + decay)"); print("=" * 56)
    try:
        total = 100; base = 5e-5; warmup = 10
        lrs = [cosine_lr_schedule(s, total, base, warmup) for s in range(total)]
        # warmup 阶段单调上升
        assert all(lrs[i] <= lrs[i+1] for i in range(warmup-1))
        # peak 在 warmup 处
        assert abs(lrs[warmup] - base) < 1e-9
        # 末端接近 0
        assert lrs[-1] < base * 0.05
        print(f"  step=0:  lr={lrs[0]:.2e}  (warmup 起点)")
        print(f"  step={warmup}: lr={lrs[warmup]:.2e}  (峰值 = base_lr)")
        print(f"  step={total//2}: lr={lrs[total//2]:.2e}  (cosine 衰减中)")
        print(f"  step={total-1}: lr={lrs[-1]:.2e}  (接近 0)")
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 3))
            plt.plot(lrs); plt.axvline(warmup, color='r', linestyle='--', label='warmup end')
            plt.xlabel('step'); plt.ylabel('lr'); plt.title('Cosine LR Schedule with Warmup')
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
        except Exception:
            pass
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


# Mini-Project: 保留原有结构（已经填好了答案），只是加上 verify 和 进阶 提示
MINI = '''# ============================================================
# Mini-Project | SFT 部署决策备忘录 + 灰度发布
# ============================================================
#
# 【基础】(人人必做，10 min)
#   填写部署备忘录 6 个字段（已基于前几个练习计算的实际数字）
#
# 【进阶】(技术学员选做，10 min)
#   补一个 rollout_plan(base_acc, sft_acc, threshold=2.0) → 灰度建议
#   返回 (建议%, 理由)：提升 < threshold% 不上；2-5% 灰度 10%；> 5% 灰度 50%
# ============================================================

base_acc_value = sum(1 for r in base_results if r['true']==r['pred']) / len(base_results) * 100
sft_acc_value = sum(1 for r in sft_results if r['true']==r['pred']) / len(sft_results) * 100
improvement = sft_acc_value - base_acc_value

improvements_dict = {l: sft_acc_dict.get(l, 0) - base_acc_dict.get(l, 0) for l in LABELS}
best_improved_label = max(improvements_dict, key=improvements_dict.get)
worst_sft_label = min(sft_acc_dict, key=sft_acc_dict.get)
top_confusion = f"{sft_conf[0][0][0]} -> {sft_conf[0][0][1]} ({sft_conf[0][1]}次)" if sft_conf else "无明显混淆"
model_size_mb = sum(p.numel() for p in sft_model.parameters()) * 4 / 1024 / 1024
gpu_mem_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0


def make_memo():
    """【基础】生成部署决策备忘录"""
    # ↓↓↓ 【基础】填空（约 20 行模板）↓↓↓
    memo = f"""
{'='*60}
SFT 部署决策备忘录
{'='*60}

1. 准确率对比
   - Base 模型准确率:  {base_acc_value:.1f}%
   - SFT 模型准确率:   {sft_acc_value:.1f}%
   - 提升幅度:          {improvement:+.1f}%

2. 错误分析
   - SFT 改善最大的类别: {best_improved_label}
   - SFT 仍然较弱的类别: {worst_sft_label}
   - 最常见的混淆模式:    {top_confusion}

3. 训练资源
   - 训练数据量:     {len(TRAIN_RECORDS)} 条
   - 训练时间:       约 {sft_total_time/60:.1f} 分钟
   - 显存占用:       约 {gpu_mem_gb:.1f} GB

4. 部署考虑
   - 模型大小:       {model_size_mb:.0f} MB
   - 最低硬件:       GPU 4GB+ 或 CPU（较慢）

5. 建议
   {'部署 SFT 模型' if improvement > 2 else '继续用 Base+Prompt'}: 提升 {improvement:+.1f}%

6. 下一步优化方向
   - 增加 {worst_sft_label} 类别的训练样本
   - 尝试 LoRA 等参数高效微调（下午会做）
{'='*60}
"""
    return memo
    # ↑↑↑ 【基础】结束 ↑↑↑


def rollout_plan(base_acc, sft_acc, threshold=2.0):
    """【进阶】根据准确率提升给灰度建议"""
    # ↓↓↓ 【进阶】填空（约 8 行）↓↓↓
    delta = sft_acc - base_acc
    if delta < threshold:
        return (0, f"提升 {delta:+.1f}% < {threshold}% 阈值，建议不上线，先补数据再训")
    if delta < 5.0:
        return (10, f"提升 {delta:+.1f}% 在 {threshold}-5% 之间，建议灰度 10% 流量观察 1 周")
    if delta < 10.0:
        return (50, f"提升 {delta:+.1f}% 在 5-10%，建议灰度 50% 流量")
    return (100, f"提升 {delta:+.1f}% > 10%，可全量上线（仍建议先 24h 灰度）")
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】部署备忘录"); print("=" * 56)
    try:
        memo = make_memo()
        print(memo)
        assert "Base 模型准确率" in memo
        assert "SFT 模型准确率" in memo
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】灰度发布建议"); print("=" * 56)
    try:
        for delta in [0.5, 3.0, 7.0, 15.0]:
            pct, reason = rollout_plan(80.0, 80.0 + delta)
            print(f"  提升 {delta:+.1f}% → 灰度 {pct}%: {reason}")
        actual_pct, actual_reason = rollout_plan(base_acc_value, sft_acc_value)
        print(f"\\n  ▶ 本次实际：灰度 {actual_pct}%  | {actual_reason}")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


REWRITES = {
    "练习 1：修改训练数据": EX1,
    "练习 2：分析 Loss 曲线": EX2,
    "练习 3：实现 SFT Loss Masking": EX3,
    "练习 4：SFT 错误分析": EX4,
    "练习 5：调超参观察效果": EX5,
    "Mini-Project：SFT 部署决策备忘录": MINI,
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
            if marker in first or marker.split("：")[1].split("(")[0] in first:
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
