"""Batch 5 rewrite for Day1_下午 Transformer: 6 cells (5 exercises + Mini-Project)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_lib import load_nb, save_nb, cell_source, set_cell_source, add_tag

PATH = Path("assets/enterprise_ver2/instructor/Day1_下午_Transformer架构.ipynb")


EX1 = '''# ============================================================
# 练习 1 | 手算 Self-Attention 一步
# ============================================================
#
# 【基础】(人人必做，10 min)
#   按 4 步算 q2 的注意力输出：点积 → 缩放 → softmax → 加权求和
#   提示：用 numpy + torch.softmax；d_k = 64
#
# 【进阶】(技术学员选做，10 min)
#   实现支持 causal mask 的版本 attention_step_causal(...)：
#   把 scores 中『被禁止看的位置』设 -inf 再过 softmax
#   验证：q2 只能看 q1 和 q2 自己，不能看 q3
# ============================================================
import numpy as np
import torch
import torch.nn.functional as F

d_k = 64
x1_val, x2_val = 1.323, 1.134

def basic_attention_step():
    """【基础】计算 q2 attend 到 [k1, k2] 的输出"""
    # ↓↓↓ 【基础】填空（4 步）↓↓↓
    # Step 1: 点积
    q2_k1 = d_k * x2_val * x1_val
    q2_k2 = d_k * x2_val * x2_val
    # Step 2: 缩放
    sqrt_dk = np.sqrt(d_k)
    scaled = torch.tensor([q2_k1 / sqrt_dk, q2_k2 / sqrt_dk])
    # Step 3: softmax
    weights = F.softmax(scaled, dim=-1)
    # Step 4: 加权求和
    z2 = weights[0].item() * x1_val + weights[1].item() * x2_val
    # ↑↑↑ 【基础】结束 ↑↑↑
    return q2_k1, q2_k2, weights, z2


def attention_step_causal(query_idx, x_vals):
    """【进阶】带 causal mask 的注意力一步：query_idx 只能看 [0..query_idx]"""
    # ↓↓↓ 【进阶】填空（约 8 行）↓↓↓
    n = len(x_vals)
    q = x_vals[query_idx]
    scores = torch.tensor([d_k * q * x_vals[i] / np.sqrt(d_k) for i in range(n)])
    # Causal mask: 把 i > query_idx 的位置设 -inf
    mask = torch.arange(n) > query_idx
    scores = scores.masked_fill(mask, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    z = sum(weights[i].item() * x_vals[i] for i in range(n))
    # ↑↑↑ 【进阶】结束 ↑↑↑
    return weights, z


def verify():
    print("=" * 56); print("【基础】手算 q2 → [k1, k2] 注意力"); print("=" * 56)
    try:
        q2_k1, q2_k2, weights, z2 = basic_attention_step()
        print(f"  q2·k1 = {q2_k1:.2f}  (期望 ≈ 96.27)")
        print(f"  q2·k2 = {q2_k2:.2f}  (期望 ≈ 82.31)")
        print(f"  attention weights = [{weights[0]:.3f}, {weights[1]:.3f}]")
        print(f"  z2 = {z2:.4f}  (期望 ≈ 1.295)")
        assert abs(q2_k1 - 96.27) < 0.5
        assert abs(q2_k2 - 82.31) < 0.5
        assert abs(z2 - 1.295) < 0.05
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】带 causal mask 的注意力（q2 不能看 q3）"); print("=" * 56)
    try:
        x_vals = [1.323, 1.134, 0.987]
        w_q2, z_q2 = attention_step_causal(query_idx=1, x_vals=x_vals)
        print(f"  q2 的 attention weights = [{w_q2[0]:.3f}, {w_q2[1]:.3f}, {w_q2[2]:.3f}]")
        print(f"  期望 weights[2] = 0 (causal mask 屏蔽未来)")
        assert abs(w_q2[2].item()) < 1e-6, f"causal mask 未生效，weights[2]={w_q2[2]:.4f}"
        # q3 可以看所有
        w_q3, _ = attention_step_causal(query_idx=2, x_vals=x_vals)
        assert abs(w_q3.sum().item() - 1.0) < 1e-5
        print("✅ 进阶通过 — q2 看不到 q3，q3 能看所有")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


EX2 = '''# ============================================================
# 练习 2 | 注意力模式侦探
# ============================================================
#
# 【基础】(人人必做，10 min)
#   提取每个 head 的 attention weight 矩阵 + 计算注意力熵
#   熵低 = 关注集中（专家型）；熵高 = 关注分散（综合型）
#
# 【进阶】(技术学员选做，10 min)
#   检测 "Attention Sink" 现象：
#   每个 head 是否过度关注第一个 token (常见于 GPT 类模型) ？
#   实现 detect_attention_sink(weights) → 返回每个 head 关注 token 0 的平均比例
# ============================================================
import torch, numpy as np
import torch.nn.functional as F

d_model = 32; n_heads = 4; seq_len = 6
mha = MultiHeadAttention(d_model, n_heads)
torch.manual_seed(42)
input_a = torch.randn(1, seq_len, d_model)
input_b = torch.randn(1, seq_len, d_model) * 2 + 1


def get_attention_weights(mha, x):
    """【基础】提取每个 head 的注意力权重矩阵 (n_heads, T, T)"""
    # ↓↓↓ 【基础】填空（约 6 行）↓↓↓
    B, T, C = x.shape
    d_k = d_model // n_heads
    Q = mha.W_q(x).view(B, T, n_heads, d_k).transpose(1, 2)
    K = mha.W_k(x).view(B, T, n_heads, d_k).transpose(1, 2)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return weights.squeeze(0)
    # ↑↑↑ 【基础】结束 ↑↑↑


def attention_entropy(weights):
    """【基础】计算注意力分布的平均熵 (越小 = 越集中)"""
    # ↓↓↓ 【基础】填空（约 3 行）↓↓↓
    eps = 1e-8
    H = -(weights * torch.log(weights + eps)).sum(dim=-1).mean()
    return H.item()
    # ↑↑↑ 【基础】结束 ↑↑↑


def detect_attention_sink(weights):
    """【进阶】检测每个 head 关注第一个 token 的平均比例 → (n_heads,) numpy 数组"""
    # ↓↓↓ 【进阶】填空（约 3 行）↓↓↓
    # weights shape: (n_heads, T, T) — 每个 query token 的 attention 分布
    # 取每个 head 在所有 query 上对 token 0 的平均权重
    return weights[:, :, 0].mean(dim=-1).cpu().numpy()
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】attention 权重 + 熵"); print("=" * 56)
    try:
        with torch.no_grad():
            wa = get_attention_weights(mha, input_a)
            wb = get_attention_weights(mha, input_b)
        assert wa.shape == (n_heads, seq_len, seq_len), f"权重形状应为 ({n_heads},{seq_len},{seq_len})"
        print(f"  weights shape: {tuple(wa.shape)} ✓")
        print(f"  {'Head':>4} | {'输入A 熵':>9} | {'输入B 熵':>9}")
        for h in range(n_heads):
            ha = attention_entropy(wa[h]); hb = attention_entropy(wb[h])
            print(f"  {h:>4} | {ha:>9.4f} | {hb:>9.4f}")
        h_test = attention_entropy(wa[0])
        assert 0 < h_test < 5, f"熵 {h_test} 不在合理范围"
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】Attention Sink 检测"); print("=" * 56)
    try:
        sink_ratios = detect_attention_sink(wa)
        print(f"  各 head 对 token 0 的平均关注比例: {sink_ratios.round(3)}")
        avg = sink_ratios.mean()
        print(f"  平均 sink 比例: {avg:.3f}  (随机分布期望 ≈ 1/{seq_len} = {1/seq_len:.3f})")
        assert sink_ratios.shape == (n_heads,)
        assert (sink_ratios >= 0).all() and (sink_ratios <= 1).all()
        if avg > 1 / seq_len * 1.5:
            print("  💡 检测到 attention sink — 此模型部分 head 过度依赖首 token")
        else:
            print("  💡 sink 不明显 — 模型未显著偏向首 token")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


EX3 = '''# ============================================================
# 练习 3 | TransformerBlock 拼装：Pre-LN vs Post-LN
# ============================================================
#
# 【基础】(人人必做，10 min)
#   拼 Pre-LN 版 (现代主流：LN→Attn→Residual→LN→FFN→Residual)
#   提示：用 nn.LayerNorm + MultiHeadAttentionV2 + FeedForward
#
# 【进阶】(技术学员选做，10 min)
#   实现 Post-LN 变体 (原版 Transformer：Attn→Residual→LN→FFN→Residual→LN)
#   现场观察两者输出差异 + 思考为何现代 LLM 都用 Pre-LN（训练更稳，不需要 warmup）
# ============================================================

class PreLNBlock(nn.Module):
    """【基础】Pre-LN：先归一化，再算 attn / ffn，最后残差"""
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        # ↓↓↓ 【基础】填空（5 行）↓↓↓
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttentionV2(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        # ↑↑↑ 【基础】结束 ↑↑↑

    def forward(self, x, mask=None):
        # ↓↓↓ 【基础】填空（2 行）↓↓↓
        x = x + self.dropout(self.attention(self.ln1(x), mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        # ↑↑↑ 【基础】结束 ↑↑↑
        return x


class PostLNBlock(nn.Module):
    """【进阶】Post-LN：原版 Transformer 顺序"""
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        # ↓↓↓ 【进阶】填空（5 行）↓↓↓
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttentionV2(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        # ↑↑↑ 【进阶】结束 ↑↑↑

    def forward(self, x, mask=None):
        # ↓↓↓ 【进阶】填空（2 行；先 attn+residual 再 ln，再 ffn+residual 再 ln）↓↓↓
        x = self.ln1(x + self.dropout(self.attention(x, mask)))
        x = self.ln2(x + self.dropout(self.ffn(x)))
        # ↑↑↑ 【进阶】结束 ↑↑↑
        return x


def verify():
    print("=" * 56); print("【基础】Pre-LN Block"); print("=" * 56)
    try:
        block = PreLNBlock(d_model=64, n_heads=8, d_ff=256)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == torch.Size([2, 10, 64]), f"形状错: {out.shape}"
        n_params = sum(p.numel() for p in block.parameters())
        print(f"  Pre-LN 输出 shape: {tuple(out.shape)}  |  参数量: {n_params:,}")
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】Post-LN Block + 与 Pre-LN 对比"); print("=" * 56)
    try:
        post = PostLNBlock(d_model=64, n_heads=8, d_ff=256)
        x = torch.randn(2, 10, 64)
        out_post = post(x)
        out_pre = block(x)
        diff = (out_pre - out_post).abs().mean().item()
        print(f"  Post-LN 输出 shape: {tuple(out_post.shape)}")
        print(f"  Pre-LN vs Post-LN 输出平均差: {diff:.4f}")
        print(f"  💡 现代 LLM (GPT-2+, LLaMA) 都用 Pre-LN — 梯度更稳，不需 warmup")
        assert out_post.shape == out_pre.shape
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


EX4 = '''# ============================================================
# 练习 4 | 采样策略：Top-k 与 Top-p (nucleus)
# ============================================================
#
# 【基础】(人人必做，10 min)
#   实现 temperature scaling + top-k filter
#   提示：用 torch.topk 找前 k 个，其余 logits 设 -inf
#
# 【进阶】(技术学员选做，15 min)
#   实现 top-p (nucleus) sampling：
#   累积概率排序后，保留累积 ≤ p 的最小集合，其余设 -inf
#   提示：torch.sort + torch.cumsum + 创建 mask
# ============================================================
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

logits = torch.tensor([2.0, 1.0, 0.5, 0.1, -1.0, -2.0])
vocab = ["好", "的", "很", "不", "了", "吗"]


def my_temperature_scale(logits, temperature):
    """【基础】温度缩放：< 1 更确定，> 1 更随机"""
    # ↓↓↓ 【基础】填空（1 行）↓↓↓
    return logits / temperature
    # ↑↑↑ 【基础】结束 ↑↑↑


def my_top_k_filter(logits, k):
    """【基础】Top-k：保留前 k 高 logit，其余设 -inf"""
    # ↓↓↓ 【基础】填空（约 4 行）↓↓↓
    top_values, _ = torch.topk(logits, k)
    threshold = top_values[-1]
    logits = logits.clone()
    logits[logits < threshold] = float('-inf')
    return logits
    # ↑↑↑ 【基础】结束 ↑↑↑


def my_top_p_filter(logits, p):
    """【进阶】Top-p (nucleus)：累积概率到 p 为止保留，其余设 -inf"""
    # ↓↓↓ 【进阶】填空（约 8 行）↓↓↓
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # mask: 累积概率 > p 的丢掉（保留至少 1 个 token）
    drop_mask = cumsum > p
    drop_mask[..., 1:] = drop_mask[..., :-1].clone()
    drop_mask[..., 0] = False
    out = logits.clone()
    out[sorted_idx[drop_mask]] = float('-inf')
    return out
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】Temperature + Top-k"); print("=" * 56)
    try:
        scaled = my_temperature_scale(logits.clone(), 2.0)
        assert torch.allclose(scaled, logits / 2.0)
        filtered = my_top_k_filter(torch.tensor([3.0, 1.0, 2.0, 0.5, -1.0]), 2)
        assert (filtered == float('-inf')).sum() == 3, "Top-2 应留 2 个，过滤 3 个"
        # 可视化（不影响 verify）
        configs = [{"name": "保守 (T=0.5,k=3)", "T":0.5, "k":3},
                   {"name": "随机 (T=2.0,k=6)", "T":2.0, "k":6},
                   {"name": "平衡 (T=1.0,k=2)", "T":1.0, "k":2}]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i, cfg in enumerate(configs):
            f = my_top_k_filter(my_temperature_scale(logits.clone(), cfg["T"]), cfg["k"])
            axes[i].bar(vocab, F.softmax(f, dim=-1).numpy())
            axes[i].set_title(cfg["name"]); axes[i].set_ylim(0,1)
        plt.tight_layout(); plt.show()
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】Top-p (nucleus)"); print("=" * 56)
    try:
        # p=0.5 应该比 p=0.95 更激进地裁剪
        out_p50 = my_top_p_filter(logits.clone(), 0.5)
        out_p95 = my_top_p_filter(logits.clone(), 0.95)
        kept_50 = (out_p50 != float('-inf')).sum().item()
        kept_95 = (out_p95 != float('-inf')).sum().item()
        print(f"  p=0.5 保留 token 数: {kept_50}")
        print(f"  p=0.95 保留 token 数: {kept_95}")
        assert kept_50 <= kept_95, "p 越大应保留越多"
        assert kept_50 >= 1, "至少保留 1 个 token"
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i, p in enumerate([0.5, 0.95]):
            f = my_top_p_filter(logits.clone(), p)
            axes[i].bar(vocab, F.softmax(f, dim=-1).numpy())
            axes[i].set_title(f"top-p (p={p})"); axes[i].set_ylim(0,1)
        plt.tight_layout(); plt.show()
        print(f"  💡 实际生产 (GPT-4 / Claude) 都用 top-p ≈ 0.9-0.95，比 top-k 更适应分布形状")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


EX5 = '''# ============================================================
# 练习 5 | 参数量估算器 + KV Cache 估算
# ============================================================
#
# 【基础】(人人必做，10 min)
#   纯公式估算 GPT 参数量（不用真创建模型）
#   组件：token_emb + pos_emb + n_layer × (attn + ffn + ln) + final_ln
#
# 【进阶】(技术学员选做，10 min)
#   实现 estimate_kv_cache_gb(...)：估算推理时 KV cache 占用显存
#   公式：2 × n_layer × seq_len × n_embd × precision_bytes × batch
#   这是大模型推理 OOM 的最常见原因（比模型权重还大）
# ============================================================

def estimate_params(vocab_size, n_layer, n_head, n_embd, block_size):
    """【基础】纯公式计算 GPT 参数量"""
    # ↓↓↓ 【基础】填空（约 8 行）↓↓↓
    token_emb = vocab_size * n_embd
    pos_emb = block_size * n_embd
    per_layer_attn = 4 * n_embd ** 2 + 4 * n_embd
    per_layer_ffn = 8 * n_embd ** 2 + 5 * n_embd
    per_layer_ln = 4 * n_embd
    final_ln = 2 * n_embd
    total = token_emb + pos_emb + n_layer * (per_layer_attn + per_layer_ffn + per_layer_ln) + final_ln
    return {"token_emb": token_emb, "pos_emb": pos_emb,
            "per_layer_attn": per_layer_attn, "per_layer_ffn": per_layer_ffn,
            "per_layer_ln": per_layer_ln, "final_ln": final_ln, "total": total}
    # ↑↑↑ 【基础】结束 ↑↑↑


def estimate_kv_cache_gb(n_layer, n_embd, seq_len, batch=1, precision_bytes=2):
    """【进阶】估算 KV cache 显存 (GB)
    KV cache = 2 (K和V各一份) × n_layer × seq_len × n_embd × bytes × batch
    """
    # ↓↓↓ 【进阶】填空（约 2 行）↓↓↓
    bytes_total = 2 * n_layer * seq_len * n_embd * precision_bytes * batch
    return bytes_total / (1024 ** 3)
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】estimate_params"); print("=" * 56)
    try:
        small = estimate_params(vocab_size=1503, n_layer=4, n_head=4, n_embd=128, block_size=64)
        print("  小 GPT 各组件参数量:")
        for k, v in small.items():
            print(f"    {k}: {v:,}")
        llama = estimate_params(vocab_size=32000, n_layer=32, n_head=32, n_embd=4096, block_size=2048)
        print(f"\\n  🔮 LLaMA-7B 估算: {llama['total']/1e9:.2f}B 参数 (实际 6.7B，误差 < 5%)")
        assert 6e9 < llama['total'] < 8e9, "估算量级应在 7B 范围"
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】estimate_kv_cache_gb"); print("=" * 56)
    try:
        # LLaMA-7B FP16, seq_len=2048, batch=1
        kv_2k = estimate_kv_cache_gb(n_layer=32, n_embd=4096, seq_len=2048, batch=1, precision_bytes=2)
        kv_8k = estimate_kv_cache_gb(n_layer=32, n_embd=4096, seq_len=8192, batch=1, precision_bytes=2)
        kv_8k_b16 = estimate_kv_cache_gb(n_layer=32, n_embd=4096, seq_len=8192, batch=16, precision_bytes=2)
        print(f"  LLaMA-7B FP16 @ seq_len=2048 batch=1:  {kv_2k:.2f} GB")
        print(f"  LLaMA-7B FP16 @ seq_len=8192 batch=1:  {kv_8k:.2f} GB")
        print(f"  LLaMA-7B FP16 @ seq_len=8192 batch=16: {kv_8k_b16:.2f} GB")
        print(f"\\n  💡 模型权重约 14 GB；当 batch×seq_len 增大，KV cache 反而变成主要开销")
        print(f"     这是为何要 PagedAttention / vLLM / FlashAttention 的根本原因")
        assert 0.5 < kv_2k < 2.0, f"2k cache 应约 1GB 量级，得到 {kv_2k}"
        assert kv_8k > kv_2k * 3
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


MINI = '''# ============================================================
# Mini-Project | 企业场景选生成策略
# ============================================================
#
# 【基础】(人人必做，10 min)
#   3 个企业场景：客服、营销文案、代码补全。各填好/差参数 + 理由
#   原则：准确性场景用低 T + 小 k；创意场景用高 T + 大 k
#
# 【进阶】(技术学员选做，15 min)
#   实现 suggest_params(use_case_keywords) 自动选参数 + 加第 4 个场景"诗歌创作"
#   要求 keywords 含『准确/合规/事实』时返回低温配置；含『创意/营销/诗歌』时返回高温配置
# ============================================================

# ──── 【基础】3 场景标准答案 ────
# ↓↓↓ 【基础】填空 ↓↓↓
scenarios = {
    "客服Bot": {
        "prompt": "客户问：你们的退货政策是什么？\\n回答：",
        "good_params": {"temperature": 0.3, "top_k": 5},
        "bad_params": {"temperature": 1.5, "top_k": 50},
        "理由": "客服回答需准确一致；低 T + 小 k 确保稳定可靠",
    },
    "营销文案": {
        "prompt": "为一款智能手表写一句广告语：",
        "good_params": {"temperature": 1.0, "top_k": 40},
        "bad_params": {"temperature": 0.1, "top_k": 2},
        "理由": "营销需要创意多样；高 T + 大 k 鼓励新颖表达",
    },
    "代码补全": {
        "prompt": "def fibonacci(n):\\n    # 返回第n个斐波那契数\\n",
        "good_params": {"temperature": 0.2, "top_k": 5},
        "bad_params": {"temperature": 1.5, "top_k": 50},
        "理由": "代码需精确合理；低 T + 小 k 确保语法正确",
    },
}
# ↑↑↑ 【基础】结束 ↑↑↑


# ──── 【进阶】自动选参数 + 第 4 场景 ────
def suggest_params(use_case_keywords):
    """【进阶】根据关键词自动建议参数"""
    # ↓↓↓ 【进阶】填空（约 8 行）↓↓↓
    text = use_case_keywords.lower()
    accuracy_kw = ["准确", "合规", "事实", "代码", "客服", "fact"]
    creative_kw = ["创意", "营销", "诗歌", "文案", "故事"]
    if any(k in text for k in accuracy_kw):
        return {"temperature": 0.3, "top_k": 5, "原因": "关键词偏准确性 → 低温"}
    if any(k in text for k in creative_kw):
        return {"temperature": 1.0, "top_k": 40, "原因": "关键词偏创意 → 高温"}
    return {"temperature": 0.7, "top_k": 20, "原因": "默认平衡配置"}

scenarios_advanced = dict(scenarios)
scenarios_advanced["诗歌创作"] = {
    "prompt": "写一首关于秋天的现代诗，4 行：\\n",
    "good_params": suggest_params("诗歌 创意"),
    "bad_params": {"temperature": 0.1, "top_k": 2},
    "理由": "诗歌需要意象多样 + 跳跃；suggest_params 自动给出高温配置",
}
# ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】3 场景选参数对比"); print("=" * 56)
    try:
        assert len(scenarios) == 3
        for name, s in scenarios.items():
            assert "good_params" in s and "bad_params" in s and "理由" in s
            assert s["good_params"] != s["bad_params"]
            print(f"  ✓ {name}: good={s['good_params']}  理由={s['理由'][:40]}...")
        # 用 gpt 模型快速演示一个场景（避免训练，只跑一次）
        if 'gpt' in dir():
            model = gpt
            scene = scenarios["客服Bot"]
            idx = torch.randint(0, config['vocab_size'], (1, 5))
            text = generate(model, idx.clone(), max_new_tokens=10, **scene["good_params"])
            print(f"  ▸ 客服Bot good_params 试运行 OK (输出 token: {text[0].tolist()[:6]}...)")
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】suggest_params 自动选参数"); print("=" * 56)
    try:
        case_a = suggest_params("准确客服回答")
        case_b = suggest_params("营销文案创意")
        case_c = suggest_params("一般问题")
        print(f"  '准确客服回答' → T={case_a['temperature']}, k={case_a['top_k']}  ({case_a['原因']})")
        print(f"  '营销文案创意' → T={case_b['temperature']}, k={case_b['top_k']}  ({case_b['原因']})")
        print(f"  '一般问题' → T={case_c['temperature']}, k={case_c['top_k']}  ({case_c['原因']})")
        assert case_a["temperature"] < case_b["temperature"]
        assert "诗歌创作" in scenarios_advanced
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


REWRITES = {
    "练习 1：手算并验证 q2 的注意力输出": EX1,
    "练习 2：注意力模式侦探": EX2,
    "练习 3：从组件拼装 TransformerBlock": EX3,
    "练习 4：自己实现 Top-k 采样": EX4,
    "练习 5：参数量估算器": EX5,
    "Mini-Project：企业场景选生成策略": MINI,
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
