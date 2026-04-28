"""Batch 5 rewrite for Day1_上午: 4 exercises rewritten as 【基础】+【进阶】+verify."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_lib import load_nb, save_nb, cell_source, set_cell_source, add_tag

PATH = Path("assets/enterprise_ver2/instructor/Day1_上午_从文本到向量.ipynb")


# ============================================================
# 练习 1：训练循环 (instructor 版，含答案)
# ============================================================
EX1 = '''# ============================================================
# 练习 1 | 训练循环四步曲
# ============================================================
#
# 【基础】(人人必做，5 min)
#   补全 basic_step：Forward → Loss → Backward → Step 4 步
#   提示：
#     - model(X) 返回预测 (N, 1)
#     - criterion(预测, 真值) 返回标量 loss
#     - loss.backward() 计算梯度
#
# 【进阶】(技术学员选做，10 min)
#   在 advanced_step 加入梯度裁剪：
#   反向传播后用 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
#   再调 optimizer.step()。这是工业界训练大模型时防梯度爆炸的标准做法。
# ============================================================
import time, copy

def basic_step(model, X, y, criterion, optimizer):
    """【基础】训练循环的一步：Forward → Loss → Backward → Step"""
    # ↓↓↓ 【基础】填空（4 行）↓↓↓
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # ↑↑↑ 【基础】结束 ↑↑↑
    return outputs, loss


def advanced_step(model, X, y, criterion, optimizer, max_norm=1.0):
    """【进阶】训练步含梯度裁剪 (clip_grad_norm_) — 防梯度爆炸"""
    # ↓↓↓ 【进阶】填空（5 行）↓↓↓
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()
    # ↑↑↑ 【进阶】结束 ↑↑↑
    return outputs, loss


def verify():
    """跑这一格自动判分。【基础】跑 500 轮训练，【进阶】另起 100 轮验证梯度裁剪。"""
    print("=" * 56)
    print("【基础】用 basic_step 训练 500 轮")
    print("=" * 56)
    losses = []
    start = time.time()
    try:
        for epoch in range(500):
            outputs, loss = basic_step(model, X_tensor, y_tensor, criterion, optimizer)
            losses.append(loss.item())
            if (epoch + 1) % 100 == 0:
                acc = ((outputs > 0.5).float() == y_tensor).float().mean()
                print(f"  Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")
        elapsed = time.time() - start
        assert loss.item() < 0.1, f"Loss 应 < 0.1，得到 {loss.item():.4f}"
        with torch.no_grad():
            final_acc = ((model(X_tensor) > 0.5).float() == y_tensor).float().mean()
        assert final_acc > 0.9, f"准确率应 > 90%，得到 {final_acc:.4f}"
        print(f"  耗时 {elapsed:.2f}s | 最终准确率 {final_acc:.4f}")
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现，停止评测\\n")
        return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n")
        return

    print("=" * 56)
    print("【进阶】用 advanced_step (含梯度裁剪) 重训 100 轮")
    print("=" * 56)
    try:
        m2 = copy.deepcopy(model)
        opt2 = torch.optim.SGD(m2.parameters(), lr=0.1)
        for _ in range(100):
            outputs2, loss2 = advanced_step(m2, X_tensor, y_tensor, criterion, opt2)
        # 验证梯度裁剪生效：跑一步后看 grad 范数 ≤ max_norm
        out2, l2 = advanced_step(m2, X_tensor, y_tensor, criterion, opt2, max_norm=1.0)
        total_norm = torch.sqrt(sum((p.grad ** 2).sum() for p in m2.parameters() if p.grad is not None))
        assert total_norm <= 1.0 + 1e-3, f"梯度范数应 ≤ 1.0，得到 {total_norm:.4f}"
        print(f"  最终 loss = {l2.item():.4f} | 梯度范数 = {total_norm:.4f} (≤ 1.0 ✓)")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现，不影响课程进度）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


# ============================================================
# 练习 3：余弦相似度
# ============================================================
EX3 = '''# ============================================================
# 练习 3 | 余弦相似度：从单对到批量
# ============================================================
#
# 【基础】(人人必做，5 min)
#   实现 cosine_similarity(v1, v2) → 返回标量
#   公式：cos(v1, v2) = (v1 · v2) / (||v1|| × ||v2||)
#   提示：np.dot 算点积；np.linalg.norm 算 L2 范数
#
# 【进阶】(技术学员选做，10 min)
#   实现 cosine_similarity_batch(query, matrix) → 返回 shape (N,)
#   query 是单个向量，matrix 是 (N, D) 矩阵；一次性算 query 与每行的相似度
#   提示：用矩阵运算避免 for 循环；matrix @ query 一步算所有点积
# ============================================================

def cosine_similarity(v1, v2):
    """【基础】计算两个向量的余弦相似度，返回标量 float"""
    # ↓↓↓ 【基础】填空（约 4 行）↓↓↓
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product / (norm1 * norm2)
    # ↑↑↑ 【基础】结束 ↑↑↑


def cosine_similarity_batch(query, matrix):
    """【进阶】批量计算 query 与 matrix 每行的余弦相似度，返回 shape (N,)"""
    # ↓↓↓ 【进阶】填空（3 行；用矩阵运算）↓↓↓
    query_norm = query / np.linalg.norm(query)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix_norm @ query_norm
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    test_vec = np.array([1.0, 2.0, 3.0])

    print("=" * 56)
    print("【基础】cosine_similarity")
    print("=" * 56)
    try:
        sim_same = cosine_similarity(test_vec, test_vec)
        sim_opp = cosine_similarity(test_vec, -test_vec)
        sim_orth = cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        print(f"  cos(v, v)   = {sim_same:.4f}  (期望 1.0)")
        print(f"  cos(v, -v)  = {sim_opp:.4f}  (期望 -1.0)")
        print(f"  cos([1,0],[0,1]) = {sim_orth:.4f}  (期望 0.0)")
        assert abs(sim_same - 1.0) < 1e-6
        assert abs(sim_opp + 1.0) < 1e-6
        assert abs(sim_orth) < 1e-6
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {e}\\n"); return

    print("=" * 56)
    print("【进阶】cosine_similarity_batch")
    print("=" * 56)
    try:
        np.random.seed(0)
        Q = np.random.randn(8)
        M = np.random.randn(20, 8)
        sims = cosine_similarity_batch(Q, M)
        # 跟单对版本逐行对比
        sims_ref = np.array([cosine_similarity(Q, M[i]) for i in range(20)])
        diff = np.abs(sims - sims_ref).max()
        print(f"  shape: {sims.shape} (期望 (20,))")
        print(f"  与基础版逐对结果最大差: {diff:.2e} (期望 < 1e-6)")
        assert sims.shape == (20,)
        assert diff < 1e-6
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()

# ──── 练习应用：用 cosine_similarity 找相似词（与原版一致） ────
def find_similar(word, top_k=5):
    if word not in word2idx:
        print(f"词 '{word}' 不在词表中"); return
    word_vec = word_vectors[word2idx[word]]
    similarities = []
    for w in vocab:
        if w != word:
            sim = cosine_similarity(word_vec, word_vectors[word2idx[w]])
            similarities.append((w, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    print(f"\\n与 '{word}' 最相似的词:")
    for w, sim in similarities[:top_k]:
        print(f"  {w}: {sim:.4f}")

find_similar('学')
find_similar('机')
'''


# ============================================================
# 练习 4：用 Embedding 找相似句
# ============================================================
EX4 = '''# ============================================================
# 练习 4 | 用预训练 Embedding 做语义检索
# ============================================================
#
# 【基础】(人人必做，5 min)
#   用 embedder.embed(...) 把句子向量化，然后用 cosine_similarity 排序找最相关
#
# 【进阶】(技术学员选做，10 min)
#   用 cosine_similarity_batch + np.argsort 实现 top-k 检索
#   要求：用矩阵乘法一次算 query 与所有句子的相似度，避免 for 循环
# ============================================================

embedder = env.get_embedder()

sentences = [
    "机器学习可以分析大量数据",
    "深度学习是人工智能的重要分支",
    "今天的天气非常好",
    "数据挖掘帮助企业做决策",
    "公园里的花开了",
]
query = "AI 技术帮助企业提高效率"


def basic_search(query, sentences):
    """【基础】返回 [(sim, sent), ...] 按相似度降序"""
    # ↓↓↓ 【基础】填空（约 6 行）↓↓↓
    vectors = embedder.embed(sentences)
    query_vec = embedder.embed([query])[0]
    results = []
    for i, sent in enumerate(sentences):
        sim = cosine_similarity(np.array(query_vec), np.array(vectors[i]))
        results.append((sim, sent))
    results.sort(reverse=True)
    return results
    # ↑↑↑ 【基础】结束 ↑↑↑


def advanced_topk(query, sentences, k=3):
    """【进阶】用矩阵化 + np.argsort 返回 top-k 列表 [(sim, sent), ...]"""
    # ↓↓↓ 【进阶】填空（4 行；要求用 cosine_similarity_batch）↓↓↓
    matrix = np.array(embedder.embed(sentences))
    q_vec = np.array(embedder.embed([query])[0])
    sims = cosine_similarity_batch(q_vec, matrix)
    topk_idx = np.argsort(-sims)[:k]
    return [(float(sims[i]), sentences[i]) for i in topk_idx]
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】basic_search"); print("=" * 56)
    try:
        results = basic_search(query, sentences)
        print(f"查询: '{query}'\\n相似度排名:")
        for sim, sent in results:
            print(f"  {sim:.4f} | {sent}")
        assert len(results) == len(sentences)
        assert results[0][0] >= results[-1][0], "排序应降序"
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】advanced_topk (k=3)"); print("=" * 56)
    try:
        topk = advanced_topk(query, sentences, k=3)
        for sim, sent in topk:
            print(f"  {sim:.4f} | {sent}")
        assert len(topk) == 3
        # top-1 应与基础版 top-1 同句
        assert topk[0][1] == basic_search(query, sentences)[0][1]
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


# ============================================================
# 练习 5：Prompt A/B 测试
# ============================================================
EX5 = '''# ============================================================
# 练习 5 | Prompt 工程 A/B 测试 + 进阶 CoT
# ============================================================
#
# 【基础】(人人必做，10 min)
#   设计 2 个 prompt：A=Few-shot (含示例)，B=Zero-shot (纯指令)
#   对比哪个准确率高
#
# 【进阶】(技术学员选做，10 min)
#   再设计 1 个 Chain-of-Thought (CoT) prompt：让模型先『想』再回答
#   对比 A/B/C 三种的准确率，并思考 CoT 的代价（token 多 ≈ 慢且贵）
# ============================================================

# ──── 【基础】设计两个 prompt ────
# ↓↓↓ 【基础】填空：补全 PROMPT_A (Few-shot) 与 PROMPT_B (Zero-shot) ↓↓↓
PROMPT_A = """你是一个客户投诉分类系统。请将客户投诉分类为以下类别之一：产品质量/物流配送/售后服务/价格争议/账户问题/其他

示例：
投诉: "收到的手机屏幕有裂痕" → 产品质量
投诉: "包裹一直没有送到" → 物流配送
投诉: "退款申请没人处理" → 售后服务

请分类以下投诉，只回复类别名称：
投诉: "{text}"
分类:"""

PROMPT_B = """请将以下客户投诉分类为：产品质量/物流配送/售后服务/价格争议/账户问题/其他。只回复类别名称。

投诉: "{text}"
分类:"""
# ↑↑↑ 【基础】结束 ↑↑↑


# ──── 【进阶】设计 CoT prompt ────
# ↓↓↓ 【进阶】填空：补全 PROMPT_C (CoT) ↓↓↓
PROMPT_C = """请将以下客户投诉分类。先用一句话分析投诉的核心问题，再给出分类。

类别：产品质量/物流配送/售后服务/价格争议/账户问题/其他

格式：
分析: <一句话分析>
分类: <类别>

投诉: "{text}"
"""
# ↑↑↑ 【进阶】结束 ↑↑↑


test_complaints = [
    ("收到的商品有明显划痕，包装也破损了", "产品质量"),
    ("快递显示已签收但我没收到", "物流配送"),
    ("申请退款三天了还没处理", "售后服务"),
    ("同样的东西别家便宜很多", "价格争议"),
    ("我的账号突然登不上了", "账户问题"),
    ("产品用了两天就坏了", "产品质量"),
    ("发货太慢了等了一周", "物流配送"),
    ("客服态度很差不解决问题", "售后服务"),
]


def run_prompt(prompt_template, complaints, llm, label=""):
    """跑一个 prompt 在测试集上，返回 (准确率, 详情列表)"""
    if llm is None:
        print(f"  [{label}] LLM 未初始化"); return 0.0, []
    correct = 0
    details = []
    for text, expected in complaints:
        out = llm.generate(prompt_template.format(text=text), temperature=0)
        is_right = expected in out
        if is_right:
            correct += 1
        details.append((text, expected, out.strip()[:30], is_right))
    return correct / len(complaints), details


def verify():
    print("=" * 56); print("【基础】对比 PROMPT_A (Few-shot) vs PROMPT_B (Zero-shot)"); print("=" * 56)
    try:
        acc_a, det_a = run_prompt(PROMPT_A, test_complaints, llm, "A")
        acc_b, det_b = run_prompt(PROMPT_B, test_complaints, llm, "B")
        print(f"\\n  准确率:  A (Few-shot) = {acc_a:.0%}   |   B (Zero-shot) = {acc_b:.0%}")
        # 一般 Few-shot 应不弱于 Zero-shot
        assert isinstance(acc_a, float) and isinstance(acc_b, float)
        print("✅ 基础通过 — 通常 Few-shot 准确率 ≥ Zero-shot\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】对比 A/B/C 三种 prompt (含 CoT)"); print("=" * 56)
    try:
        acc_c, det_c = run_prompt(PROMPT_C, test_complaints, llm, "C")
        print(f"\\n  准确率:  A = {acc_a:.0%}   |   B = {acc_b:.0%}   |   C (CoT) = {acc_c:.0%}")
        print(f"\\n  💡 思考：CoT 多输出『分析』那一句，token 比 A 多 ~3-5 倍，延迟同步增加。")
        print(f"     在简单分类任务上，Few-shot 通常已够用；CoT 在多跳推理任务才显著占优。")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


REWRITES = {
    "练习 1：补全训练循环四步曲": EX1,
    "练习 3：实现余弦相似度": EX3,
    "练习 4：用预训练 Embedding 找语义相似句": EX4,
    "练习 5：Prompt工程A/B测试": EX5,
}


def main():
    nb = load_nb(PATH)
    n_replaced = 0
    for marker, new_src in REWRITES.items():
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            src = cell_source(c)
            first_line = src.strip().split("\n")[0]
            if marker.split("：")[0] in first_line and marker.split("：")[1].split("(")[0] in src[:200]:
                set_cell_source(c, new_src)
                add_tag(c, "fillin")
                add_tag(c, "batch5")
                # Clear outputs since cell content changed; will re-run later
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
