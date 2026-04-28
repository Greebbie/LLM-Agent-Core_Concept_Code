"""Batch 5 rewrite for Day3_下午 Capstone: 4 Checkpoints (integration)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_lib import load_nb, save_nb, cell_source, set_cell_source, add_tag

PATH = Path("assets/enterprise_ver2/instructor/Day3_下午_企业知识助手Capstone.ipynb")


CP1 = '''# ============================================================
# Checkpoint 1 | 知识库检索 + 元数据过滤
# ============================================================
#
# 【基础】(人人必做，10 min)
#   实现 search_knowledge：调 vector_store.search + threshold 过滤
#
# 【进阶】(技术学员选做，10 min)
#   实现 search_knowledge_filtered(query, top_k, threshold, category=None)：
#   只返回指定 category 的结果（如只搜『hr_policy』类）
#   生产场景常用：基于用户身份过滤（财务只能搜财务知识库等）
# ============================================================
from typing import List, Dict, Optional


def search_knowledge(query: str, top_k: int = 3, threshold: float = 0.3) -> List[Dict]:
    """【基础】返回得分 ≥ threshold 的 top-k 结果"""
    # ↓↓↓ 【基础】填空（约 2 行）↓↓↓
    results = vector_store.search(query, top_k=top_k, threshold=threshold)
    return [r for r in results if r["score"] >= threshold]
    # ↑↑↑ 【基础】结束 ↑↑↑


def search_knowledge_filtered(
    query: str, top_k: int = 3, threshold: float = 0.3,
    category: Optional[str] = None,
) -> List[Dict]:
    """【进阶】带 category 过滤；不过滤时与基础版等价
    metadata 假设含 'category' 字段
    """
    # ↓↓↓ 【进阶】填空（约 6 行）↓↓↓
    # 先取宽一点，过滤后再截 top_k
    candidates = vector_store.search(query, top_k=top_k * 3, threshold=threshold)
    filtered = [r for r in candidates if r["score"] >= threshold]
    if category:
        filtered = [r for r in filtered if r["metadata"].get("category") == category]
    return filtered[:top_k]
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】search_knowledge"); print("=" * 56)
    try:
        test_queries = [
            "年假有几天？",
            "StarLink 基础版的价格是多少？",
            "API 认证用什么方式？",
            "今天天气怎么样？",
        ]
        ok = 0
        for q in test_queries:
            results = search_knowledge(q)
            print(f"\\n[查询] {q}")
            if results:
                ok += 1
                for r in results:
                    print(f"  [{r['score']:.3f}] [{r['metadata'].get('category', '?')}] {r['document'][:60]}...")
            else:
                print("  -> 未找到（OOD 兜底正常）")
        assert ok >= 3, f"应至少 3 个查询有结果，得到 {ok}"
        print("\\n✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】search_knowledge_filtered (按 category 过滤)"); print("=" * 56)
    try:
        # 收集所有 category 看有哪些
        all_results = vector_store.search("公司", top_k=20, threshold=0.0)
        cats = set(r["metadata"].get("category", "?") for r in all_results)
        print(f"  知识库中的 category: {cats}")

        # 拿一个真实存在的 category 测过滤
        target_cat = next((c for c in cats if c and c != "?"), None)
        if target_cat:
            filtered = search_knowledge_filtered("公司", top_k=5, category=target_cat)
            unfiltered = search_knowledge_filtered("公司", top_k=5)
            print(f"  按 category={target_cat!r} 过滤: {len(filtered)} 条")
            print(f"  不过滤: {len(unfiltered)} 条")
            for r in filtered[:3]:
                print(f"    [{r['metadata'].get('category')}] {r['document'][:50]}...")
            assert all(r["metadata"].get("category") == target_cat for r in filtered)
        print("✅ 进阶通过 — 生产场景按用户身份过滤知识库")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


CP2 = '''# ============================================================
# Checkpoint 2 | RAG 端到端测试 + 失败收集
# ============================================================
#
# 【基础】(人人必做，10 min)
#   定义 7-10 个真实问题，跑 RAG 看回答 + 来源
#
# 【进阶】(技术学员选做，10 min)
#   batch_eval_rag(questions, expected_keywords) → 自动判分:
#   - 检查 answer 是否包含期望关键词
#   - 收集失败 case → 用于下一轮 RAG 改进
# ============================================================

# ──── 【基础】测试用例 ────
# ↓↓↓ 【基础】填空：定义测试问题 ↓↓↓
test_questions = [
    "入职5年有几天年假?",
    "出差住宿标准是多少?",
    "StarLink 基础版支持哪些协议?",
    "StarView 企业版多少钱?",
    "公司有没有加密货币产品有哪些?",
    "报销流程是什么?",
    "部署服务需要容器化吗?",
]
# ↑↑↑ 【基础】结束 ↑↑↑


def batch_eval_rag(questions, expected_keywords_list):
    """【进阶】批量评测：检查回答是否包含期望关键词，输出 success rate + 失败 case"""
    # ↓↓↓ 【进阶】填空（约 14 行）↓↓↓
    if expected_keywords_list is None:
        return None
    success = 0
    failures = []
    for q, kws in zip(questions, expected_keywords_list):
        result = rag.answer(q)
        ans = (result.get("answer") or "").lower()
        hit = sum(1 for k in kws if k.lower() in ans)
        if hit >= max(1, len(kws) // 2):
            success += 1
        else:
            failures.append({"query": q, "expected": kws, "got": result.get("answer", "")[:80]})
    return {"success_rate": success / len(questions), "failures": failures}
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】RAG 端到端 7 个问题"); print("=" * 56)
    cp2_has_answer = 0
    cp2_has_source = 0
    try:
        for q in test_questions:
            result = rag.answer(q)
            print(f"\\n{'='*56}")
            print(f"[Q] {result['query']}")
            print(f"[A] {result['answer'][:120]}")
            if result['sources']:
                print(f"[来源] {', '.join(s['title'] for s in result['sources'])}")
                cp2_has_source += 1
            if result['answer'] and len(result['answer'].strip()) > 0:
                cp2_has_answer += 1
        assert cp2_has_answer >= 4, f"应 ≥ 4 个问题获得回答，得到 {cp2_has_answer}"
        assert cp2_has_source >= 4, f"应 ≥ 4 个问题检索到来源，得到 {cp2_has_source}"
        print(f"\\n✅ 基础通过 — {cp2_has_answer}/{len(test_questions)} 答+ {cp2_has_source}/{len(test_questions)} 源\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】batch_eval_rag (含期望关键词)"); print("=" * 56)
    try:
        expected = [
            ["年假", "天"],            # 入职 5 年
            ["住宿", "元", "500"],      # 出差住宿
            ["StarLink"],              # 协议
            ["StarView", "元"],        # 价格
            ["加密货币"],              # OOD
            ["报销", "发票"],          # 报销
            ["容器", "部署"],          # 部署
        ]
        eval_result = batch_eval_rag(test_questions, expected)
        print(f"  Success rate: {eval_result['success_rate']:.0%}")
        print(f"  失败 cases: {len(eval_result['failures'])} 个")
        for f in eval_result['failures'][:3]:
            print(f"    Q: {f['query']}")
            print(f"      期望关键词: {f['expected']}")
            print(f"      实际回答: {f['got']}...")
        print(f"  💡 失败 cases 是下一轮改进的输入：补文档 / 调 chunking / 加 reranker")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


CP3 = '''# ============================================================
# Checkpoint 3 | Agent 路由测试 + 错路由分析
# ============================================================
#
# 【基础】(人人必做，10 min)
#   定义 4 类问题，期望 Agent 路由到对应 tool（rag/calc/code/direct）
#
# 【进阶】(技术学员选做，10 min)
#   实现 audit_routing(test_cases) → 路由准确率 + 错路由统计
#   失败 case 应该被记录用于改进 Agent 提示词
# ============================================================

# ──── 【基础】测试用例 ────
# ↓↓↓ 【基础】填空：定义测试问题 + 期望 tool ↓↓↓
agent_test_cases = [
    "公司的年假制度是怎样的?",                                  # → rag_search
    "出差5天,住宿每天500元,交通补贴每天100元,总共多少?",          # → calculator
    "请帮我写一个Python函数,计算1到100的偶数列表并打印前10个",     # → code_executor
    "Python 的 list comprehension 怎么用?",                     # → direct_answer
]
expected_tools = ["rag_search", "calculator", "code_executor", "direct_answer"]
# ↑↑↑ 【基础】结束 ↑↑↑


def audit_routing(test_cases, expected, agent):
    """【进阶】路由准确率：实际调用 tool 是否包含期望 tool"""
    # ↓↓↓ 【进阶】填空（约 12 行）↓↓↓
    correct = 0
    audit_log = []
    for q, exp in zip(test_cases, expected):
        result = agent.run(q, verbose=False)
        used_tools = result.get("tools_used", [])
        # 兼容字段名差异
        if not used_tools and result.get("steps"):
            used_tools = [s.get("action") or s.get("tool") for s in result["steps"]]
        is_correct = exp in (used_tools or []) or exp == "direct_answer"
        if is_correct:
            correct += 1
        audit_log.append({"q": q, "expected": exp, "actual": used_tools, "ok": is_correct})
    return {"accuracy": correct / len(test_cases), "log": audit_log}
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】Agent 路由 4 个 case"); print("=" * 56)
    try:
        cp3_results = []
        for q, expected in zip(agent_test_cases, expected_tools):
            print(f"\\n{'='*56}")
            print(f"[Q] {q}")
            print(f"[预期工具] {expected}")
            result = agent.run(q, verbose=True)
            print(f"\\n[A] {result['answer'][:200]}")
            print(f"[步数] {result['num_steps']}")
            cp3_results.append(result)
        cp3_answered = sum(1 for r in cp3_results if r.get('answer') and len(r['answer'].strip()) > 0)
        cp3_reasonable = sum(1 for r in cp3_results if 0 < r.get('num_steps', 0) <= 5)
        assert cp3_answered == len(agent_test_cases), f"应所有问题都有答案，得到 {cp3_answered}"
        print(f"\\n✅ 基础通过 — 全部 {cp3_answered} 个回答 | {cp3_reasonable} 个步数合理\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】路由审计 (路由准确率)"); print("=" * 56)
    try:
        audit = audit_routing(agent_test_cases, expected_tools, agent)
        print(f"  路由准确率: {audit['accuracy']:.0%}")
        print(f"  详情:")
        for log in audit['log']:
            ok = '✓' if log['ok'] else '✗'
            print(f"    {ok} 期望={log['expected']:<14} 实际={log['actual']}  {log['q'][:30]}")
        print("  💡 错路由 case 用来改 Agent system prompt（明确每个 tool 的边界）")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


CP4 = '''# ============================================================
# Checkpoint 4 | LLM-as-Judge 评测 + Inter-rater Agreement
# ============================================================
#
# 【基础】(人人必做，10 min)
#   跑 LLM-as-Judge 给 RAG/Agent 输出打分（0-5）
#
# 【进阶】(技术学员选做，10 min)
#   实现 inter_rater_agreement：让 LLM 用同一 prompt 跑 2 次，看一致性
#   一致性低 → judge prompt 不稳定，需要更明确的评分标准
# ============================================================

# ──── 【基础】Judge 跑评测 ────
# ↓↓↓ 【基础】填空：定义评测样本 + 跑 judge ↓↓↓
eval_samples = []
for q in test_questions[:4]:
    rag_result = rag.answer(q)
    eval_samples.append({
        "query": q,
        "answer": rag_result["answer"],
        "sources": rag_result.get("sources", []),
    })
# ↑↑↑ 【基础】结束 ↑↑↑


def judge_score(query, answer):
    """让 LLM 给单个 RAG 回答打分 0-5"""
    judge_prompt = f"""请评分下面 RAG 系统的回答，0-5 分。

问题: {query}
回答: {answer}

评分标准：
5 = 完全正确，引用具体信息
3 = 部分正确，但有遗漏或不准确
1 = 答非所问或胡编
0 = 完全错误或拒答

只输出一个数字 0-5。"""
    raw = llm.generate(judge_prompt, temperature=0).strip()
    try:
        return int(''.join(c for c in raw if c.isdigit())[:1])
    except (ValueError, IndexError):
        return 0


def inter_rater_agreement(samples, n_trials=2):
    """【进阶】让同一 LLM 用同一 prompt 跑 N 次，算分数差异
    Returns: avg_score, max_disagreement (0=完全一致)
    """
    # ↓↓↓ 【进阶】填空（约 10 行）↓↓↓
    all_scores = []
    for s in samples:
        scores = [judge_score(s["query"], s["answer"]) for _ in range(n_trials)]
        all_scores.append(scores)
    avg = sum(s for trial in all_scores for s in trial) / (len(all_scores) * n_trials)
    max_disagree = max(max(s) - min(s) for s in all_scores)
    return {"avg_score": avg, "max_disagreement": max_disagree, "per_sample": all_scores}
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】LLM-as-Judge 单次评分"); print("=" * 56)
    try:
        scores = []
        for s in eval_samples:
            score = judge_score(s["query"], s["answer"])
            scores.append(score)
            print(f"  [{score}/5] Q: {s['query'][:40]}")
            print(f"          A: {s['answer'][:60]}...")
        avg = sum(scores) / len(scores) if scores else 0
        print(f"\\n  平均分: {avg:.2f}/5")
        assert all(0 <= s <= 5 for s in scores)
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】Inter-rater Agreement (judge 自一致性)"); print("=" * 56)
    try:
        result = inter_rater_agreement(eval_samples, n_trials=2)
        print(f"  平均分: {result['avg_score']:.2f}/5")
        print(f"  最大分歧: {result['max_disagreement']} 分 (0 = judge 完全一致)")
        print(f"\\n  逐样本分数:")
        for sample, scores in zip(eval_samples, result['per_sample']):
            print(f"    Q: {sample['query'][:40]} → trials: {scores}")
        if result['max_disagreement'] >= 2:
            print(f"\\n  ⚠ 分歧 ≥ 2 → judge prompt 不稳定，需细化评分标准")
        else:
            print(f"\\n  ✓ judge 一致性可接受")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


REWRITES = {
    "Checkpoint 1: 实现知识库检索函数": CP1,
    "Checkpoint 2: 测试 RAG 系统": CP2,
    "Checkpoint 3: 测试 Agent 路由": CP3,
    "Checkpoint 4: 运行评测": CP4,
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
