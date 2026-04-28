"""Batch 5 rewrite for Day3_上午: 5 exercises (RAG + Agent)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_lib import load_nb, save_nb, cell_source, set_cell_source, add_tag

PATH = Path("assets/enterprise_ver2/instructor/Day3_上午_RAG与Agent实战.ipynb")


EX1 = '''# ============================================================
# 练习 1 | 知识库切块：句 / 段 / 滑窗
# ============================================================
#
# 【基础】(人人必做，10 min)
#   实现 split_by_sentence + split_by_paragraph
#   提示：句用 [。！？] 切；段用 \\n\\n 切
#
# 【进阶】(技术学员选做，10 min)
#   实现 split_sliding_window(text, size, stride)：滑窗带重叠
#   工业界（LangChain/LlamaIndex 默认）的策略；保上下文连贯性
# ============================================================
import re

raw_wiki_text = """
公司差旅报销制度（2024年修订版）

一、差旅标准
员工因公出差需提前填写出差申请单，经部门经理审批后方可出行。交通方面，普通员工乘坐高铁二等座或经济舱，部门经理及以上可乘坐高铁一等座或商务舱。住宿标准为一线城市每晚不超过500元，二三线城市每晚不超过350元。

二、报销流程
出差结束后5个工作日内提交报销申请，需提供发票原件、出差申请单及行程单。财务审核后通过工资卡发放。

三、特殊情况
因特殊原因（如不可抗力、海外出差）超标，需事前书面说明并经分管领导批准。招待费需双人核签且不得超过总差旅费30%。
"""


def split_by_sentence(text):
    """【基础】句级切块"""
    # ↓↓↓ 【基础】填空（约 2 行）↓↓↓
    sentences = re.split(r'[。！？]', text)
    return [s.strip() for s in sentences if s.strip()]
    # ↑↑↑ 【基础】结束 ↑↑↑


def split_by_paragraph(text):
    """【基础】段落切块"""
    # ↓↓↓ 【基础】填空（约 2 行）↓↓↓
    paragraphs = text.split('\\n\\n')
    return [p.strip() for p in paragraphs if p.strip()]
    # ↑↑↑ 【基础】结束 ↑↑↑


def split_sliding_window(text, size=200, stride=150):
    """【进阶】滑窗切块：每块 size 字符，步长 stride（重叠 = size - stride）"""
    # ↓↓↓ 【进阶】填空（约 5 行）↓↓↓
    chunks = []
    text = text.replace('\\n', ' ')
    for start in range(0, max(len(text) - size + stride, 1), stride):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】split_by_sentence + split_by_paragraph"); print("=" * 56)
    try:
        sentences = split_by_sentence(raw_wiki_text)
        paragraphs = split_by_paragraph(raw_wiki_text)
        print(f"  句级切块: {len(sentences)} 块  | 平均长度 {sum(len(s) for s in sentences)/max(len(sentences),1):.0f} 字")
        print(f"  段级切块: {len(paragraphs)} 块  | 平均长度 {sum(len(p) for p in paragraphs)/max(len(paragraphs),1):.0f} 字")
        assert len(sentences) > len(paragraphs), "句级应比段级更碎"
        assert all(len(s) > 0 for s in sentences)
        print("\\n  💡 段级保留更完整上下文；句级粒度细但易丢上下文")
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】滑窗切块（含重叠）"); print("=" * 56)
    try:
        windows = split_sliding_window(raw_wiki_text, size=120, stride=80)
        print(f"  滑窗 (size=120, stride=80, 重叠 40 字): {len(windows)} 块")
        for i, w in enumerate(windows[:3]):
            print(f"    [{i}] {w[:60]}...")
        # 相邻块应有重叠
        if len(windows) >= 2:
            common = set(windows[0]) & set(windows[1])
            assert len(common) > 5, "相邻块应共享多个字符（重叠）"
        print("  💡 滑窗 = LangChain RecursiveCharacterTextSplitter 默认；保 chunk 之间上下文连续")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


EX2 = '''# ============================================================
# 练习 2 | RAG 失败模式手册：检测 + 兜底
# ============================================================
#
# 【基础】(人人必做，10 min)
#   实现 detect_low_confidence(results, threshold)：
#   - 无结果 / 最高分 < threshold / top1 与 top2 太接近 → 都判为低置信度
#
# 【进阶】(技术学员选做，10 min)
#   实现 fallback_response(query, results, low_conf, reason)：
#   - 若 OOD：返回安全模板『抱歉，知识库未覆盖...』
#   - 若 ambiguous：反问『你是想了解 X 还是 Y？』
#   - 否则：正常 prompt
# ============================================================

def detect_low_confidence(search_results, threshold=0.4):
    """【基础】返回 (is_low_conf, reason)"""
    # ↓↓↓ 【基础】填空（约 8 行）↓↓↓
    if not search_results:
        return True, "无检索结果"
    top_score = search_results[0].get("score", 0)
    if top_score < threshold:
        return True, f"最高相关度 {top_score:.3f} < 阈值 {threshold}"
    if len(search_results) > 1 and abs(search_results[0]["score"] - search_results[1]["score"]) < 0.05:
        return True, "top1 与 top2 分数过于接近"
    return False, "置信度正常"
    # ↑↑↑ 【基础】结束 ↑↑↑


def fallback_response(query, results, is_low_conf, reason):
    """【进阶】根据失败模式给不同的兜底回复"""
    # ↓↓↓ 【进阶】填空（约 10 行）↓↓↓
    if not is_low_conf:
        return None  # 让正常 RAG 流程处理
    if "无检索" in reason or "阈值" in reason:
        return f"抱歉，知识库未覆盖『{query}』相关内容。建议联系业务部门或查询权威来源。"
    if "接近" in reason:
        # 列出 top 2 片段让用户澄清
        top1 = results[0].get("document", "")[:60]
        top2 = results[1].get("document", "")[:60] if len(results) > 1 else ""
        return f"问题可能涉及多个方面，请澄清：\\n  A. {top1}...\\n  B. {top2}...\\n你想了解哪个？"
    return f"抱歉，无法确定查询『{query}』的可靠答案。"
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】detect_low_confidence"); print("=" * 56)
    try:
        cases = [
            ([], "无结果应返回 True"),
            ([{"score": 0.2}], "低分应返回 True"),
            ([{"score": 0.8}, {"score": 0.79}], "top1/top2 太近应返回 True"),
            ([{"score": 0.8}, {"score": 0.5}], "正常应返回 False"),
        ]
        for results, desc in cases:
            low, reason = detect_low_confidence(results, threshold=0.4)
            print(f"  {desc}: low={low}  reason={reason}")
        # 单元测试
        assert detect_low_confidence([])[0] == True
        assert detect_low_confidence([{"score": 0.2}])[0] == True
        assert detect_low_confidence([{"score": 0.8}, {"score": 0.5}])[0] == False
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】fallback_response — 不同失败给不同兜底"); print("=" * 56)
    try:
        # OOD 场景
        r1 = fallback_response("今天天气", [], True, "无检索结果")
        print(f"  OOD: {r1[:60]}")
        assert "抱歉" in r1 or "未覆盖" in r1
        # ambiguous 场景
        r2 = fallback_response("差旅", [{"document": "A 关于差旅标准...", "score": 0.7}, {"document": "B 关于报销流程...", "score": 0.69}], True, "top1 与 top2 分数过于接近")
        print(f"  Ambiguous: {r2[:80]}")
        assert "澄清" in r2 or "想了解" in r2
        # 正常场景
        r3 = fallback_response("差旅标准", [{"document": "高铁二等座", "score": 0.9}], False, "置信度正常")
        assert r3 is None
        print("  正常: None (走正常 RAG)")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


EX3 = '''# ============================================================
# 练习 3 | 给 Agent 添加单位转换工具 + 扩展支持
# ============================================================
#
# 【基础】(人人必做，10 min)
#   把 unit_converter 注册成 Tool，加到 Agent 工具列表
#
# 【进阶】(技术学员选做，10 min)
#   实现 robust_converter(value, from_unit, to_unit)：
#   - 自动 strip 大小写、空格
#   - 支持别名（"摄氏度"="C", "公里"="km", "华氏度"="F"）
# ============================================================

def unit_converter(value, from_unit, to_unit):
    """【基础】单位转换"""
    value = float(value)
    conversions = {
        ("C", "F"): lambda v: v * 9/5 + 32,
        ("F", "C"): lambda v: (v - 32) * 5/9,
        ("C", "K"): lambda v: v + 273.15,
        ("K", "C"): lambda v: v - 273.15,
        ("m", "ft"): lambda v: v * 3.28084,
        ("ft", "m"): lambda v: v / 3.28084,
        ("km", "mi"): lambda v: v * 0.621371,
        ("mi", "km"): lambda v: v / 0.621371,
        ("kg", "lb"): lambda v: v * 2.20462,
        ("lb", "kg"): lambda v: v / 2.20462,
    }
    key = (from_unit, to_unit)
    if key in conversions:
        return f"{value} {from_unit} = {conversions[key](value):.2f} {to_unit}"
    return f"不支持的转换: {from_unit} → {to_unit}"


# ──── 【基础】注册 Tool ────
# ↓↓↓ 【基础】填空（约 8 行）↓↓↓
converter_tool = Tool(
    name="unit_converter",
    description="单位转换工具，支持温度(C/F/K)、长度(m/ft/km/mi)、重量(kg/lb)",
    parameters=["value", "from_unit", "to_unit"],
    func=unit_converter,
)

new_tools = ALL_TOOLS + [converter_tool]
agent_v2 = ReActAgent(llm, new_tools)
# ↑↑↑ 【基础】结束 ↑↑↑


# ──── 【进阶】带别名 + 容错的 robust_converter ────
def robust_converter(value, from_unit, to_unit):
    """【进阶】容错版：strip + lowercase + 中文别名"""
    # ↓↓↓ 【进阶】填空（约 14 行）↓↓↓
    aliases = {
        "摄氏度": "C", "摄氏": "C",
        "华氏度": "F", "华氏": "F",
        "开尔文": "K", "开氏": "K",
        "米": "m", "公里": "km", "千米": "km",
        "英尺": "ft", "英里": "mi",
        "公斤": "kg", "千克": "kg", "磅": "lb",
    }
    def _normalize(u):
        u = str(u).strip().lower()
        return aliases.get(u, u.upper() if u in {"c", "f", "k"} else u)
    return unit_converter(value, _normalize(from_unit), _normalize(to_unit))
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】Agent 加上单位转换 tool"); print("=" * 56)
    try:
        assert any(t.name == "unit_converter" for t in new_tools)
        # 直接调 tool 测一下
        r = unit_converter(100, "C", "F")
        print(f"  unit_converter(100, C, F) → {r}")
        assert "212" in r
        print(f"  Agent v2 工具列表: {[t.name for t in new_tools]}")
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】robust_converter (中文别名 + 容错)"); print("=" * 56)
    try:
        cases = [
            (100, "摄氏度", "华氏度"),
            (5, "公里", "英里"),
            (10, " kg ", "lb"),
            (1, "米", "ft"),
        ]
        for v, fu, tu in cases:
            r = robust_converter(v, fu, tu)
            print(f"  robust_converter({v}, {fu!r}, {tu!r}) → {r}")
            assert "不支持" not in r, f"未识别 {fu}/{tu}"
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


EX4 = '''# ============================================================
# 练习 4 | Agent + RAG 集成：单源 → 混合检索
# ============================================================
#
# 【基础】(人人必做，10 min)
#   把 RAG 包装成 Tool，加到 Agent 工具列表
#
# 【进阶】(技术学员选做，10 min)
#   实现 hybrid_retrieve(query, top_k)：
#   同时跑 vector search + 关键词匹配（简单 substring），合并去重
#   返回前 top_k 个 — 工业界 BM25+Vector hybrid 的简化版
# ============================================================

# ──── 【基础】RAG 作为 Agent 的 tool ────
def rag_search(query: str) -> str:
    """【基础】调用现有 rag.retrieve 并格式化结果"""
    # ↓↓↓ 【基础】填空（约 3 行）↓↓↓
    results = rag.retrieve(query, top_k=3)
    return "\\n".join([f"- {r['document'][:150]}" for r in results])
    # ↑↑↑ 【基础】结束 ↑↑↑


# ↓↓↓ 【基础】填空：注册 RAG tool 并组装 Agent ↓↓↓
rag_tool = Tool(
    name="knowledge_search",
    description="在知识库中搜索信息，适合回答技术知识问题（如 Transformer、LoRA、RLHF 等）",
    parameters=["query"],
    func=rag_search,
)
rag_agent_tools = [calculator_tool, datetime_tool, rag_tool]
rag_agent = ReActAgent(llm, rag_agent_tools)
# ↑↑↑ 【基础】结束 ↑↑↑


# ──── 【进阶】hybrid retrieve ────
def hybrid_retrieve(query, top_k=5):
    """【进阶】合并 vector search + keyword match
    简化的 BM25+Vector hybrid，工业界常见做法
    """
    # ↓↓↓ 【进阶】填空（约 12 行）↓↓↓
    vec_results = rag.retrieve(query, top_k=top_k)
    # 关键词匹配：取 query 中长度 ≥ 2 的字 / 词
    keywords = [w for w in query.replace('，', ' ').replace('？', ' ').split() if len(w) >= 2]
    if not keywords:
        return vec_results[:top_k]
    keyword_hits = []
    for doc in rag.documents:
        if any(kw in doc for kw in keywords):
            keyword_hits.append({"document": doc, "score": 0.5, "source": "keyword"})
    # 合并去重（按 doc[:50] 当 key）
    seen = set()
    merged = []
    for r in vec_results + keyword_hits:
        key = r["document"][:50]
        if key not in seen:
            seen.add(key)
            merged.append(r)
    return merged[:top_k]
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】RAG 作 Agent tool"); print("=" * 56)
    try:
        assert any(t.name == "knowledge_search" for t in rag_agent_tools)
        # 直接调
        r = rag_search("Transformer 架构")
        print(f"  rag_search('Transformer 架构') 返回 (前 100 字):")
        print(f"    {r[:100]}...")
        print(f"  rag_agent 工具列表: {[t.name for t in rag_agent_tools]}")
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】Hybrid Retrieve (vector + keyword)"); print("=" * 56)
    try:
        # 比较 vector-only vs hybrid
        vec_only = rag.retrieve("LoRA 微调", top_k=3)
        hybrid = hybrid_retrieve("LoRA 微调", top_k=5)
        print(f"  vector-only top-3: {len(vec_only)} 结果")
        print(f"  hybrid top-5: {len(hybrid)} 结果（含 keyword 兜底）")
        for i, r in enumerate(hybrid[:3]):
            print(f"    [{i}] {r['document'][:50]}... (score={r.get('score', 0):.2f}, src={r.get('source', 'vector')})")
        assert len(hybrid) > 0
        print("  💡 hybrid 在『罕见术语 / 数字 / ID』查询上比纯 vector 更稳")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


EX5 = '''# ============================================================
# 练习 5 | Code Agent 提示词：基础 → 健壮 → 自验
# ============================================================
#
# 【基础】(人人必做，10 min)
#   写 analysis_prompt：让 LLM 写 Python 代码统计员工绩效
#
# 【进阶】(技术学员选做，10 min)
#   实现 prompt_with_self_check(data, expected_output_keys)：
#   prompt 末尾加『检查项』要求 LLM 在输出末尾确认所有要点已计算
# ============================================================

employee_data = """
name,department,score,tenure_years
张三,技术部,92,5
李四,市场部,78,3
王五,技术部,88,7
赵六,销售部,85,4
孙七,技术部,95,6
周八,市场部,72,2
吴九,销售部,90,8
"""


# ──── 【基础】写基础分析 prompt ────
# ↓↓↓ 【基础】填空：补全 analysis_prompt ↓↓↓
analysis_prompt = f"""请分析以下员工绩效 CSV 数据，用 Python 代码完成分析并用 print 输出结果。

数据如下:
{employee_data}

请计算并输出:
1. 每个部门的平均绩效分数（保留 2 位小数）
2. 绩效最高和最低的员工姓名及分数
3. 工龄(tenure_years)和绩效分数(score)的皮尔逊相关系数
4. 每个部门的人数

注意: 请用 csv 模块或手动解析数据，不要使用 pandas。用 print() 输出所有结果。"""
# ↑↑↑ 【基础】结束 ↑↑↑


# ──── 【进阶】带自检的健壮 prompt ────
def prompt_with_self_check(data, expected_keys):
    """【进阶】生成含 self-check 步骤的 prompt：
    - 处理边界（缺失值 / 单人部门）
    - 末尾要求 LLM 列出 self-check 清单
    """
    # ↓↓↓ 【进阶】填空（约 12 行）↓↓↓
    keys_str = "\\n".join(f"   - [ ] {k}" for k in expected_keys)
    return f"""请分析以下员工绩效 CSV 数据。要求：

数据：
{data}

需要计算：
{chr(10).join(f"{i+1}. {k}" for i, k in enumerate(expected_keys))}

健壮性要求：
- 跳过缺失值的行
- 若某部门只有 1 人，标注『标准差不适用』
- 不使用 pandas，用 csv 模块或手动解析

代码末尾必须添加自检 print：
```python
print("\\n--- 自检 ---")
{keys_str}
```
然后逐项把 [ ] 改成 [x] 表示已完成。"""
    # ↑↑↑ 【进阶】结束 ↑↑↑


def verify():
    print("=" * 56); print("【基础】analysis_prompt → 让 Code Agent 跑"); print("=" * 56)
    try:
        assert "csv" in analysis_prompt or "解析" in analysis_prompt
        assert "皮尔逊" in analysis_prompt or "相关" in analysis_prompt
        # 用 code_agent 跑一次
        if 'code_agent' in dir():
            print("  ▶ Code Agent 执行中...")
            result = code_agent.run(analysis_prompt)
            print(f"  最终输出 (前 200 字):\\n  {str(result)[:200]}...")
        else:
            print("  (code_agent 不可用，跳过实际运行)")
        print("✅ 基础通过\\n")
    except NotImplementedError:
        print("⏭ 基础未实现\\n"); return
    except Exception as e:
        print(f"❌ 基础未通过: {type(e).__name__}: {e}\\n"); return

    print("=" * 56); print("【进阶】prompt_with_self_check (带自检清单)"); print("=" * 56)
    try:
        keys = [
            "每个部门的平均分 + 标准差",
            "绩效最高/最低员工",
            "工龄-绩效相关系数",
            "缺失值数量",
        ]
        rich_prompt = prompt_with_self_check(employee_data, keys)
        assert "自检" in rich_prompt and "[ ]" in rich_prompt
        print(f"  生成的 prompt 长度: {len(rich_prompt)} 字符")
        print(f"  含 self-check 清单: ✓")
        if 'code_agent' in dir():
            print("  ▶ Code Agent 执行中...")
            result = code_agent.run(rich_prompt)
            print(f"  最终输出 (前 200 字):\\n  {str(result)[:200]}...")
            # 检查输出里是否真有 self-check 标记
            has_check = '[x]' in str(result) or '✓' in str(result)
            print(f"  ✓ LLM 完成 self-check: {has_check}")
        print("✅ 进阶通过")
    except NotImplementedError:
        print("⏭ 进阶跳过（未实现）")
    except Exception as e:
        print(f"❌ 进阶未通过: {type(e).__name__}: {e}")

verify()
'''


REWRITES = {
    "练习 1：设计企业知识库": EX1,
    "练习 2：RAG失败模式手册": EX2,
    "练习 3: 给 Agent 添加单位转换工具": EX3,
    "练习 4: 让 Agent 把 RAG 知识库作为工具使用": EX4,
    "练习 5：设计 Code Agent 提示词": EX5,
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
