# 5 天版构建说明（给维护者）

学员只读 `README.md`。本文件给维护者：从 0 重建 5 天版的步骤。

---

## 工作流总览

```
assets/enterprise_ver2/  (3 天版，已经做好)
        │
        │ ① 复制 7 本 notebook + utils/data/fonts (Batch 1)
        ▼
assets/enterprise_5days/instructor/  (Day 0-3 与 3 天版一致)
        │
        │ ② 新建 4 本 Day4-5 notebook (Batch 2-5)
        │   - Day4_上午_Multi-Agent协作
        │   - Day4_下午_MCP协议实战
        │   - Day5_上午_Agentic_RAG
        │   - Day5_下午_LLMOps与生产Capstone
        │ + 新建 utils/{multi_agent, mcp_helpers, observability}.py
        │ + 建 mcp_server_demo/ 独立项目
        ▼
assets/enterprise_5days/instructor/  (11 本完整 notebook + 工具)
        │
        │ ③ jupyter nbconvert --execute (Batch 6)
        ▼
assets/enterprise_5days/instructor/  (11 本含 cell output)
        │
        │ ④ tools/derive_student.py (Batch 6)
        ▼
assets/enterprise_5days/student/  (11 本学员版)
```

---

## Batch 1 — 骨架与复用

```bash
mkdir -p assets/enterprise_5days/{instructor,student,utils,data/enterprise_docs,fonts,mcp_server_demo}

# 7 nb 复用
cp assets/enterprise_ver2/instructor/Day{0,1_上午,1_下午,2_上午,2_下午,3_上午,3_下午}*.ipynb \
   assets/enterprise_5days/instructor/

# utils + data + fonts 复用
cp assets/enterprise_ver2/utils/{__init__,llm_backend,embedding_backend,config}.py assets/enterprise_5days/utils/
cp assets/enterprise_ver2/data/custom_pretrain_corpus.txt assets/enterprise_5days/data/
cp assets/enterprise_ver2/fonts/NotoSansCJKsc-Regular.otf assets/enterprise_5days/fonts/
```

新写：`README.md` / `BUILD.md` / `.env.example` / `requirements.txt`

## Batch 2-5 — 4 本新 notebook

每个 batch = 1 本新 notebook + 可能的新 utils 文件。详见 `tools/build5d_day{4am,4pm,5am,5pm}.py`（构建过程中创建）。

每本 notebook 严格按 3 天版的【基础+进阶+verify】范式：
- 章首 markdown「📋 讲课提示」（tag `instructor_only`）
- 4 个练习 cell（tag `fillin` + `batch5`），含 `↓↓↓ 【基础】填空 ↓↓↓` 与 `↓↓↓ 【进阶】填空 ↓↓↓` 两块
- 每练习末尾 `verify()` 自动 assert ✅/❌/⏭

## Batch 6 — 预跑 + 派生 student

```bash
conda activate llmc
cd assets/enterprise_5days
cp .env.example .env  # 填 DASHSCOPE_API_KEY + LANGFUSE_*

# 预跑 4 本新 notebook（Day0-3 复用 ver2 已跑过的 output）
for nb in instructor/Day{4,5}*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=2400 "$nb"
done

# 派生学员版
python ../../tools/derive_student.py  # 默认指向 ver2，改路径或在 5days 下另存

rm .env  # 立即清！
```

注意：`tools/derive_student.py` 当前默认指向 `enterprise_ver2/`。复用时复制一份 `tools/derive_student_5days.py`，改 `INSTRUCTOR` 与 `STUDENT` 路径。

---

## 验证清单

```bash
# 1. 文件数
ls assets/enterprise_5days/instructor/*.ipynb | wc -l   # 11
ls assets/enterprise_5days/student/*.ipynb | wc -l       # 11

# 2. Day1-3 与 3 天版一致
for nb in Day0 Day1_上午 Day1_下午 Day2_上午 Day2_下午 Day3_上午 Day3_下午; do
  cmp assets/enterprise_ver2/instructor/${nb}*.ipynb \
      assets/enterprise_5days/instructor/${nb}*.ipynb && echo "✓ $nb" || echo "✗ $nb"
done

# 3. Day4-5 砍肉验证（通过 3 天版的 grep 规则）
grep -rE "节省¥|月省|年节省|商业决策分析|成本计算器" \
  assets/enterprise_5days/instructor/Day{4,5}*.ipynb 2>/dev/null | head

# 4. 学员版 fillin 数量
python -c "
import json, glob
for nb in sorted(glob.glob('assets/enterprise_5days/student/Day{4,5}*.ipynb')):
    n = json.load(open(nb, encoding='utf-8'))
    n_fill = sum(1 for c in n['cells']
                 if 'fillin' in c.get('metadata',{}).get('tags',[]))
    print(f'{nb}: {n_fill} fillin')
# 期望 Day4_上午:4 / Day4_下午:4 / Day5_上午:4 / Day5_下午:3
"

# 5. MCP server 独立验证
cd assets/enterprise_5days/mcp_server_demo
python server.py &      # 起 server
python client_test.py   # 应能列出 3 个 tool
```

---

## 已知陷阱

1. **MCP SDK 版本敏感**：`mcp>=0.9` 是 stdio transport 的稳定版；如装更新版接口可能变。requirements.txt 已锁。
2. **Langfuse 双模式**：cloud 注册麻烦时改本地 docker-compose（`docker compose up langfuse`）。`utils/observability.py` 自动检测 `LANGFUSE_PUBLIC_KEY` 是否填，没填走 mock observer 不阻塞课程。
3. **bge-reranker 首次下载约 400MB**：建议讲师课前预下载到 `~/.cache/huggingface/`，学员开课直接走缓存。
4. **Multi-Agent 烧 token**：默认全用 qwen-turbo（不用 qwen-plus），单练习 ≤ 30 turns，预算可控。
5. **升级 Capstone (Day5_下午) 不要从 0 写**：讲师版预跑跑通，学员只改其中 1-2 模块（脚手架已给）。

---

## 后续扩展（不在本次范围）

如要再扩主题，下一波建议：
- Vision LLM / 多模态 RAG（图表理解 / OCR + RAG）
- Voice agents（ASR + TTS 流水线）
- Reasoning models（o1/o3/Claude 3.7 thinking 模式调用 + 评测）
- Computer Use（Claude 3.5+ 浏览器自动化）
- Safety / Guardrails / Red-team
- RLHF（repo 里有 `Bonus_A_RLHF` 完整素材，可作 6 天版加一天）
