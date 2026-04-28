# Enterprise AI 5-Day Training

5 天 / 10 个 3 小时 session 的完整培训：从 LLM 原理到 2026 工业级 Multi-Agent / MCP / Agentic RAG / LLMOps。

**与 3 天版（`../enterprise_ver2/`）的关系：**
- Day 1-3 与 3 天版**完全一致**
- Day 4-5 是新增 4 个 session + 升级 Capstone（全是 2026 工业应用 + Agent 主题）
- 上过 3 天班的学员可**直接来 Day 4-5**

---

## 文件夹说明

```
enterprise_5days/
├── instructor/             # 讲师版 11 本（含答案 + 内嵌讲课提示 + 全 cell output）⚠️ 不要群发学员
├── student/                # 学员版 11 本（留填空 + 演示类 output 保留）✅ 发给学员
├── utils/                  # 共享工具（**与仓库根 utils/ 同源**）
│   ├── llm_backend.py / embedding_backend.py / config.py  # 复用 3 天版
│   ├── multi_agent.py      # ★ Multi-Agent 调度器
│   ├── mcp_helpers.py      # ★ MCP server/client 简化封装
│   ├── observability.py    # ★ Langfuse trace 包装
│   └── skills_helpers.py   # ★ Anthropic Skills 解析与路由
│   # 同源策略：根 utils/ 是唯一源，本目录在 Linux/macOS 上是指向 ../../utils 的
│   # symlink；Windows 默认权限下退化为拷贝（带 .utils_is_copy 标记）。
│   # 修改时请改根 utils/ 然后跑 `python tools/restore_utils_symlink.py` 同步。
├── data/
│   ├── custom_pretrain_corpus.txt  # Day 2 用
│   ├── enterprise_docs/            # ★ 升级 Capstone 用多文档语料
│   └── eval_dataset.jsonl          # ★ 升级 Capstone 评测集
├── fonts/                  # 中文字体
├── mcp_server_demo/        # ★ Day 4 下午独立 MCP server 项目
├── requirements.txt
├── .env.example
├── README.md
└── BUILD.md                # 维护者构建说明
```

---

## 课程节奏

| Day | 上午 (9-12) | 下午 (14-17) |
|---|---|---|
| **Day 0** | 环境配置（开班前自助 20 min） | — |
| **Day 1** | 从文本到向量（训练循环 / Tokenizer / Embedding / Prompt） | Transformer 架构（Self-Attn / Block / GPT 文本生成） |
| **Day 2** | 预训练 + SFT 完整闭环 | LoRA + DPO + 评测三件套 |
| **Day 3** | RAG 基础 + ReAct/Code Agent | 小 Capstone：企业知识助手 (RAG+Agent) |
| **Day 4** | **Multi-Agent 协作**（hierarchical / debate / handoff） | **MCP 协议实战**（写 server + client + 接企业 API） |
| **Day 5** | **Agentic RAG**（Self-RAG / CRAG / Hybrid Retrieval） | **LLMOps + 升级 Capstone**（Langfuse trace + 大型集成） |

---

## 快速开始

### 1. 准备环境

```bash
conda activate llmc        # 与 3 天版同环境
pip install -r requirements.txt   # 装 mcp / langfuse / rank-bm25 等新依赖
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env 填 DASHSCOPE_API_KEY；如要做 Day 5 LLMOps 也填 LANGFUSE_*
```

### 3. 跑 Day0 验证

```bash
jupyter lab instructor/Day0_环境配置与测试.ipynb
```

---

## Day 4-5 新增内容详细

### Day 4 上午 — Multi-Agent 协作
3 大模式：**Hierarchical**（Planner→Worker→Reviewer）/ **Debate**（多 Agent 投票）/ **Handoff**（OpenAI Swarm 模式）。素材基于 `Applications/App4_Multi_Agent.ipynb` 拓展，4 个【基础+进阶+verify】练习。

### Day 4 下午 — MCP (Model Context Protocol)
Anthropic 推的跨厂工具协议，类比 USB-C。覆盖 Tools / Resources / Prompts 三件套，**学员产出 1 个能跑的 `mcp_server_demo/` 项目**。

### Day 5 上午 — Agentic RAG
基础 RAG 升级：**Self-RAG**（自反思）/ **CRAG**（错误修正）/ **Hybrid Retrieval**（Vector+BM25+RRF）/ **Cross-Encoder Reranker**（bge-reranker）/ **MMR** 多样性去重。

### Day 5 下午 — LLMOps + 升级 Capstone
- 块 1：**Langfuse 集成**实操，给所有 LLM 调用加 `@observe`，dashboard 看 trace + token + 延迟
- 块 2：**升级 Capstone** = Day 3 Capstone + Multi-Agent + MCP + Agentic RAG + 全程可观测

---

## 给讲师的发布约定

- **务必发 `student/` 版**给学员（已剥离解答和讲课提示）
- `instructor/` 版**不要进学员群**——含全部解答 + 章节开头的「📋 讲课提示」
- 上课时讲师在自己屏幕上开 instructor 版，学员屏幕开 student 版

---

## 与 3 天版的差异（细化）

- **新依赖**：`mcp`, `langfuse`, `rank-bm25`, `sentence-transformers>=3.0`, `opentelemetry-*`
- **新 utils**：`multi_agent.py`, `mcp_helpers.py`, `observability.py`
- **新 data**：`enterprise_docs/`（多文档企业语料）+ `eval_dataset.jsonl`
- **新独立项目**：`mcp_server_demo/`（学员可 fork 改成自己公司用）
- **out of scope**：Vision / Voice / Reasoning / Computer Use / Safety / RLHF（按"应用+Agent 重度"路线砍）

---

## 验证

```bash
# 文件数量
ls instructor/*.ipynb | wc -l   # 期望 11
ls student/*.ipynb | wc -l       # 期望 11

# Day1-3 与 3 天版一致
diff -r ../enterprise_ver2/instructor/Day0_环境配置与测试.ipynb instructor/Day0_环境配置与测试.ipynb

# Day4-5 跑通（conda llmc + 已配 .env）
for nb in instructor/Day{4,5}*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 "$nb"
done
```
