# Enterprise AI 5-Day Training

5 天 / 10 个 3 小时 session 的完整培训：从 LLM 原理过渡到 Multi-Agent、MCP、Agentic RAG 与 LLMOps 的工程实践。

**与 3 天版（`../enterprise_ver2/`）的关系：**
- Day 1-3 与 3 天版在课程内容、练习和节奏上保持一致；少量路径提示会按各自包名显示
- Day 4-5 是新增 4 个 session + 升级 Capstone，聚焦 Agent 工程化与生产前验证
- 上过 3 天班的学员可**直接来 Day 4-5**

---

## 文件夹说明

```
enterprise_5days/
├── instructor/             # 讲师版 11 本（含答案、讲课提示和完整输出；仅供讲师使用）
├── student/                # 学员版 11 本（保留填空，outputs 已清空；用于课堂发放）
├── utils/                  # 共享工具（**与仓库根 utils/ 同源**）
│   ├── llm_backend.py / embedding_backend.py / config.py  # 复用 3 天版
│   ├── multi_agent.py      # Multi-Agent 编排示例
│   ├── mcp_helpers.py      # MCP server/client 示例
│   ├── observability.py    # Langfuse trace 封装
│   └── skills_helpers.py   # Skills 风格工作流示例
│   # 同源策略：根 utils/ 是唯一源，本目录在 Linux/macOS 上是指向 ../../utils 的
│   # symlink；Windows 默认权限下退化为拷贝（带 .utils_is_copy 标记）。
│   # 修改时请改根 utils/ 然后跑 `python tools/restore_utils_symlink.py` 同步。
├── data/
│   ├── custom_pretrain_corpus.txt  # Day 2 用
│   ├── enterprise_docs/            # 升级 Capstone 用多文档语料
│   └── eval_dataset.jsonl          # 升级 Capstone 评测集
├── fonts/                  # 中文字体
├── mcp_server_demo/        # Day 4 下午独立 MCP server 项目
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

## 与主线章节的对应关系

5 天版不是把 `Ch0-12` 逐章搬进课堂，而是按企业班节奏重组：

- Day 1：对应 Ch0-Ch6 的核心路径，覆盖训练循环、Tokenizer、Embedding、Self-Attention、Transformer Block 和 GPT 生成。
- Day 2：对应 Ch7-Ch10，覆盖预训练、SFT、LoRA、DPO 与基础评测。训练数据是教学受控数据，目标是看清训练机制和评估边界，不把课堂指标包装成真实业务泛化。
- Day 3：对应 Ch12 和 Applications App1-App4，覆盖 RAG、ReAct、Code Agent、Multi-Agent 的基础应用，并落到企业知识助手 Capstone。
- Day 4-5 是新增 4 个 session + 升级 Capstone，聚焦 Agent 工程化与生产前验证

因此，5 天版是当前交付主线：前 3 天保留原理密度，后 2 天补齐 Agent 工程化与生产前验证。

---

## 快速开始

### 1. 准备环境

```bash
conda create -n llmcs python=3.11 -y
conda activate llmcs

# 有 NVIDIA GPU（如 RTX 50 系）优先装 CUDA 12.8 版 PyTorch；CPU 机器把 cu128 改成 cpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt   # 安装除 PyTorch 外的训练栈、官方 mcp SDK、Langfuse、Agentic RAG 依赖
```

说明：5 天版交付包以 `llmcs` / Python 3.11 为主线环境。课堂仍保留自实现 stdio JSON-RPC MCP demo 与 mock observability，方便没有 Langfuse key 的学员也能跑；如果填写 `LANGFUSE_*`，会切到真实 Langfuse dashboard。

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env 填 DASHSCOPE_API_KEY；课堂默认模型是 qwen-plus-2025-01-25 + text-embedding-v3
# 如要做 Day 5 LLMOps，再填 LANGFUSE_*
```

也可以直接运行 `Day0_环境配置与测试.ipynb` 的 Step 4，只改一行 `DASHSCOPE_API_KEY = ""`。后续 notebook 都会自动读取 `.env`，不需要重复填写。

默认使用固定快照 `qwen-plus-2025-01-25`，便于课堂复现；如需使用最新别名，可把 `.env` 里的 `LLM_MODEL` 改成 `qwen-plus`，其它都不用改。

### 3. 跑 Day0 验证

```bash
jupyter lab instructor/Day0_环境配置与测试.ipynb
```

---

## Day 4-5 新增内容详细

### Day 4 上午 — Multi-Agent 协作
3 大模式：**Hierarchical**（Planner→Worker→Reviewer）/ **Debate**（多 Agent 投票）/ **Handoff**（OpenAI Swarm 模式）。素材基于 `Applications/App4_Multi_Agent.ipynb` 拓展，4 个【基础+进阶+verify】练习。

### Day 4 下午 — MCP (Model Context Protocol) + Anthropic Skills
MCP 是 Anthropic 推出的开放工具协议。课程覆盖 Tools / Resources / Prompts 三类原语，学员会产出 1 个能跑的 `mcp_server_demo/` 项目。Skills 部分讲 SKILL.md 格式、progressive disclosure 三层加载，以及 Skills 与 MCP 的集成方式，对应 `skills_demo/` 下三个可 fork 的范例。

### Day 5 上午 — Agentic RAG
基础 RAG 升级：**Self-RAG**（自反思）/ **CRAG**（错误修正）/ **Hybrid Retrieval**（Vector+BM25+RRF）/ **Cross-Encoder Reranker**（bge-reranker）/ **MMR** 多样性去重。

### Day 5 下午 — LLMOps + 升级 Capstone
- 块 1：**Langfuse 集成**实操，给所有 LLM 调用加 `@observe`，dashboard 看 trace + token + 延迟
- 块 2：**升级 Capstone** = Day 3 Capstone + Multi-Agent + MCP + Agentic RAG + 全程可观测

---

## 给讲师的发布约定

- **务必发 `student/` 版**给学员（已剥离解答和讲课提示）
- `instructor/` 版**不要进学员群**——含全部解答和章节开头的讲课提示
- 上课时讲师在自己屏幕上开 instructor 版，学员屏幕开 student 版

---

## 与 3 天版的差异（细化）

- **新依赖**：`rank-bm25`, `sentence-transformers>=5.0,<6`, `opentelemetry-*`, `mcp>=1.20,<2`, `langfuse>=3.0,<4`
- **新工具**：`multi_agent.py`, `mcp_helpers.py`, `observability.py`, `skills_helpers.py`，用于 Day 4-5 的 Agent 工程化练习
- **新 data**：`eval_dataset.jsonl`（10 题课堂小评测集）+ `enterprise_docs/`（扩展企业语料占位，当前 notebook 默认使用内嵌语料）
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

# Day4-5 跑通（conda llmcs + 已配 .env）
for nb in instructor/Day{4,5}*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 "$nb"
done
```
