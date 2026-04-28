# 架构全景图

本文是 **Ch 主线 → Applications → utils 后端** 三层学习路径的全景图。
目的：让你在跳进任何一个 notebook 前，先看清它在整张图里的位置。

---

## 三层关系

```
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 1 · 主线章节 Ch00–Ch12                                        │
│  从原理到 Agent 的渐进式课程；每章一个核心概念，配可视化与 toy demo  │
└──────────────────────────────────────────────────────────────────────┘
                              │  概念准备
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 2 · Applications/App1–App8                                    │
│  把 Ch 学到的能力组合成可运行的工业级案例；每个 App 一个垂直主题     │
└──────────────────────────────────────────────────────────────────────┘
                              │  调用
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 3 · utils/                                                    │
│  跨 App 复用的后端抽象（LLM 后端、Embedding、Multi-Agent、MCP、      │
│  Observability、Skills、RAG 模式）；唯一源在 repo 根 utils/          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Ch 主线 → App 的"概念落地"映射

| 主线章节 | 教什么 | 落地到 |
|---|---|---|
| Ch01–Ch07 | Autograd / Embedding / Attention / Transformer / Tokenizer / Pretraining | 是后续 Apps 的"白盒底座"，所有 Apps 假设你看过这些 |
| Ch08–Ch10 | SFT / LoRA / DPO（Route A） | App1–App8 选择 LLM 后端时背后机制的解释 |
| Ch11 | KV Cache | 推理优化直觉 |
| Ch12 | Agent & RAG 原理 | **App1–App8 的直接前置**——读完 Ch12 再做 Apps |

> 后续将新增 Ch13/14/15（骨架已就位）：Multi-Agent / MCP / LLMOps，作为 App4/App5/App7 的概念章。

---

## Applications 内部依赖

```
Ch12 (前置)
   │
   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ App1 ReAct   │  │ App2 RAG     │  │ App3 Code    │  │ App4 Multi   │
│ (基础 Agent) │  │ (5 高级模式) │  │ (sandbox)    │  │ (3 编排+CB)  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       └──────────────────┴──────────────────┴──────────────────┘
                                  │  组合
                                  ▼
       ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
       │ App5 MCP     │  │ App6 Skills  │  │ App7 LLMOps  │
       │ (协议)       │  │ (能力包)     │  │ (可观测性)   │
       └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
              └──────────────────┴──────────────────┘
                                  │  集大成
                                  ▼
                        ┌────────────────────┐
                        │ App8 Capstone      │
                        │ Multi×MCP×RAG×Ops  │
                        └────────────────────┘
```

App1–App4 是"基础 Agent 4 项"，App5–App8 是"2026 工业栈 4 项"。不同学习目标对应不同最短路径——见末尾的"读这门课的最短路径"表格。

---

## utils/ 与 Apps 的引用矩阵

| utils 模块 | 主要消费者（Apps） | 干什么 |
|---|---|---|
| `config.py` | 全部 Apps | 统一 .env 加载 + LLM/Embedding 后端工厂 |
| `llm_backend.py` | 全部 Apps | OpenAI / DashScope / Ollama / HF / vLLM 后端的统一接口 |
| `embedding_backend.py` | App2, App8 | SentenceTransformers / OpenAI / HF embedding |
| `multi_agent.py` | **App4**, App8 | `BaseAgent` + `Orchestrator`（Hierarchical/Debate/Handoff）+ `CircuitBreaker` |
| `mcp_helpers.py` | App3, **App5**, App8 | `EduMCPServer/Client` + `tool_from_function` |
| `observability.py` | App1, App4, **App7**, App8 | `@observe` + `MockObserver` + Langfuse fallback |
| `skills_helpers.py` | **App6**, App8 | Anthropic Skills 解析 + 路由 + progressive disclosure |
| `rag_patterns.py` _(规划中)_ | App2, App8 | Self-RAG / CRAG / Hybrid+RRF / Reranker / MMR |

加粗 = 该 App 是这个 module 的"主战场"。

---

## 与 5 天企业版 (`assets/enterprise_5days/`) 的关系

`enterprise_5days/utils/` 与本仓库根 `utils/` **共享同一份代码**：

- Linux/macOS：`assets/enterprise_5days/utils/` 是指向 `../../utils/` 的 symlink
- Windows 默认权限下：自动 fallback 为定期同步的拷贝（含 `.utils_is_copy` 标记）

任何 utils 改动都做在根 `utils/`，然后跑：

```bash
python tools/restore_utils_symlink.py
```

让 5days 那侧重新对齐。两侧 import 的代码永远是同一份。

---

## 读这门课的最短路径

| 你的目标 | 推荐路径 |
|---|---|
| 只想跑通 Agent | App0 → Ch12 → App1 → App2 → App4 |
| 想理解 LLM 工业栈（2026） | App0 → App1–App4 → App5 → App6 → App7 → App8 |
| 学院派完整旅程 | Ch01–Ch12 → App1–App8 |
| 5 天集训复刻 | `assets/enterprise_5days/student/` Day0 → Day5 |
