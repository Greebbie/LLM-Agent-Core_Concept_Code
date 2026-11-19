# 架构全景图

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
│  ReAct、文档检索、代码执行、工具协议和综合实验                       │
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

## 章节与实验

| 主线章节 | 主题 | 对应实验 |
|---|---|---|
| Ch01–Ch07 | Autograd / Embedding / Attention / Transformer / Tokenizer / Pretraining | [组装 GPT](../Ch5_GPT_Assembly/Ch5_GPT_Assembly.ipynb)、[预训练](../Ch7_Pretraining/Ch7_Pretraining.ipynb) |
| Ch08–Ch10 | SFT / LoRA / DPO（Route A） | [工单分类](../Ch8_SFT/Ch8_SFT.ipynb)、[展馆问答](../Ch10_DPO/Ch10_DPO.ipynb) |
| Ch11 | KV Cache | [比较有无缓存的增量推理](../Ch11_KV_Cache/Ch11_KV_Cache.ipynb) |
| Ch12 | Agent & RAG 原理 | [App1 工具调用](../Applications/App1_ReAct_Agent.ipynb)、[App2 文档检索](../Applications/App2_RAG_System.ipynb) |


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

---

## utils/ 与 Apps 的引用矩阵

| utils 模块 | 使用该模块的 App | 干什么 |
|---|---|---|
| `config.py` | 全部 Apps | 统一 .env 加载 + LLM/Embedding 后端工厂 |
| `llm_backend.py` | 全部 Apps | OpenAI / DashScope / Ollama / HF / vLLM 后端的统一接口 |
| `embedding_backend.py` | App2, App8 | SentenceTransformers / OpenAI / HF embedding |
| `multi_agent.py` | **App4**, App8 | `BaseAgent` + `Orchestrator`（Hierarchical/Debate/Handoff）+ `CircuitBreaker` |
| `mcp_helpers.py` | App3, **App5**, App8 | `EduMCPServer/Client` + `tool_from_function` |
| `observability.py` | App1, App4, **App7**, App8 | `@observe` + `MockObserver` + Langfuse fallback |
| `skills_helpers.py` | **App6**, App8 | Anthropic Skills 解析 + 路由 + progressive disclosure |

---

## 读这门课的最短路径

| 你的目标 | 推荐路径 |
|---|---|
| 只想跑通 Agent | App0 → Ch12 → App1 → App2 → App4 |
| 想理解生产级 Agent 栈 | App0 → App1–App4 → App5 → App6 → App7 → App8 |
| 从头学习 | Ch0–Ch12 → App1–App8 |
