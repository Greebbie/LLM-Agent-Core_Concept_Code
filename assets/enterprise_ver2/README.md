# Enterprise AI 3-Day Training (v2)

面向企业班的 LLM 与 Agent 实战教材。3 天 × 6 个 3 小时 session，定位为**混合班偏技术**——保留 attention/训练/LoRA/RAG 的原理推导与实操，但讲法形象、节奏稳，业务背景的同学也能跟上。

---

## 文件夹说明

```
enterprise_ver2/
├── instructor/      # 讲师版（含解答、讲课提示和完整输出；仅供讲师使用）
├── student/         # 学员版（保留填空，outputs 已清空；用于课堂发放）
├── utils/           # 共享工具：统一 LLM/Embedding 后端入口
├── data/            # 教学数据
├── fonts/           # 中文字体（图表渲染）
├── requirements.txt # 依赖
├── .env.example     # 环境变量模板
└── README.md        # 本文件
```

---

## 快速开始

### 1. 准备环境

```bash
# 如果没装过 llmcs，先建好（推荐 Python 3.11；当前课程主环境为 Python 3.11.15）
conda create -n llmcs python=3.11 -y
conda activate llmcs

# 先安装 PyTorch：有 NVIDIA GPU（如 RTX 50 系）优先装 CUDA 12.8 版；CPU 机器把 cu128 改成 cpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 再安装项目依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 API key（推荐 DashScope，国内稳定快速）
```

也可以直接运行 `Day0_环境配置与测试.ipynb` 的 Step 4，只改一行 `DASHSCOPE_API_KEY = ""`。课堂默认模型是 `qwen-plus-2025-01-25` + `text-embedding-v3`，一般不用改模型名。

默认使用固定快照 `qwen-plus-2025-01-25`，便于课堂复现；如需使用最新别名，把 `.env` 里的 `LLM_MODEL` 改成 `qwen-plus` 即可。

### 3. 跑 Day0 验证

```bash
jupyter lab instructor/Day0_环境配置与测试.ipynb
# 学员发布场景：jupyter lab student/Day0_环境配置与测试.ipynb
```

Day0 会自动检查 Python 版本、依赖完整性、API 连通性和字体路径。全部通过后再开课。

---

## 课程节奏

| Day | 上午 (9:00-12:00) | 下午 (14:00-17:00) |
|---|---|---|
| **Day 1** | 从文本到向量（训练循环 / Tokenizer / Embedding / Prompt） | Transformer 架构（Self-Attention / Block / GPT 文本生成） |
| **Day 2** | 预训练完整闭环（小型 GPT 训练 + 生成对比） | SFT 动手 + LoRA + DPO + 评测三件套 |
| **Day 3** | RAG + Agent 实战（向量检索 / ReAct / Code Agent） | 企业知识助手 Capstone（拼装演示） |

每个 session 含 1 次 10-15 min 休息。Day0 环境配置约 20 min，建议开班前自助完成。

---

## 与主线章节的对应关系

3 天版是主线内容的压缩版，适合先交付基础能力：

- Day 1：覆盖 Ch0-Ch6 的核心概念，重点是训练循环、Tokenizer、Embedding、Self-Attention、Transformer Block 和 GPT 生成。
- Day 2：覆盖 Ch7-Ch10 的训练链路，重点是预训练、SFT、LoRA、DPO 与评估边界。
- Day 3：覆盖 Ch12 与 Applications App1-App4 的基础应用，落到 RAG、Agent、Code Agent 和企业知识助手 Capstone。

3 天版不包含 5 天版新增的 MCP、Skills、Agentic RAG 与 LLMOps；这些内容放在 `../enterprise_5days/` 的 Day 4-5。这样分层更清楚：3 天版负责建立主线能力，5 天版负责补齐 Agent 工程化与生产前验证。

---

## 填空 / 练习设计

每个练习采用**双轨模板**：

```python
# 【基础】（人人必做，5 min）
def basic_solution(...):
    return ___   # 1-2 行核心填空，提示充分

# 【进阶】（技术学员选做，15 min）
def advanced_solution(...):
    # 继续补全完整实现
    pass

verify()  # 用断言检查关键结果
```

业务背景的学员稳拿基础分；技术学员可冲进阶；verify() 立即知道做对没。

---

## 给讲师的发布约定

- **务必发 `student/` 版**给学员（已剥离解答和讲课提示）
- `instructor/` 版**不要进学员群**——含全部解答和章节开头的讲课提示（含开场提问、互动设计、常见误解、时间紧时跳哪段）
- 上课时讲师在自己屏幕上开 instructor 版，学员屏幕开 student 版

---

## 后端切换

教材已统一通过 `utils/config.py:env` 入口切换 LLM 与 embedding 后端，**Day0 完整讲一次**，其余 notebook 一句 import 完事：

```python
from utils.config import env
llm = env.get_llm()         # 自动按 .env 里的 LLM_BACKEND 加载
emb = env.get_embedder()    # 自动按 .env 里的 EMBEDDING_BACKEND 加载
```

支持的后端：
- `dashscope`（推荐国内 / qwen-plus-2025-01-25 / text-embedding-v3）
- `openai`（gpt-4o-mini / text-embedding-3-small）
- `ollama`（本地 / qwen2.5 / nomic-embed-text）
- `huggingface`（本地 / sentence-transformers）

切换时只改 `.env` 一处，notebook 不动。

---

## 与 v1 (`assets/enterprise/`) 的差异

- **新增 instructor/student 双版本** —— 学员有参考答案视觉效果可看
- **删除"成本计算器/年节省¥XX"等灌水内容** —— 把时间还给原理与实操
- **Day2 节奏重排** —— SFT 动手部分从上午搬到下午，避免单本过载
- **填空模板分层** —— 基础+进阶+verify()，混合班双轨进阶
- **每章节开头加「📋 讲课提示」**（仅讲师版）—— 取代独立讲课笔记
- **conda llmcs 环境约定** —— 所有 notebook 头部统一指向

---

## 验证

```bash
# 跑通验证（在 conda llmcs 环境）
for nb in instructor/*.ipynb; do
  jupyter nbconvert --to notebook --execute "$nb" \
    --output "${nb%.ipynb}_check.ipynb" \
    --ExecutePreprocessor.timeout=600
done

# 砍肉验证（应返回 0 行）
grep -rE "节省¥|月省|年节省|成本计算器" .

# 填空数量验证
python -c "
import json, glob
for nb in sorted(glob.glob('student/*.ipynb')):
    n = json.load(open(nb, encoding='utf-8'))
    fillins = sum(1 for c in n['cells'] if c['cell_type']=='code' and '___' in ''.join(c['source']))
    print(f'{nb}: {fillins} fillins')
"
```

---

## 已知限制

- 训练类演示 cell（Day2 预训练 / SFT / LoRA）用 tiny dataset + 少 epoch，跑通约 5 min；output 演示效果即可，生产场景请用大数据多轮训练
- Day0 与 utils 假定 `.env` 与 notebook 在同一目录运行；如改换执行路径请调整 `.env` 加载逻辑

---

## License & 反馈

教材内部使用。错漏与改进建议直接 issue。
