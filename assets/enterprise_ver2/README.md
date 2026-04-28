# Enterprise AI 3-Day Training (v2)

面向企业班的 LLM 与 Agent 实战教材。3 天 × 6 个 3 小时 session，定位为**混合班偏技术**——保留 attention/训练/LoRA/RAG 的原理推导与实操，但讲法形象、节奏稳，业务背景的同学也能跟上。

---

## 文件夹说明

```
enterprise_ver2/
├── instructor/      # 讲师版（含解答 + 内嵌讲课提示 + 全 cell output）⚠️ 不要群发学员
├── student/         # 学员版（留填空 + 演示类 output 保留）✅ 发给学员
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
# 推荐使用 conda 隔离环境
conda activate llmc

# 如果没装过 llmc，先建好（Python 3.10+）
conda create -n llmc python=3.10 -y
conda activate llmc
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 API key（推荐 DashScope，国内稳定快速）
```

### 3. 跑 Day0 验证

```bash
jupyter lab instructor/Day0_环境配置与测试.ipynb
# 学员发布场景：jupyter lab student/Day0_环境配置与测试.ipynb
```

Day0 会自动检查 Python 版本、依赖完整性、API 连通性、字体路径。**全部 ✅ 后再开课。**

---

## 课程节奏

| Day | 上午 (9:00-12:00) | 下午 (14:00-17:00) |
|---|---|---|
| **Day 1** | 从文本到向量（训练循环 / Tokenizer / Embedding / Prompt） | Transformer 架构（Self-Attention / Block / GPT 文本生成） |
| **Day 2** | 预训练完整闭环（小型 GPT 训练 + 生成对比） | SFT 动手 + LoRA + DPO + 评测三件套 |
| **Day 3** | RAG + Agent 实战（向量检索 / ReAct / Code Agent） | 企业知识助手 Capstone（拼装演示） |

每个 session 含 1 次 10-15 min 休息。Day0 环境配置约 20 min，建议开班前自助完成。

---

## 填空 / 练习设计

每个练习采用**双轨模板**：

```python
# 【基础】（人人必做，5 min）
def basic_solution(...):
    return ___   # 1-2 行核心填空，提示充分

# 【进阶】（技术学员选做，15 min）
def advanced_solution(...):
    # TODO: 完整实现
    pass

verify()  # 自动 assert，✅/❌ 即时反馈
```

业务背景的学员稳拿基础分；技术学员可冲进阶；verify() 立即知道做对没。

---

## 给讲师的发布约定

- **务必发 `student/` 版**给学员（已剥离解答和讲课提示）
- `instructor/` 版**不要进学员群**——含全部解答 + 章节开头的「📋 讲课提示」（含开场提问、互动设计、常见误解、时间紧时跳哪段）
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
- `dashscope`（推荐国内 / qwen-plus / text-embedding-v3）
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
- **conda llmc 环境约定** —— 所有 notebook 头部统一指向

---

## 验证

```bash
# 跑通验证（在 conda llmc 环境）
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
