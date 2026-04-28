# 教材构建说明（给维护者）

学员只需读 `README.md`。本文件给维护者：解释 v2 是怎么生成的、怎么重跑、怎么补 cell output、怎么扩练习。

---

## 工作流：3 个脚本 + 1 次预跑

```
assets/enterprise/  (v1 源)
        │
        │ ① cp 到 instructor/，先不改
        ▼
assets/enterprise_ver2/instructor/  (7 本 raw)
        │
        │ ② tools/transform_v2.py   ← 砍肉 + 加📋讲课提示 + Day2 跨本搬 + 打 fillin tag
        ▼
assets/enterprise_ver2/instructor/  (7 本 加工后)
        │
        │ ③ jupyter nbconvert --execute  ← 在 conda llmc 跑通，保留 cell outputs
        ▼
assets/enterprise_ver2/instructor/  (7 本 含 outputs)
        │
        │ ④ tools/derive_student.py  ← 去📋讲课提示 + fillin cell 替换为 TODO 桩 + 清那些 cell 的 output
        ▼
assets/enterprise_ver2/student/  (7 本 学员版)
```

---

## ① 重置 instructor/ 为原始拷贝

如果要从头来：

```bash
for nb in Day0_环境配置与测试 Day1_上午_从文本到向量 Day1_下午_Transformer架构 \
          Day2_上午_预训练与SFT Day2_下午_LoRA对齐与评测 \
          Day3_上午_RAG与Agent实战 Day3_下午_企业知识助手Capstone; do
  cp "assets/enterprise/${nb}.ipynb" "assets/enterprise_ver2/instructor/${nb}.ipynb"
done
```

## ② 跑 transform_v2

```bash
conda activate llmc   # 推荐
python tools/transform_v2.py
```

输出会显示每本的 cell 数变化、删了几个成本/灌水段、打了几个 fillin tag、加了几个 instructor_only tag。

**注意：** `transform_v2.py` **不是幂等的**——重复运行会重复插入 lecture note。要重跑，先回到步骤 ①。

`transform_v2.py` 会做这些事：
- 在每本 notebook 头部插一个"自动定位课程根目录"的 path-fix cell（确保从 instructor/ 或 student/ 都能跑）
- 在每本 notebook 的标题后插一个 📋 讲课提示 markdown（只讲师版含）
- 修补字体相对路径（`../fonts/` → `fonts/`）
- 删除指定的成本计算/商业话术 cells
- 把 Day2_上午 的 SFT 块整体迁移到 Day2_下午 块 1（节奏均衡）
- 合并 Day3_下午 末尾松散的 `# ===` 分隔 cells
- 给所有"练习/Exercise/Checkpoint/Mini-Project"打 `fillin` tag
- 给 📋 讲课提示打 `instructor_only` tag

## ③ 预跑取 cell outputs（耗时步骤）

学员版要看演示效果，所以 instructor 必须**预跑保留 output**。

```bash
conda activate llmc
cd assets/enterprise_ver2

# 一次跑 1 本（GPU 资源紧张时）
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 \
  instructor/Day0_环境配置与测试.ipynb

# 或者全跑（约 30-60 分钟，取决于硬件）
for nb in instructor/*.ipynb; do
  echo "▶ Executing: $nb"
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1200 \
    "$nb" || echo "  ⚠ FAILED: $nb (跳过，单独排查)"
done
```

**已知会慢的 cell：**
- Day2_上午：从零预训练小型 GPT，约 3-8 min（取决于 epoch 数）
- Day2_下午：SFT GPT-2 中文，约 3-8 min；LoRA + DPO 训练各 5-15 min
- Day3 RAG 第一次跑：embedding 生成可能要等 30s 网络

**如果某本失败：**
1. 在 jupyter lab 里手动打开 `instructor/<那本>.ipynb`，跑到失败 cell
2. 看错误：`ModuleNotFoundError`?  `ConnectionError`?  `OutOfMemoryError`?
3. 修后再跑 nbconvert 即可（已成功的 cell 输出会保留）

## ④ 派生 student/

```bash
python tools/derive_student.py
```

输出会显示每本删了几个 lecture cells、blank 了几个 fillin cells、保留了几个 pass-through。

`derive_student.py` 会做这些事：
- 读 `instructor/*.ipynb`
- 删除 `instructor_only` tag 的 cells（📋 讲课提示）
- 处理 `fillin` tag 的代码 cells：
  - 如果有 `↓↓↓ 你的代码 ↓↓↓ ... ↑↑↑ 你的代码 ↑↑↑` 标记 → 只 blank 框内
  - 否则 → 保留开头注释行 + 加 `# TODO: 完成此练习` + `pass`
  - 清空该 cell 的 outputs
- 其它 cells（演示、可视化、讲解） → 原样保留含 outputs
- 保存到 `student/` 同名文件

---

## 验证清单（每次构建后跑一遍）

```bash
# 1. 文件数量
ls assets/enterprise_ver2/instructor/*.ipynb | wc -l   # 期望 7
ls assets/enterprise_ver2/student/*.ipynb | wc -l       # 期望 7

# 2. 砍肉验证（只该剩 1 个：SFT 部署决策备忘录里的"训练资源估算"技术性提法）
grep -rE "节省¥|月省|年节省|节约 ¥" assets/enterprise_ver2/instructor/   # 期望 0
grep -rE "成本计算器" assets/enterprise_ver2/instructor/                   # 期望 0
grep -rE "商业决策分析" assets/enterprise_ver2/instructor/                  # 期望 0

# 3. 学员版无答案
grep -rE "节省¥|月省|年节省|节约 ¥" assets/enterprise_ver2/student/        # 期望 0
grep -rE "成本计算器" assets/enterprise_ver2/student/                       # 期望 0

# 4. 学员版无讲课提示
python -c "
import json, glob
for p in sorted(glob.glob('assets/enterprise_ver2/student/*.ipynb')):
    nb = json.load(open(p, encoding='utf-8'))
    n_lecture = sum(1 for c in nb['cells']
                    if 'instructor_only' in c.get('metadata',{}).get('tags',[]))
    assert n_lecture == 0, f'{p}: 残留讲课提示 {n_lecture} 个！'
    print(f'{p}: ✓')
"

# 5. 学员版有 fillin TODO 桩
python -c "
import json, glob
for p in sorted(glob.glob('assets/enterprise_ver2/student/*.ipynb')):
    nb = json.load(open(p, encoding='utf-8'))
    n = sum(1 for c in nb['cells']
            if c['cell_type']=='code' and 'TODO: 完成' in ''.join(c['source']))
    print(f'{p}: {n} TODO 桩')
"

# 6. instructor 版含 cell outputs（仅在跑过 ③ 后通过）
python -c "
import json, glob
for p in sorted(glob.glob('assets/enterprise_ver2/instructor/*.ipynb')):
    nb = json.load(open(p, encoding='utf-8'))
    n_with_output = sum(1 for c in nb['cells']
                        if c.get('cell_type')=='code' and c.get('outputs'))
    print(f'{p}: {n_with_output} cells 含 output')
"
```

---

## 后续优化（Batch 5）

当前 fillin 模板还停留在「保留代码骨架 + 框内 TODO 桩」级别。

`plan` 第二阶段优化目标是把每个练习改写为：

```python
# 【基础】（人人必做，5 min）
def basic_solution(...):
    return ___   # 1-2 行核心填空

# 【进阶】（技术学员选做，15 min）
def advanced_solution(...):
    # TODO: 完整实现
    pass

verify()  # 自动 assert ✅/❌
```

涉及约 30 个练习，每个需重设计基础题/进阶题/verify 三件套。建议**单独一次会话专做**。

完成时机：当学员上完一遍课、收集到"哪个练习卡壳/简单/无聊"反馈后再做最针对性。

---

## 文件版本控制建议

```
.gitignore：
  assets/enterprise_ver2/.env       # 含 API key，绝不提交
  tools/_day2_sft_staging.json       # 中间产物，每次重跑都会变
```

`student/` 与 `instructor/` 都应入 git，互相是独立可发布的产物。
