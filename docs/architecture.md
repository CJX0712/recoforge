# RecoForge 架构文档

> 作者：晨星 · 版本：0.1.0 · 域：推荐系统（隐式反馈协同过滤）

## 1. 设计原则

- **单向无环**：`cli → pipeline → {data, reco, eval} → core`，禁止反向依赖与环状调用。
- **接口契约**：`Recommender / DataSource / Splitter` 均为 Protocol，便于替换实现与离线兜底。
- **全局确定性**：唯一入口 `core.seed.set_all(seed)`，numpy/random/库级一次设齐；demo 两次运行核心指标逐位一致。
- **离线兜底优先**：SOTA 后端（`implicit`）缺失时自动降级，系统仍可一键跑通。
- **指标真实**：`benchmark.json` 每个数字来自真实运行，禁止编造（见 V3 §4）。

## 2. 目录结构

```
recoforge/
  core/        types · errors(E100~E500) · config(ENV 覆盖+schema校验) · interfaces(Protocol) · seed(确定性)
  data/        synthetic(合成数据) · loaders(npz 载入/落盘)
  reco/        baselines( Popularity/Random/GlobalMean ) · knn(ItemKNN) · als(手写WRMF) · implicit_backend(Tier-0)
  eval/        metrics(手写 Precision/Recall/NDCG/MAP@K)
  preprocess/  split(每用户留一法，防泄漏)
  pipeline/    RecommendationPipeline.run() + benchmark() + grade_report()
  cli.py       argparse 入口
  examples/run_demo.py  端到端演示（落盘 benchmark.json + 确定性校验）
tests/         pytest 单测（含 CLI 冒烟 + 离线兜底路径）
docs/          architecture.md · model_card.md
```

## 3. 数据流向

```
set_all(seed)
   │
   ▼
generate_dataset() ──► InteractionData
   │
   ▼
leave_n_out_split() ──► SplitResult(train, test)   # 训练/测试交互互不相交
   │
   ▼
for method in [Popularity, Random, GlobalMean, ItemKNN, ALS, (ImplicitALS)]:
   method.fit(train) ──► scores = method.scores(users)   # (n_users, n_items)
   │
   ▼
top_k_metrics(scores, train_mask, test_items, k) ──► MetricResult
   │
   ▼
BenchmarkReport(results, ablation, baseline_reference, environment)
```

## 4. 关键模块职责

| 模块 | 职责 | 不变量 |
|------|------|--------|
| `core.seed` | 全局确定性 | 同 seed → 同随机序列 |
| `data.synthetic` | 生成带潜在结构的隐式反馈 | 每用户交互数 ≥ 切分所需 |
| `preprocess.split` | 每用户留一法 | 训练/测试集合不相交（防泄漏） |
| `reco.als` | 手写 WRMF-ALS | 闭式更新，确定性收敛 |
| `reco.implicit_backend` | 封装 implicit ALS | `available_implicit()` 探测，缺失降级 |
| `eval.metrics` | Top-K 排序指标 | 排除训练物品后排名 |
| `pipeline` | 编排 + 报告 + 分级 | 数字全部来自运行 |

## 5. 防数据泄漏设计

- 切分仅在「用户自身交互」内做确定性置换，绝不跨用户全局 shuffle。
- 评测时通过 `train_mask` 将训练已交互物品置为 `-inf`，从排名中排除。
- scaler / 特征选择 / HPO 在此任务中无跨折拟合需求，交互矩阵直接 fit 于 train 折。

## 6. 确定性保障

- `generate_dataset` 使用 `np.random.default_rng(seed)` 显式种子。
- ALS 初始化与迭代均确定性；`implicit` 传 `random_state=seed`。
- `run_demo` 连续运行两次并断言核心指标逐位相等，失败即报错。

## 7. 复现命令

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -r requirements-sota.txt   # 可选 Tier-0
.venv/Scripts/python -m recoforge.examples.run_demo --seed 42 --out benchmark.json
```
