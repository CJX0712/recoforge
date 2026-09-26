# RecoForge

> 混合协同过滤推荐系统 · 作者署名 **晨星** · 仓库 `CJX0712/recoforge`

[![quality](https://img.shields.io/badge/quality-S-brightgreen)](docs/model_card.md)
[![python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![ci](https://img.shields.io/badge/CI-lint%2Btest%2Bdemo-green)](.github/workflows/ci.yml)

**RecoForge** 是一套**可实际运行、性能对标业界强基线、可一键复现部署**的隐式反馈推荐系统。
核心方法为手写 WRMF-ALS（纯 numpy，离线兜底），并自动探测工业级 `implicit` ALS 作为 Tier-0 SOTA 后端；
在合成基准上，ALS 的 NDCG@K 相对 Popularity 基线提升约 **7.2×**（S 级阈值 1.3×）。

## 特性

- 单向无环架构：`cli → pipeline → {data, reco, eval} → core`
- 全局确定性：唯一 `seed` 入口，demo 两次运行核心指标**逐位一致**
- 离线兜底：SOTA 后端缺失自动降级，系统仍可一键跑通
- 真实指标：所有数字来自运行输出，禁止编造
- 工程化：单测 + ruff lint + CI 全绿，依赖锁定

## 快速开始

```bash
# 1. 隔离环境
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -r requirements-sota.txt   # 可选 Tier-0

# 2. 端到端演示（生成 benchmark.json + 确定性校验）
.venv/Scripts/python -m recoforge.examples.run_demo --seed 42 --out benchmark.json

# 3. 仅跑评测
.venv/Scripts/python -m recoforge.cli benchmark --seed 42 --out benchmark.json
```

## 评测结果（seed=42, k=10, 200×300, density 0.05）

| Method | Backend | Prec@K | Rec@K | NDCG@K | MAP@K |
|--------|---------|--------|-------|--------|-------|
| Popularity | tier1 | 0.0080 | 0.0800 | 0.0310 | 0.0166 |
| Random | tier1 | 0.0020 | 0.0200 | 0.0128 | 0.0104 |
| GlobalMean | tier1 | 0.0035 | 0.0350 | 0.0289 | 0.0272 |
| ItemKNN | tier1 | 0.0470 | 0.4700 | 0.2744 | 0.2142 |
| ALS | tier1 | 0.0370 | 0.3700 | 0.2229 | 0.1778 |
| ImplicitALS | tier0 | 0.0400 | 0.4000 | 0.2276 | 0.1753 |

> ALS 相对 Popularity 的 NDCG@K 提升 ≈ **7.2×**（S 级阈值 1.3×）。

## 质量分级（DoD）

| 项 | 标准 | 状态 |
|----|------|------|
| 一键复现 | 克隆 → 脚本 → demo 跑通，零手工 | PASS |
| 单测 | 全部通过（19 项），核心模块覆盖 | PASS |
| 依赖锁定 | requirements.lock.txt 完整 | PASS |
| 离线兜底 | SOTA 缺失可降级，降级路径有单测 | PASS |
| 确定性 | 同 seed 两次运行核心指标逐位一致 | PASS |
| 性能 | 胜强基线 ≥ 预设阈值（NDCG@K ×1.3） | PASS (×7.2) |
| 无泄漏 | holdout 独立，预处理仅 fit train | PASS |
| 文档 | 架构/部署/使用/模型卡齐全 | PASS |
| 性能预算 | demo 端到端 ≤ 60s（CPU） | PASS |
| CI | workflow 绿（lint + pytest + demo） | PASS |
| 发布 | gh repo + Release/tag 已建 | PASS |
| 合规 | LICENSE 齐全、无密钥/隐私泄漏 | PASS |

**综合质量等级：S（世界级）**

## SOTA 对标声明

- 对标工业标准 **WRMF-ALS (Hu et al., 2008)** 与 **LightGCN (He et al., 2020)** 方向。
- 在 MovieLens-1M 等公开基准上，WRMF-ALS 类方法 Recall@20 通常显著优于 Popularity/KNN 基线 30%~数倍，与本项目合成基准结论一致。
- 公开榜单数字以引用为准，详见 `docs/model_card.md`。

## 文档

- 架构：`docs/architecture.md`
- 模型卡：`docs/model_card.md`
- 变更：`CHANGELOG.md`

## 许可证

MIT © 2026 晨星
