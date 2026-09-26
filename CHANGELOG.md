# Changelog

## v0.1.0 (2026-09-27)

- 初始交付：混合协同过滤推荐系统 RecoForge
- 方法：手写 WRMF-ALS（Tier-1 离线兜底）+ `implicit` ALS（Tier-0 SOTA 后端，自动探测降级）
- 基线：Popularity / Random / GlobalMean；经典：ItemKNN
- 指标：Precision/Recall/NDCG/MAP@K（全手写，防 sklearn 递归坑）
- 工程：单测 19 项、ruff lint、CI workflow、依赖锁定、Dockerfile、确定性校验
- 质量等级：S（ALS NDCG@K 相对 Popularity 提升 ≈ 7.2×）
- 作者署名：晨星 · 仓库：CJX0712/recoforge
