"""推荐流水线：生成 -> 切分 -> 多方法评测 -> 消融 -> 报告。

单向无环：pipeline -> {data, reco, eval} -> core。
所有数字来自真实运行输出，杜绝编造（V3 §4 指标幻觉坑）。
"""

from __future__ import annotations

import platform
import sys

import numpy as np

from ..core.config import RecoForgeConfig
from ..core.seed import set_all
from ..core.types import BenchmarkReport, InteractionData, MetricResult, SplitResult
from ..data.synthetic import generate_dataset
from ..eval.metrics import top_k_metrics
from ..preprocess.split import leave_n_out_split
from ..reco import (
    ALSRecommender,
    GlobalMeanRecommender,
    ImplicitALSRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    RandomRecommender,
    available_implicit,
)


def _build_train_mask(split: SplitResult) -> np.ndarray:
    n_users = split.train.n_users
    n_items = split.train.n_items
    mask = np.zeros((n_users, n_items), dtype=bool)
    mask[split.train.user_indices, split.train.item_indices] = True
    return mask


def _build_test_items(split: SplitResult) -> list[np.ndarray]:
    n_users = split.test.n_users
    items: list[list[int]] = [[] for _ in range(n_users)]
    for u, i in zip(split.test.user_indices.tolist(), split.test.item_indices.tolist()):
        items[int(u)].append(int(i))
    return [np.array(x, dtype=np.int64) for x in items]


class RecommendationPipeline:
    def __init__(self, config: RecoForgeConfig | None = None):
        self.config = config or RecoForgeConfig()

    def run(self, data: InteractionData | None = None) -> BenchmarkReport:
        set_all(self.config.seed)
        cfg = self.config
        if data is None:
            data = generate_dataset(
                n_users=cfg.n_users,
                n_items=cfg.n_items,
                n_factors=8,
                density=cfg.density,
                seed=cfg.seed,
                noise_std=1.0,
                n_test_per_user=cfg.n_test_per_user,
            )
        split = leave_n_out_split(
            data, n_test_per_user=cfg.n_test_per_user, seed=cfg.seed
        )
        return self.benchmark(split)

    def benchmark(self, split: SplitResult) -> BenchmarkReport:
        cfg = self.config
        k = cfg.k
        train_mask = _build_train_mask(split)
        test_items = _build_test_items(split)
        n_users = split.train.n_users
        users = np.arange(n_users)

        # 方法清单（Tier-1 全跑；Tier-0 视可用性）
        methods = [
            PopularityRecommender(seed=cfg.seed),
            RandomRecommender(seed=cfg.seed),
            GlobalMeanRecommender(seed=cfg.seed),
            ItemKNNRecommender(seed=cfg.seed),
            ALSRecommender(
                n_factors=cfg.n_factors,
                reg=cfg.reg,
                iterations=cfg.iterations,
                alpha=cfg.alpha,
                seed=cfg.seed,
            ),
        ]
        if available_implicit():
            methods.append(
                ImplicitALSRecommender(
                    factors=cfg.n_factors,
                    reg=cfg.reg,
                    iterations=cfg.iterations,
                    alpha=cfg.alpha,
                    seed=cfg.seed,
                )
            )

        results: list[MetricResult] = []
        for rec in methods:
            try:
                rec.fit(split.train)
                scores = rec.scores(users)
                m = top_k_metrics(scores, train_mask, test_items, k)
                results.append(
                    MetricResult(
                        method=rec.name,
                        backend=rec.backend,
                        available=True,
                        **m,
                    )
                )
            except Exception as exc:  # noqa: BLE001  # 单方法失败不影响整体（离线兜底原则）
                results.append(
                    MetricResult(
                        method=rec.name,
                        backend=rec.backend,
                        available=False,
                        error=str(exc),
                        precision_at_k=0.0,
                        recall_at_k=0.0,
                        ndcg_at_k=0.0,
                        map_at_k=0.0,
                    )
                )

        # 消融：正则化开关（ALS reg=0 vs reg=config.reg）
        ablation = self._ablation(split, train_mask, test_items, users, k)

        report = BenchmarkReport(
            seed=cfg.seed,
            k=k,
            dataset={
                "n_users": split.train.n_users,
                "n_items": split.train.n_items,
                "n_train": split.train.n_interactions,
                "n_test": split.test.n_interactions,
                "density": round(
                    split.train.n_interactions
                    / (split.train.n_users * split.train.n_items),
                    4,
                ),
                "split": "leave-n-out",
            },
            results=[r.__dict__ for r in results],
            ablation=ablation,
            baseline_reference=self._baseline_reference(),
            environment=self._environment(),
        )
        return report

    def _ablation(self, split, train_mask, test_items, users, k) -> list[dict]:
        out: list[dict] = []
        try:
            base = ALSRecommender(
                n_factors=self.config.n_factors,
                reg=self.config.reg,
                iterations=self.config.iterations,
                alpha=self.config.alpha,
                seed=self.config.seed,
            )
            base.fit(split.train)
            base_scores = base.scores(users)
            base_m = top_k_metrics(base_scores, train_mask, test_items, k)

            no_reg = ALSRecommender(
                n_factors=self.config.n_factors,
                reg=0.0,
                iterations=self.config.iterations,
                alpha=self.config.alpha,
                seed=self.config.seed,
            )
            no_reg.fit(split.train)
            no_reg_scores = no_reg.scores(users)
            no_reg_m = top_k_metrics(no_reg_scores, train_mask, test_items, k)

            out.append(
                {
                    "component": "ALS regularization",
                    "variant_a": f"reg={self.config.reg}",
                    "variant_b": "reg=0 (off)",
                    "metric": "ndcg_at_k",
                    "value_a": round(base_m["ndcg_at_k"], 4),
                    "value_b": round(no_reg_m["ndcg_at_k"], 4),
                    "delta": round(base_m["ndcg_at_k"] - no_reg_m["ndcg_at_k"], 4),
                }
            )
        except Exception as exc:  # noqa: BLE001
            out.append({"component": "ALS regularization", "error": str(exc)})
        return out

    @staticmethod
    def _baseline_reference() -> dict:
        return {
            "strong_baseline": "Popularity / ItemKNN (经典协同过滤)",
            "sota_reference": "WRMF-ALS (Hu et al. 2008); LightGCN (He et al. 2020)",
            "note": "本系统对标隐式反馈 CF 工业标准；公开榜单数字见 model_card.md",
        }

    @staticmethod
    def grade_report(report: BenchmarkReport, threshold_ratio: float = 1.3) -> dict:
        """质量分级（性能维度）：以 ALS 相对 Popularity 的 NDCG@K 提升判定。

        注意：完整 DoD（CI / Release / 文档）需在交付阶段另行校验，此处仅给性能分级参考。
        """
        res = {r["method"]: r for r in report.results}
        als = res.get("ALS")
        pop = res.get("Popularity")
        if not als or not pop:
            return {"grade": "C", "reason": "缺少 ALS 或 Popularity 结果"}
        ratio = (
            als["ndcg_at_k"] / pop["ndcg_at_k"]
            if pop["ndcg_at_k"] > 0
            else float("inf")
        )
        implicit = res.get("ImplicitALS")
        implicit_available = bool(implicit and implicit.get("available", True))
        if ratio >= threshold_ratio and implicit_available:
            grade, reason = "S", "击败强基线且 Tier-0 SOTA 后端可用"
        elif ratio >= threshold_ratio:
            grade, reason = "A", "击败强基线（Tier-0 后端缺失，降级为生产级）"
        elif ratio >= 1.0:
            grade, reason = "B", "仅平 baseline"
        else:
            grade, reason = "C", "未击败 baseline"
        return {
            "grade": grade,
            "reason": reason,
            "als_ndcg_at_k": round(als["ndcg_at_k"], 4),
            "pop_ndcg_at_k": round(pop["ndcg_at_k"], 4),
            "ratio": round(ratio, 4),
            "threshold_ratio": threshold_ratio,
            "implicit_available": implicit_available,
        }

    @staticmethod
    def _environment() -> dict:
        env: dict = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "implicit_available": available_implicit(),
        }
        for mod in ("numpy", "scipy", "sklearn"):
            try:
                m = __import__(mod)
                env[mod] = getattr(m, "__version__", "unknown")
            except Exception:  # noqa: BLE001
                env[mod] = "unavailable"
        try:
            import implicit

            env["implicit"] = getattr(implicit, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            env["implicit"] = "unavailable"
        return env
