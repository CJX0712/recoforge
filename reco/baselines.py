"""基线推荐器（强基线对照用）。

全部为 O(1)~O(n) 简单模型，作为「naive/经典方法」对照，验证主方法显著优于之。
"""

from __future__ import annotations

import numpy as np

from ..core.types import InteractionData


class PopularityRecommender:
    """按物品全局热度排序（最朴素的强基线之一）。"""

    name = "Popularity"
    backend = "tier1"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._pop: np.ndarray | None = None

    def fit(self, data: InteractionData) -> None:
        pop = np.zeros(data.n_items, dtype=np.float64)
        np.add.at(pop, data.item_indices, data.weights)
        # 平滑：未交互物品热度置为全局最小，保持全量可排
        if pop.max() > 0:
            pop = pop / pop.max()
        self._pop = pop

    def scores(self, users: np.ndarray) -> np.ndarray:
        if self._pop is None:
            raise RuntimeError("请先 fit")
        return np.tile(self._pop, (len(users), 1))


class GlobalMeanRecommender:
    """全局均值常量打分（最弱基线）。"""

    name = "GlobalMean"
    backend = "tier1"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._mean: float = 0.0
        self._n_items: int = 0

    def fit(self, data: InteractionData) -> None:
        self._mean = float(np.mean(data.weights)) if data.n_interactions else 0.0
        self._n_items = data.n_items

    def scores(self, users: np.ndarray) -> np.ndarray:
        return np.full((len(users), self._n_items), self._mean, dtype=np.float64)


class RandomRecommender:
    """固定 seed 的随机打分（消融 / 下限基线）。"""

    name = "Random"
    backend = "tier1"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._scores: np.ndarray | None = None
        self._n_items: int = 0

    def fit(self, data: InteractionData) -> None:
        rng = np.random.default_rng(self.seed)
        self._scores = rng.random((data.n_users, data.n_items))
        self._n_items = data.n_items

    def scores(self, users: np.ndarray) -> np.ndarray:
        if self._scores is None:
            raise RuntimeError("请先 fit")
        return self._scores[users]
