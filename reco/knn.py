"""ItemKNN 近邻协同过滤（经典方法，作为对照基线 / 主方法候选）。

物品相似度基于「共同交互用户」的余弦相似度；用户 u 对物品 i 的打分 =
u 已交互物品与 i 的相似度之和。全手写，零 sklearn 依赖。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import ModelError
from ..core.types import InteractionData


class ItemKNNRecommender:
    name = "ItemKNN"
    backend = "tier1"

    def __init__(self, k: int = 0, seed: int = 0):
        """
        k : 仅保留 top-k 近邻（0=全部）。控制平滑度。
        """
        self.k = k
        self.seed = seed
        self._sim: np.ndarray | None = None
        self._train_bin: np.ndarray | None = None

    def fit(self, data: InteractionData) -> None:
        B = data.to_binary_matrix()  # (n_users, n_items)
        self._train_bin = B
        I = B.T  # (n_items, n_users)：每行一个物品
        norms = np.sqrt(np.sum(I * I, axis=1))
        safe = np.where(norms > 0, norms, 1.0)
        In = I / safe[:, None]
        sim = In @ In.T  # (n_items, n_items)
        # 去除自相似，避免数值噪声
        np.fill_diagonal(sim, 0.0)
        if self.k and self.k > 0:
            sim = self._keep_top_k(sim, self.k)
        self._sim = sim

    @staticmethod
    def _keep_top_k(sim: np.ndarray, k: int) -> np.ndarray:
        out = np.zeros_like(sim)
        rows = sim.shape[0]
        for i in range(rows):
            row = sim[i]
            if row.size == 0:
                continue
            # 取 top-k（含负相似度时也能取到最大的 k 个）
            n = min(k, row.size)
            thr = np.partition(row, row.size - n)[row.size - n]
            mask = row >= thr
            out[i, mask] = row[mask]
        return out

    def scores(self, users: np.ndarray) -> np.ndarray:
        if self._sim is None or self._train_bin is None:
            raise ModelError("ItemKNN 尚未 fit")
        B = self._train_bin[users]  # (len(users), n_items)
        # scores[u,i] = sum_j B[u,j] * sim[i,j] = (B @ sim.T)[u,i]
        return B @ self._sim.T
