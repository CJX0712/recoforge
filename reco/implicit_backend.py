"""Tier-0 SOTA 后端封装：implicit 库 ALS（工业级隐式反馈实现）。

装得上才用：`available_implicit()` 探测；缺失时 benchmark 自动跳过并在报告中标注。
确定性依赖 implicit 的 random_state 参数（固定 seed）。
"""

from __future__ import annotations

import importlib.util

import numpy as np

from ..core.errors import ModelError
from ..core.types import InteractionData

IMPLICIT_AVAILABLE = importlib.util.find_spec("implicit") is not None


def available_implicit() -> bool:
    return IMPLICIT_AVAILABLE


class ImplicitALSRecommender:
    name = "ImplicitALS"
    backend = "tier0"

    def __init__(
        self,
        factors: int = 32,
        reg: float = 0.1,
        iterations: int = 15,
        alpha: float = 8.0,
        seed: int = 0,
    ):
        self.factors = factors
        self.reg = reg
        self.iterations = iterations
        self.alpha = alpha
        self.seed = seed
        self._model = None
        self._n_users = 0
        self._n_items = 0

    def fit(self, data: InteractionData) -> None:
        if not IMPLICIT_AVAILABLE:
            raise ModelError("implicit 后端不可用，已自动降级")
        import os

        import implicit
        import scipy.sparse as sp

        # 抑制 implicit 的 OpenBLAS 多线程性能告警，保持 demo 输出干净
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        rows = data.user_indices.astype(np.int64)
        cols = data.item_indices.astype(np.int64)
        vals = (1.0 + self.alpha * data.weights).astype(np.float64)
        C = sp.csr_matrix((vals, (rows, cols)), shape=(data.n_users, data.n_items))
        # 直接以 (users, items) 送入：implicit 的 user_factors/item_factors 分别对齐
        # 矩阵的行/列轴，即 user_factors -> 用户、item_factors -> 物品，便于后续 dot 计算评分。
        np.random.seed(self.seed)
        self._model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.reg,
            iterations=self.iterations,
            random_state=self.seed,
        )
        self._model.fit(C)
        self._n_users = data.n_users
        self._n_items = data.n_items

    def scores(self, users: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ModelError("ImplicitALS 尚未 fit")
        uf = self._model.user_factors
        itf = self._model.item_factors
        return uf[np.asarray(users)] @ itf.T
