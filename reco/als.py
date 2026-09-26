"""手写 ALS 矩阵分解（隐式反馈 WRMF，Hu et al. 2008）。

Tier-1 离线兜底主模型：纯 numpy 实现，确定性，无需任何重型依赖。
置信度 C_ui = 1 + alpha * weight；偏好 p_ui = 1(观测) / 0。
交替最小二乘闭式更新，是工业界隐式反馈推荐的标准强方法。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import ModelError
from ..core.types import InteractionData


class ALSRecommender:
    name = "ALS"
    backend = "tier1"

    def __init__(
        self,
        n_factors: int = 32,
        reg: float = 0.1,
        iterations: int = 15,
        alpha: float = 8.0,
        seed: int = 0,
    ):
        self.n_factors = n_factors
        self.reg = reg
        self.iterations = iterations
        self.alpha = alpha
        self.seed = seed
        self._X: np.ndarray | None = None
        self._Y: np.ndarray | None = None
        self._n_users = 0
        self._n_items = 0
        self._conf: np.ndarray | None = None  # (n_users, n_items) 置信度
        self._pref: np.ndarray | None = None  # (n_users, n_items) 偏好 0/1

    def fit(self, data: InteractionData) -> None:
        self._n_users, self._n_items = data.n_users, data.n_items
        rng = np.random.default_rng(self.seed)

        # 置信度矩阵与偏好矩阵
        pref = data.to_binary_matrix()  # 0/1
        # 用真实权重重建置信度
        W = np.zeros((self._n_users, self._n_items), dtype=np.float64)
        W[data.user_indices, data.item_indices] = data.weights
        conf = 1.0 + self.alpha * W
        self._conf = conf
        self._pref = pref

        # 初始化因子（小随机，确定性）
        self._X = rng.standard_normal((self._n_users, self.n_factors)) * 0.01
        self._Y = rng.standard_normal((self._n_items, self.n_factors)) * 0.01

        for _ in range(self.iterations):
            self._update_users()
            self._update_items()

    def _observed(self, matrix: np.ndarray) -> list[np.ndarray]:
        """返回每行的非零列索引（按行分组）。"""
        out: list[np.ndarray] = []
        for r in range(matrix.shape[0]):
            idx = np.nonzero(matrix[r])[0]
            out.append(idx)
        return out

    def _update_users(self) -> None:
        Y = self._Y
        YtY = Y.T @ Y
        I = np.eye(self.n_factors, dtype=np.float64)
        obs = self._observed(self._pref)
        for u in range(self._n_users):
            cols = obs[u]
            A = YtY.copy()
            rhs = np.zeros(self.n_factors, dtype=np.float64)
            if cols.size:
                c = self._conf[u, cols]  # (m,)
                c_minus = (c - 1.0)[:, None]  # (m,1)
                Yc = Y[cols]  # (m, f)
                A += (c_minus * Yc).T @ Yc  # sum (c-1) y y^T
                rhs = (c[:, None] * Yc).sum(axis=0)  # sum c y
            A += self.reg * I
            self._X[u] = np.linalg.solve(A, rhs)

    def _update_items(self) -> None:
        X = self._X
        XtX = X.T @ X
        I = np.eye(self.n_factors, dtype=np.float64)
        # 转置视角：以物品为「用户」、用户为「物品」
        conf_T = self._conf.T
        pref_T = self._pref.T
        obs = self._observed(pref_T)
        for i in range(self._n_items):
            cols = obs[i]
            A = XtX.copy()
            rhs = np.zeros(self.n_factors, dtype=np.float64)
            if cols.size:
                c = conf_T[i, cols]
                c_minus = (c - 1.0)[:, None]
                Xc = X[cols]
                A += (c_minus * Xc).T @ Xc
                rhs = (c[:, None] * Xc).sum(axis=0)
            A += self.reg * I
            self._Y[i] = np.linalg.solve(A, rhs)

    def scores(self, users: np.ndarray) -> np.ndarray:
        if self._X is None or self._Y is None:
            raise ModelError("ALS 尚未 fit")
        return self._X[users] @ self._Y.T
