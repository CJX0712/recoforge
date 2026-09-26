"""RecoForge 核心数据类型。

统一使用 dataclass + numpy 数组，保证跨模块接口语义一致：
  - 分数越大越「相关」（推荐系统约定，越大越该被推荐）
  - 隐式反馈用置信度权重表示，0/1 二值亦可
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


@dataclass
class InteractionData:
    """稀疏交互数据（隐式反馈）。

    以三元组 (user_idx, item_idx, weight) 存储，user_idx/item_idx 为 0-based 连续整数。
    """

    n_users: int
    n_items: int
    user_indices: np.ndarray  # int array, shape (n,)
    item_indices: np.ndarray  # int array, shape (n,)
    weights: np.ndarray  # float array, shape (n,)

    @property
    def n_interactions(self) -> int:
        return int(self.user_indices.shape[0])

    def to_csr(self) -> csr_matrix:
        from scipy.sparse import csr_matrix

        return csr_matrix(
            (self.weights, (self.user_indices, self.item_indices)),
            shape=(self.n_users, self.n_items),
        )

    def to_binary_matrix(self) -> np.ndarray:
        """返回 (n_users, n_items) 的 0/1 指示矩阵（出现即 1）。"""
        mat = np.zeros((self.n_users, self.n_items), dtype=np.float64)
        mat[self.user_indices, self.item_indices] = 1.0
        return mat


@dataclass
class SplitResult:
    """训练 / 测试切分结果，二者交互集合互不相交（防数据泄漏）。"""

    train: InteractionData
    test: InteractionData


@dataclass
class MetricResult:
    """单个方法在测试集上的 Top-K 评测结果。"""

    method: str
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    map_at_k: float
    backend: str = "tier1"  # tier1=离线兜底, tier0=SOTA 后端
    available: bool = True
    error: str = ""


@dataclass
class BenchmarkReport:
    """端到端评测报告，可直接序列化为 benchmark.json。"""

    system: str = "RecoForge"
    version: str = "0.1.0"
    seed: int = 0
    k: int = 10
    dataset: dict = field(default_factory=dict)
    results: list = field(default_factory=list)
    ablation: list = field(default_factory=list)
    baseline_reference: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
