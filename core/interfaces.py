"""接口契约（Protocol）。

单向无环调用顺序：cli -> pipeline -> {data, reco, eval} -> core。
各模块面向接口编程，便于替换实现与离线兜底。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..core.types import InteractionData, SplitResult


@runtime_checkable
class Recommender(Protocol):
    """推荐器契约：输入训练交互，输出 (n_users, n_items) 评分矩阵（越大越相关）。"""

    name: str
    backend: str

    def fit(self, data: InteractionData) -> None: ...
    def scores(self, users: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class DataSource(Protocol):
    """数据源契约：产出可复现的交互数据。"""

    def generate(self, seed: int) -> InteractionData: ...


@runtime_checkable
class Splitter(Protocol):
    """切分器契约：产出训练/测试互不相交的切分。"""

    def split(self, data: InteractionData) -> SplitResult: ...
