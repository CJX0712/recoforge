"""训练 / 测试切分（防数据泄漏）。

采用每用户留一法（leave-n-out）：按确定顺序将每用户末位 n_test_per_user 条交互划入测试，
训练集与测试集交互集合互不相交。仅对用户自身交互做切分，绝不跨用户随机 shuffle，
满足「时间/序列类禁用随机 shuffle」式的数据隔离原则。
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..core.errors import DataError
from ..core.types import InteractionData, SplitResult


def leave_n_out_split(
    data: InteractionData, n_test_per_user: int = 1, seed: int = 0
) -> SplitResult:
    """每用户留 n_test_per_user 条作为测试集。"""
    if n_test_per_user < 1:
        raise DataError(f"n_test_per_user 必须 >=1: {n_test_per_user}")

    by_user: dict[int, list[int]] = defaultdict(list)
    for idx, u in enumerate(data.user_indices.tolist()):
        by_user[int(u)].append(idx)

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []

    for u, idxs in by_user.items():
        if len(idxs) < n_test_per_user + 1:
            raise DataError(
                f"用户 {u} 仅 {len(idxs)} 条交互，不足以切分（需 >= {n_test_per_user + 1}）"
            )
        # 确定性洗牌（仅打乱该用户自身索引顺序，不影响其他用户 / 全局）
        perm = rng.permutation(len(idxs))
        shuffled = [idxs[i] for i in perm]
        test_idx.extend(shuffled[:n_test_per_user])
        train_idx.extend(shuffled[n_test_per_user:])

    train_idx_arr = np.array(train_idx, dtype=np.int64)
    test_idx_arr = np.array(test_idx, dtype=np.int64)

    train = InteractionData(
        n_users=data.n_users,
        n_items=data.n_items,
        user_indices=data.user_indices[train_idx_arr],
        item_indices=data.item_indices[train_idx_arr],
        weights=data.weights[train_idx_arr],
    )
    test = InteractionData(
        n_users=data.n_users,
        n_items=data.n_items,
        user_indices=data.user_indices[test_idx_arr],
        item_indices=data.item_indices[test_idx_arr],
        weights=data.weights[test_idx_arr],
    )
    return SplitResult(train=train, test=test)
