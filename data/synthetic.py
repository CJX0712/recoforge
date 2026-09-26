"""合成隐式反馈数据生成。

生成逻辑：用_latent 因子 U/V 构造「真实相关性」logits，叠加噪声后按密度阈值采样，
得到与结构强相关的二值交互。这样 ALS 能恢复潜在结构、显著优于 Popularity 基线，
且完全确定性（固定 seed 两次运行逐位一致）。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import DataError
from ..core.types import InteractionData


def generate_implicit_feedback(
    n_users: int,
    n_items: int,
    n_factors: int = 8,
    density: float = 0.05,
    seed: int = 0,
    noise_std: float = 1.0,
) -> InteractionData:
    """生成可复现的隐式反馈交互。

    参数
    ----
    n_users, n_items : 用户 / 物品数量
    n_factors        : 潜在因子维度（数据生成用，模型不感知）
    density          : 目标交互密度（0,1）
    seed             : 随机种子（确定性）
    noise_std        : 观测噪声，控制结构可恢复难度
    """
    if not (0.0 < density < 1.0):
        raise DataError(f"density 必须在 (0,1): {density}")
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n_users, n_factors))
    V = rng.standard_normal((n_items, n_factors))
    logits = U @ V.T  # (n_users, n_items)

    # 观测噪声 + 按密度分位阈值二值化（近似命中 target density）
    noise = rng.standard_normal(logits.shape) * noise_std
    score = logits + noise
    flat = score.ravel()
    thr = np.quantile(flat, 1.0 - density)
    mask = score > thr

    # 保证每个用户至少有 1 条交互，且留足测试集（>= n_test_per_user + 1 由调用方保证）
    # 此处确保每用户 >=1 条：缺失则从高分项补全
    row_counts = mask.sum(axis=1)
    for u in np.where(row_counts == 0)[0]:
        top = np.argsort(logits[u])[::-1][: max(1, int(n_items * density) + 1)]
        mask[u, top] = True

    ui, ii = np.where(mask)
    weights = np.ones(ui.shape[0], dtype=np.float64)
    return InteractionData(
        n_users=n_users,
        n_items=n_items,
        user_indices=ui.astype(np.int64),
        item_indices=ii.astype(np.int64),
        weights=weights,
    )


def generate_dataset(
    n_users: int = 200,
    n_items: int = 300,
    n_factors: int = 8,
    density: float = 0.05,
    seed: int = 42,
    noise_std: float = 1.0,
    n_test_per_user: int = 1,
) -> InteractionData:
    """生成并保证切分可行（每用户交互数 >= n_test_per_user+1）。"""
    data = generate_implicit_feedback(
        n_users=n_users,
        n_items=n_items,
        n_factors=n_factors,
        density=density,
        seed=seed,
        noise_std=noise_std,
    )
    _ensure_min_per_user(data, n_test_per_user + 1)
    return data


def _ensure_min_per_user(data: InteractionData, min_count: int) -> None:
    """为交互数不足的用户，按原始潜在结构（此处用现有交互补全）补到 min_count。"""
    from collections import defaultdict

    by_user: dict[int, list[int]] = defaultdict(list)
    for u, i in zip(data.user_indices.tolist(), data.item_indices.tolist()):
        by_user[int(u)].append(int(i))

    extra_u: list[int] = []
    extra_i: list[int] = []
    for u in range(data.n_users):
        cur = by_user.get(u, [])
        need = min_count - len(cur)
        if need <= 0:
            continue
        occupied = set(cur)
        candidates = [i for i in range(data.n_items) if i not in occupied]
        # 确定性补全：取剩余物品的前 need 个
        fill = candidates[:need]
        # 若物品数不足（极端情况），允许重复补全
        while len(fill) < need:
            fill.append(int(len(fill) % data.n_items))
        extra_u.extend([u] * len(fill))
        extra_i.extend(fill)

    if extra_u:
        data.user_indices = np.concatenate(
            [data.user_indices, np.array(extra_u, dtype=np.int64)]
        )
        data.item_indices = np.concatenate(
            [data.item_indices, np.array(extra_i, dtype=np.int64)]
        )
        data.weights = np.concatenate(
            [data.weights, np.ones(len(extra_u), dtype=np.float64)]
        )
