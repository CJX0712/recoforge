"""Top-K 排序指标（全手写实现，零外部依赖）。

约定：score 越大越相关；评测时排除训练集已交互物品（防数据泄漏）。
指标：Precision@K / Recall@K / NDCG@K / MAP@K（均对用户取平均）。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import EvalError


def _rank_excluding_train(scores: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """返回每行排除训练物品后的降序排名索引。

    train_mask 为 True 的位置被置为 -inf，不参与排名。
    """
    blocked = scores.copy()
    blocked[train_mask] = -np.inf
    # 降序排列：用 argsort 两次得到排名，再取前 k 在调用方处理
    order = np.argsort(blocked, axis=1)[:, ::-1]
    return order


def _ideal_dcg(n_relevant: int, k: int) -> float:
    m = min(n_relevant, k)
    if m <= 0:
        return 0.0
    ranks = np.arange(1, m + 1, dtype=np.float64)
    return float(np.sum(1.0 / np.log2(ranks + 1.0)))


def top_k_metrics(
    scores: np.ndarray,
    train_mask: np.ndarray,
    test_items: list[np.ndarray],
    k: int,
) -> dict:
    """计算所有用户的 Top-K 平均指标。

    参数
    ----
    scores     : (n_users, n_items) 预测评分
    train_mask : (n_users, n_items) 布尔，True=训练集物品需排除
    test_items : 每用户测试物品索引数组（list[np.ndarray]）
    k          : 截断位置
    """
    if scores.shape != train_mask.shape:
        raise EvalError(
            f"scores 与 train_mask 形状不一致: {scores.shape} vs {train_mask.shape}"
        )
    if len(test_items) != scores.shape[0]:
        raise EvalError("test_items 长度必须等于用户数")
    if k <= 0:
        raise EvalError(f"k 必须为正: {k}")

    n_users = scores.shape[0]
    order = _rank_excluding_train(scores, train_mask)  # (n_users, n_items) 降序

    precisions, recalls, ndcgs, maps = [], [], [], []
    for u in range(n_users):
        test = test_items[u]
        if test is None or len(test) == 0:
            continue
        rel_set = {int(i) for i in test}
        top = order[u, :k]
        hits = np.array([1 if int(i) in rel_set else 0 for i in top], dtype=np.float64)
        n_rel = len(rel_set)

        precision = float(hits.sum() / k)
        recall = float(hits.sum() / n_rel)
        # NDCG
        positions = np.nonzero(hits)[0]  # 0-based rank index
        dcg = 0.0
        if positions.size > 0:
            dcg = float(np.sum(1.0 / np.log2(positions.astype(np.float64) + 2.0)))
        idcg = _ideal_dcg(n_rel, k)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        # MAP@K（截断平均精度）
        if positions.size > 0:
            cum = np.cumsum(hits)
            prec_at_hit = cum[positions] / (positions.astype(np.float64) + 1.0)
            ap = float(prec_at_hit.sum() / min(n_rel, k))
        else:
            ap = 0.0

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
        maps.append(ap)

    if not precisions:
        raise EvalError("无有效测试用户，无法计算指标")

    return {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "ndcg_at_k": float(np.mean(ndcgs)),
        "map_at_k": float(np.mean(maps)),
    }
