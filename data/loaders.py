"""交互数据载入 / 落盘（npz，零额外依赖）。"""

from __future__ import annotations

import numpy as np

from ..core.errors import DataError
from ..core.types import InteractionData


def save_npz(data: InteractionData, path: str) -> None:
    np.savez(
        path,
        n_users=np.array(data.n_users),
        n_items=np.array(data.n_items),
        user_indices=data.user_indices.astype(np.int64),
        item_indices=data.item_indices.astype(np.int64),
        weights=data.weights.astype(np.float64),
    )


def load_npz(path: str) -> InteractionData:
    if not path.endswith(".npz"):
        raise DataError(f"仅支持 .npz 格式: {path}")
    obj = np.load(path, allow_pickle=False)
    return InteractionData(
        n_users=int(obj["n_users"]),
        n_items=int(obj["n_items"]),
        user_indices=obj["user_indices"].astype(np.int64),
        item_indices=obj["item_indices"].astype(np.int64),
        weights=obj["weights"].astype(np.float64),
    )
