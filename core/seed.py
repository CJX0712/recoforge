"""全局确定性控制。

唯一 seed 入口：numpy / random / 库级（torch 可选）一次设齐。
demo 两次运行的结果必须逐位一致，依赖此模块。
"""

from __future__ import annotations

import random

import numpy as np

_SEED: int | None = None


def set_all(seed: int) -> int:
    """设置全部随机源的确定性种子，返回生效的 seed。"""
    global _SEED
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed 必须为非负整数，收到: {seed!r}")
    _SEED = int(seed)
    random.seed(_SEED)
    np.random.seed(_SEED)
    try:  # torch 可选依赖
        import torch

        torch.manual_seed(_SEED)
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.manual_seed_all(_SEED)
    except ImportError:
        pass
    return _SEED


def get_seed() -> int | None:
    return _SEED


def check_reproducible(a: float, b: float, tol: float = 0.0) -> bool:
    """确定性校验：tol=0 时要求逐位相等。"""
    return abs(a - b) <= tol
