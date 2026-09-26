"""RecoForge data 包：合成数据生成 + 载入。"""

from .loaders import load_npz, save_npz
from .synthetic import generate_dataset, generate_implicit_feedback

__all__ = [
    "generate_dataset",
    "generate_implicit_feedback",
    "load_npz",
    "save_npz",
]
