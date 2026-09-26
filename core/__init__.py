"""RecoForge core 包：类型 / 错误 / 配置 / 接口 / 种子。"""

from .config import RecoForgeConfig
from .errors import (
    ConfigError,
    DataError,
    EvalError,
    ModelError,
    PipelineError,
    RecoForgeError,
)
from .seed import check_reproducible, get_seed, set_all
from .types import (
    BenchmarkReport,
    InteractionData,
    MetricResult,
    SplitResult,
)

__all__ = [
    "BenchmarkReport",
    "ConfigError",
    "DataError",
    "EvalError",
    "InteractionData",
    "MetricResult",
    "ModelError",
    "PipelineError",
    "RecoForgeConfig",
    "RecoForgeError",
    "SplitResult",
    "check_reproducible",
    "get_seed",
    "set_all",
]
