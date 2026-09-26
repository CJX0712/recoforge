"""配置层：环境变量覆盖 + schema 校验。

环境变量前缀 `RECOFORGE_`，例如 `RECOFORGE_SEED=42`。
所有字段经 schema 校验，非法值抛出 ConfigError(E400)。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

from ..core.errors import ConfigError


@dataclass
class RecoForgeConfig:
    seed: int = 42
    n_users: int = 200
    n_items: int = 300
    n_factors: int = 32
    density: float = 0.05
    k: int = 10
    alpha: float = 8.0
    reg: float = 0.1
    iterations: int = 15
    n_test_per_user: int = 1

    _ENV_PREFIX = "RECOFORGE_"

    @classmethod
    def from_env(cls, **overrides: object) -> RecoForgeConfig:
        values: dict[str, object] = {f.name: f.default for f in fields(cls)}
        for f in fields(cls):
            env_val = os.environ.get(f"{cls._ENV_PREFIX}{f.name.upper()}")
            if env_val is not None:
                values[f.name] = _coerce(f.name, env_val, f.type)
        values.update(overrides)
        cfg = cls(**{k: values[k] for k in values})
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ConfigError(f"seed 必须为非负整数: {self.seed}")
        if self.n_users <= 0 or self.n_items <= 0:
            raise ConfigError(
                f"n_users/n_items 必须为正: {self.n_users}/{self.n_items}"
            )
        if self.n_factors <= 0:
            raise ConfigError(f"n_factors 必须为正: {self.n_factors}")
        if not (0.0 < self.density < 1.0):
            raise ConfigError(f"density 必须在 (0,1) 开区间: {self.density}")
        if self.k <= 0:
            raise ConfigError(f"k 必须为正: {self.k}")
        if self.alpha < 0:
            raise ConfigError(f"alpha 必须非负: {self.alpha}")
        if self.reg < 0:
            raise ConfigError(f"reg 必须非负: {self.reg}")
        if self.iterations <= 0:
            raise ConfigError(f"iterations 必须为正: {self.iterations}")
        if self.n_test_per_user < 1:
            raise ConfigError(f"n_test_per_user 必须 >=1: {self.n_test_per_user}")
        # 数据集规模需支撑切分：每个用户至少有 n_test_per_user+1 条交互
        min_interactions = self.n_users * (self.n_test_per_user + 1)
        expected = int(self.n_users * self.n_items * self.density)
        if expected < min_interactions:
            raise ConfigError(
                f"density 过低，预期交互数 {expected} < 切分所需最少 {min_interactions}，"
                f"请提高 density 或 n_users"
            )


def _coerce(name: str, raw: str, typ: object) -> object:
    # 兼容 from __future__ import annotations 下 f.type 为字符串的情形
    if typ is int or typ == "int":
        target = int
    elif typ is float or typ == "float":
        target = float
    else:
        return raw
    try:
        return target(raw)
    except ValueError as exc:
        raise ConfigError(
            f"环境变量 RECOFORGE_{name.upper()} 无法解析为 {typ}: {raw!r}"
        ) from exc
