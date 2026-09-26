"""RecoForge 错误码定义（E100~E500 区间）。

错误码约定：
  E000  通用错误（RecoForgeError 基类）
  E100  数据层错误（生成 / 载入 / 切分）
  E200  模型层错误（训练 / 预测 / 后端不可用）
  E300  评测层错误（指标计算）
  E400  配置层错误（环境变量 / 参数校验）
  E500  流水线层错误（编排 / 确定性校验）
"""

from __future__ import annotations


class RecoForgeError(Exception):
    """所有 RecoForge 错误的基类。"""

    code = "E000"

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code


class DataError(RecoForgeError):
    """数据生成 / 载入 / 切分相关错误。"""

    code = "E100"


class ModelError(RecoForgeError):
    """模型训练 / 预测 / 后端探测相关错误。"""

    code = "E200"


class EvalError(RecoForgeError):
    """指标计算相关错误。"""

    code = "E300"


class ConfigError(RecoForgeError):
    """配置解析 / 参数校验相关错误。"""

    code = "E400"


class PipelineError(RecoForgeError):
    """流水线编排 / 确定性校验相关错误。"""

    code = "E500"
