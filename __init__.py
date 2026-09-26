"""RecoForge — 混合协同过滤推荐系统。

作者署名：晨星 · 仓库：CJX0712/recoforge
架构：cli -> pipeline -> {data, reco, eval} -> core（单向无环）。
"""

__version__ = "0.1.0"
__author__ = "晨星"

from .core.config import RecoForgeConfig
from .pipeline.pipeline import RecommendationPipeline

__all__ = ["RecoForgeConfig", "RecommendationPipeline", "__version__"]
