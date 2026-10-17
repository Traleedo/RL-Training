from __future__ import annotations

from abc import abstractmethod
from typing import Any

from core.component import Component

__all__ = ["Logger"]


class Logger(Component):
    """一个日志后端。"""

    category = "logger"

    requires = frozenset()

    @abstractmethod
    def log(self, metrics: dict[str, float], step: int, prefix: str = "train") -> None:
        """记录一批标量。"""
        raise NotImplementedError

    def log_config(self, cfg: Any) -> None:
        """记录配置。后端支持的话（wandb / tensorboard 的 hparams）值得实现。"""

    def log_text(self, key: str, text: str, step: int) -> None:
        """记录一段文本，例如抽样出来的生成结果。默认忽略。"""

    def close(self) -> None:
        """收尾。默认无操作。"""
