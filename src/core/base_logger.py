"""日志后端抽象。

替代了原伪代码里的 ``self.dashboard``（那个对象从未被赋值过）。

支持配置成列表，于是「控制台 + tensorboard + wandb 同时开」只是 YAML 里多一行。
每个后端只实现 ``log``，其余都是可选钩子。
"""

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
