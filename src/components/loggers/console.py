"""控制台日志后端。

这是**基础设施**，不是算法实现 —— 没有它就没有任何日志接缝可验证。
"""

from __future__ import annotations

import logging

from core.base_logger import Logger
from core.registry import register

__all__ = ["ConsoleLogger"]

logger = logging.getLogger(__name__)


@register("logger", "console")
class ConsoleLogger(Logger):
    """按组对齐打印指标。"""

    def __init__(
        self,
        every_n_steps: int = 1,
        groups: list[str] | None = None,
        max_width: int = 100,
    ) -> None:
        super().__init__()
        self.every_n_steps = int(every_n_steps)
        #: 只打印带有这些前缀的指标。为空则全打印。
        self.groups = list(groups) if groups else None
        self.max_width = int(max_width)

    def log(self, metrics: dict[str, float], step: int, prefix: str = "train") -> None:
        if self.every_n_steps > 1 and step % self.every_n_steps != 0:
            return
        selected = {
            k: v
            for k, v in metrics.items()
            if self.groups is None or any(k.startswith(g) for g in self.groups)
        }
        if not selected:
            return
        logger.info("── step %d ──", step)
        width = min(
            self.max_width, max((len(k) for k in selected), default=10)
        )
        for key in sorted(selected):
            logger.info("  %-*s  %+ .6g", width, key, selected[key])
