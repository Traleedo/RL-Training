"""本地磁盘 checkpoint 后端。

这也是**基础设施**而不是算法实现 —— 没有它，``TrainerState`` 那个抽象就没有
任何可验证的落地实现，断点续训的正确性也无从测试。
"""

from __future__ import annotations

import os
import re

import torch

from core.base_checkpoint import Checkpointer, TrainerState
from core.registry import register

__all__ = ["LocalCheckpointer"]

_STEP_RE = re.compile(r"step_(\d+)$")


@register("checkpointer", "local")
class LocalCheckpointer(Checkpointer):
    """用 ``torch.save`` / ``torch.load`` 存到本地目录。"""

    def __init__(
        self,
        save_dir: str = "checkpoints",
        every_n_steps: int = 0,
        keep_last: int = 0,
    ) -> None:
        super().__init__(save_dir=save_dir, every_n_steps=every_n_steps)
        #: 只保留最近 N 个存档（0 表示全留）
        self.keep_last = int(keep_last)

    # ------------------------------------------------------------------
    def save(self, state: TrainerState, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # weights_only=False：TrainerState 里有 RNG 状态、优化器超参等非张量对象。
        # 这是可信来源（你自己刚存的文件），不是外部输入。
        torch.save(state, path)
        if self.keep_last:
            self._prune(os.path.dirname(path))

    def load(self, path: str) -> TrainerState:
        if not os.path.exists(path):
            raise FileNotFoundError(f"checkpoint 不存在：{path}")
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(obj, TrainerState):
            raise TypeError(
                f"{path} 里存的不是 TrainerState，而是 {type(obj).__name__}。"
                f"你是不是加载了一个旧格式的存档？"
            )
        return obj

    def latest(self, save_dir: str) -> str | None:
        if not os.path.isdir(save_dir):
            return None
        candidates: list[tuple[int, str]] = []
        for name in os.listdir(save_dir):
            match = _STEP_RE.search(name)
            if match:
                candidates.append((int(match.group(1)), os.path.join(save_dir, name)))
        if not candidates:
            return None
        return max(candidates)[1]

    # ------------------------------------------------------------------
    def _prune(self, directory: str) -> None:
        candidates: list[tuple[int, str]] = []
        for name in os.listdir(directory):
            match = _STEP_RE.search(name)
            if match:
                candidates.append((int(match.group(1)), os.path.join(directory, name)))
        for _, path in sorted(candidates)[: -self.keep_last]:
            try:
                os.remove(path)
            except OSError:
                pass
