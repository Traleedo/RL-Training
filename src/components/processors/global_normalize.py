from __future__ import annotations

import torch

from core import interfaces as F
from core.base_processor import RewardProcessor
from core.batch import Batch
from core.registry import register

__all__ = ["GlobalNormalizeProcessor"]


@register("processor", "global_normalize")
class GlobalNormalizeProcessor(RewardProcessor):
    """整个 batch 上的 ``(r - μ) / σ``。单采样（``num_generations=1``）用这个。"""

    normalizes_rewards = True

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self._stats: dict[str, float] = {}

    def process(self, batch: Batch) -> Batch:
        rewards = batch[F.REWARDS]
        n = rewards.numel()
        if n == 0:
            return batch
        if n == 1:
            raise ValueError(
                f"全局归一化需要至少 2 个样本才能定义标准差，收到 {n} 个。"
                f"单样本归一化会把奖励压成 0，梯度恒为 0 而训练照跑不误。"
                f"请调大 batch（rollout.num_generations × prompt 条数），"
                f"或在单样本场景下改用不做归一化的 advantage。"
            )

        std = rewards.std(unbiased=False)
        batch[F.REWARDS] = (rewards - rewards.mean()) / (std + self.eps)

        with torch.no_grad():
            self._stats = {
                "global_normalize/raw_std": float(std),
                "global_normalize/raw_mean": float(rewards.mean()),
                "global_normalize/degenerate_frac": float(
                    (std <= self.eps * 10.0).to(std.dtype)
                ),
            }
        return batch

    def metrics(self) -> dict[str, float]:
        return dict(self._stats)

    def describe(self) -> str:
        return f"{super().describe()}  eps={self.eps}"
