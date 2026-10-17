from __future__ import annotations

import torch

from core import interfaces as F
from core.base_processor import RewardProcessor
from core.batch import Batch
from core.registry import register

__all__ = ["GroupNormalizeProcessor"]


@register("processor", "group_normalize")
class GroupNormalizeProcessor(RewardProcessor):
    """组内 ``(r - μ) / σ``（``divide_std: false`` 时退化为 ``r - μ``）。"""
    normalizes_rewards = True

    def __init__(self, divide_std: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        self.divide_std = bool(divide_std)
        self.eps = float(eps)
        self._stats: dict[str, float] = {}

    def process(self, batch: Batch) -> Batch:
        rewards = batch[F.REWARDS]
        group_size = int(batch.meta.get(F.NUM_GENERATIONS, 1))

        if group_size < 2:
            raise ValueError(
                f"组内归一化需要 num_generations >= 2，收到 {group_size}。"
            )
        if len(batch) % group_size:
            raise ValueError(
                f"batch 维 {len(batch)} 不能被 num_generations {group_size} 整除，"
                f"分组会残缺。"
            )

        groups = rewards.reshape(-1, group_size)
        mean = groups.mean(dim=1, keepdim=True)
        centered = groups - mean

        if self.divide_std:
            std = groups.std(dim=1, keepdim=True, unbiased=False)
            centered = centered / (std + self.eps)
        else:
            # Dr.GRPO 路径：保留 σ 所承载的难度信息，不做缩放。
            std = groups.std(dim=1, keepdim=True, unbiased=False)

        batch[F.REWARDS] = centered.reshape(-1)

        with torch.no_grad():
            self._stats = {
                "group_normalize/mean_std": float(std.mean()),
                "group_normalize/degenerate_frac": float(
                    (std <= self.eps * 10.0).to(std.dtype).mean()
                ),
            }
        return batch

    def metrics(self) -> dict[str, float]:
        return dict(self._stats)

    def describe(self) -> str:
        return f"{super().describe()}  divide_std={self.divide_std} eps={self.eps}"
