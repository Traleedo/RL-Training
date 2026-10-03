from __future__ import annotations

import torch

from core import interfaces as F
from core.base_processor import RewardProcessor
from core.batch import Batch
from core.registry import register

__all__ = ["DynamicFilterProcessor"]


@register("processor", "dynamic_filter")
class DynamicFilterProcessor(RewardProcessor):
    """丢掉组内奖励极差 <= ``eps`` 的组（DAPO 动态采样）。"""

    #: 会让 batch 的**行数变少**。plan_assembly 靠这个标志检查它在链上的位置。
    changes_batch_size = True

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self._stats: dict[str, float] = {}

    def process(self, batch: Batch) -> Batch:
        rewards = batch[F.REWARDS]
        group_size = int(batch.meta.get(F.NUM_GENERATIONS, 1))

        if group_size < 2:
            raise ValueError(
                f"动态采样需要 num_generations >= 2，收到 {group_size}。"
            )
        if len(batch) % group_size:
            raise ValueError(
                f"batch 维 {len(batch)} 不能被 num_generations {group_size} 整除，"
            )

        groups = rewards.reshape(-1, group_size)
        spread = groups.max(dim=1).values - groups.min(dim=1).values
        keep = spread > self.eps

        kept = batch.filter_groups(keep, group_size)

        kept[F.GROUP_IDS] = kept.group_ids(group_size)

        n_groups = int(keep.numel())
        n_kept = int(keep.sum())
        with torch.no_grad():
            self._stats = {
                "dynamic_filter/groups_in": float(n_groups),
                "dynamic_filter/groups_kept": float(n_kept),
                "dynamic_filter/drop_frac": float((n_groups - n_kept) / max(n_groups, 1)),
            }
        return kept

    def metrics(self) -> dict[str, float]:
        return dict(self._stats)

    def describe(self) -> str:
        return f"{super().describe()}  eps={self.eps}"
