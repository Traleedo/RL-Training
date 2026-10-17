from __future__ import annotations

import torch

from core import interfaces as F
from core.base_advantage import AdvantageEstimator
from core.batch import Batch
from core.registry import register
from core.tensor_ops import broadcast_sequence_to_tokens, masked_mean

__all__ = ["BroadcastAdvantage"]


@register("advantage", "broadcast")
class BroadcastAdvantage(AdvantageEstimator):
    """``A_{i,t} = R_i`` —— 一个回答内的所有 token 共享同一个优势。"""

    requires = frozenset({F.REWARDS, F.RESPONSE_MASK})
    provides = frozenset({F.ADVANTAGES})
    transient_requires = frozenset()

    def __init__(self) -> None:
        super().__init__()
        self._stats: dict[str, float] = {}

    def compute(self, batch: Batch) -> Batch:
        batch.require(F.REWARDS, F.RESPONSE_MASK, who="BroadcastAdvantage")
        mask = batch[F.RESPONSE_MASK]
        advantages = broadcast_sequence_to_tokens(batch[F.REWARDS], mask)
        batch[F.ADVANTAGES] = advantages

        with torch.no_grad():
            self._stats = {
                "advantage/mean": float(masked_mean(advantages, mask)),
                "advantage/std": (
                    float(advantages[mask > 0].std()) if mask.sum() > 1 else 0.0
                ),
            }
        return batch

    def metrics(self) -> dict[str, float]:
        return dict(self._stats)
