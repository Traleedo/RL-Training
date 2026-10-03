from __future__ import annotations

import torch

from core import interfaces as F
from core.base_advantage import AdvantageEstimator
from core.batch import Batch
from core.registry import register
from core.tensor_ops import masked_mean

__all__ = ["GAEAdvantage"]


@register("advantage", "gae")
class GAEAdvantage(AdvantageEstimator):
    """``δ_t = r_t + γ·V(t+1) - V(t)``，``A_t = δ_t + γλ·A_{t+1}``。
    即时奖励的约定：序列级奖励只落在**最后一个** response token 上，其余为 0。
    """

    # ROLLOUT_VALUES 是 PPO 的身份标识 —— 它一出现，Critic 就会被构建。
    requires = frozenset({F.REWARDS, F.ROLLOUT_VALUES, F.RESPONSE_MASK})
    provides = frozenset({F.ADVANTAGES, F.RETURNS})

    def __init__(self, gamma: float = 0.99, lam: float = 0.95) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.lam = float(lam)
        self._stats: dict[str, float] = {}

    def compute(self, batch: Batch) -> Batch:
        batch.require(
            F.REWARDS, F.ROLLOUT_VALUES, F.RESPONSE_MASK, who="GAEAdvantage"
        )
        values = batch[F.ROLLOUT_VALUES]                      # [B, L]
        mask = batch[F.RESPONSE_MASK].to(values.dtype)        # [B, L]
        rewards = batch[F.REWARDS]                            # [B]

        deltas = self._deltas(rewards, values, mask)

        advantages = torch.zeros_like(deltas)
        running = torch.zeros_like(rewards)
        for t in range(deltas.shape[-1] - 1, -1, -1):
            running = deltas[:, t] + self.gamma * self.lam * running * mask[:, t]
            advantages[:, t] = running

        batch[F.ADVANTAGES] = advantages
        batch[F.RETURNS] = (advantages + values) * mask

        with torch.no_grad():
            self._stats = {
                "advantage/mean": float(masked_mean(advantages, mask)),
                "advantage/std": float(advantages[mask > 0].std()) if mask.sum() > 1 else 0.0,
            }
        return batch

    def _deltas(
        self, rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """TD 残差 ``δ_t = r_t + γ·V(t+1) - V(t)``，padding 位置为 0。"""
        lengths = mask.sum(-1, keepdim=True)
        is_last = ((mask.cumsum(-1) == lengths) & (mask > 0)).to(values.dtype)
        r_t = rewards.unsqueeze(-1) * is_last
        next_values = torch.zeros_like(values)
        next_values[:, :-1] = values[:, 1:] * mask[:, 1:]

        return (r_t + self.gamma * next_values - values) * mask

    def metrics(self) -> dict[str, float]:
        return dict(self._stats)

    def describe(self) -> str:
        return f"{super().describe()}  gamma={self.gamma} lam={self.lam}"
