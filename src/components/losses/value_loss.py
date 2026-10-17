from __future__ import annotations

import torch

from core import interfaces as F
from core.base_loss import LossTerm
from core.batch import Batch
from core.registry import register
from core.tensor_ops import masked_mean

__all__ = ["ValueLoss"]


@register("loss", "value_loss")
class ValueLoss(LossTerm):
    """带可选裁剪的均方价值误差。"""

    requires = frozenset(
        {F.VALUES, F.ROLLOUT_VALUES, F.RETURNS, F.RESPONSE_MASK}
    )

    # 注意：是 VALUES 而不是 CURR_LOGPROBS。
    # 本项完全不看策略的 logprob，照抄基类默认值会让
    # tests/test_grad_flow.py 报「声明了 curr_logprobs 可微但没有梯度」。
    grad_fields = frozenset({F.VALUES})

    def __init__(self, clip_eps: float | None = 0.2) -> None:
        super().__init__()
        self.clip_eps = None if clip_eps is None else float(clip_eps)

    def name(self) -> str:
        return "value_loss"

    def weight_hint(self) -> float:
        return 0.5

    # ------------------------------------------------------------------
    def compute(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        batch.require(
            F.VALUES, F.ROLLOUT_VALUES, F.RETURNS, F.RESPONSE_MASK, who="ValueLoss"
        )
        mask = batch[F.RESPONSE_MASK]
        values = batch[F.VALUES]
        returns = batch[F.RETURNS]
        old_values = batch[F.ROLLOUT_VALUES]

        squared = (values - returns) ** 2
        if self.clip_eps is None:
            per_token = squared
            clip_frac = 0.0
        else:
            # 与 ppo_clip 同一个形状：裁剪后的值也要和**同一个** returns 比，
            # 然后两项取逐元素最大值（即取更悲观的那个）。
            v_clipped = old_values + torch.clamp(
                values - old_values, -self.clip_eps, self.clip_eps
            )
            per_token = torch.maximum(squared, (v_clipped - returns) ** 2)
            clip_frac = float(
                masked_mean(
                    (torch.abs(values - old_values) > self.clip_eps).to(mask.dtype), mask
                ).detach()
            )

        with torch.no_grad():
            metrics = {
                "mean_value": float(masked_mean(values, mask)),
                "clip_frac": clip_frac,
                # 解释方差：critic 比「直接用 returns 的均值」好多少。
                # <= 0 意味着 critic 什么都没学到，PPO 的优势就退化成纯回报。
                "explained_variance": self._explained_variance(values, returns, mask),
            }
        return masked_mean(per_token, mask), metrics

    # ------------------------------------------------------------------
    @staticmethod
    def _explained_variance(
        values: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor
    ) -> float:
        target = returns[mask > 0]
        prediction = values[mask > 0]
        if target.numel() < 2:
            return 0.0
        residual = ((target - prediction) ** 2).sum()
        total = ((target - target.mean()) ** 2).sum()
        if float(total) == 0.0:
            return 0.0
        return float(1.0 - residual / total)
