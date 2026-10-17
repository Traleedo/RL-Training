from __future__ import annotations

import torch

from core import interfaces as F
from core.base_loss import LossTerm
from core.batch import Batch
from core.registry import register
from core.tensor_ops import masked_mean

__all__ = ["GSPOLoss"]


@register("loss", "gspo")
class GSPOLoss(LossTerm):
    """序列级 ratio 的裁剪策略梯度项。"""

    requires = frozenset(
        {F.CURR_LOGPROBS, F.ROLLOUT_LOGPROBS, F.ADVANTAGES, F.RESPONSE_MASK}
    )

    #: ``rollout_logprobs`` 只被**读**，必须作为常量参与（它已被 freeze 保护）。
    grad_fields = frozenset({F.CURR_LOGPROBS})

    def __init__(self, clip_eps: float = 3e-4) -> None:
        super().__init__()
        if clip_eps < 0:
            raise ValueError(f"clip_eps 必须非负，收到 {clip_eps}")
        self.clip_eps = float(clip_eps)

    def name(self) -> str:
        return "gspo"

    def weight_hint(self) -> float:
        return 1.0

    def describe(self) -> str:
        return f"{super().describe()}  clip_eps={self.clip_eps}（按序列裁剪）"

    # ------------------------------------------------------------------
    def sequence_advantage(
        self, advantages: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        seq_advantage = masked_mean(advantages, mask, dim=-1)
        deviation = ((advantages - seq_advantage.unsqueeze(-1)).abs() * mask).max()
        scale = seq_advantage.abs().max().clamp(min=1.0)
        if float(deviation) > 1e-4 * float(scale):
            raise ValueError(
                f"GSPO 需要**序列级**优势（同一个序列的 response token 上取值相同），"
                f"但收到的优势在序列内部不一致（最大偏差 {float(deviation):.3e}）。"
                f"最常见的原因是把 advantage 配成了 gae —— 那是逐 token 的。"
                f"GSPO 是无 critic 算法，请用 advantage: broadcast 配一个会归一化"
                f"奖励的 processor（例如 group_normalize）。"
            )
        return seq_advantage

    def compute(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        batch.require(
            F.CURR_LOGPROBS,
            F.ROLLOUT_LOGPROBS,
            F.ADVANTAGES,
            F.RESPONSE_MASK,
            who="GSPOLoss",
        )
        mask = batch[F.RESPONSE_MASK]
        log_ratio = batch[F.CURR_LOGPROBS] - batch[F.ROLLOUT_LOGPROBS]

        # 序列级对数比 = 逐 token 对数比在 response_mask 上的均值。
        # 先平均再取 exp，不要写成连乘 —— 见模块 docstring。
        seq_log_ratio = masked_mean(log_ratio, mask, dim=-1)      # [B]
        seq_ratio = torch.exp(seq_log_ratio)

        seq_advantage = self.sequence_advantage(batch[F.ADVANTAGES], mask)

        # 每个序列只裁一次。
        clipped = torch.clamp(seq_ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        objective = torch.min(seq_ratio * seq_advantage, clipped * seq_advantage)

        active = (mask.sum(dim=-1) > 0).to(objective.dtype)
        loss = -(objective * active).sum() / active.sum().clamp(min=1.0)

        with torch.no_grad():
            token_ratio = torch.exp(log_ratio)
            is_clipped = (torch.abs(seq_ratio - 1.0) > self.clip_eps).to(
                seq_ratio.dtype
            )
            metrics = {
                "mean_seq_ratio": float((seq_ratio * active).sum() / active.sum().clamp(min=1.0)),
                # 按**序列**计的裁剪比例。与 policy_gradient 的 clip_frac
                # （按 token 计）不是一回事，所以键名也不同。
                "seq_clip_frac": float((is_clipped * active).sum() / active.sum().clamp(min=1.0)),
                "min_seq_ratio": float(seq_ratio.min()),
                "max_seq_ratio": float(seq_ratio.max()),
                # 逐 token ratio 的最大值。把它和 max_seq_ratio 并排看，
                # 就是「逐 token 裁剪管不住长序列」这件事的直接证据：
                # 它是 1.01 而序列比值是 100 的时候，PPO 系一个 token 都不会裁。
                "max_token_ratio": float((token_ratio * mask).max()),
                "mean_log_ratio": float(seq_log_ratio.mean()),
                "mean_seq_len": float(mask.sum(dim=-1).to(torch.float32).mean()),
                "frac_neg_adv": float(
                    (seq_advantage < 0).to(objective.dtype).mean()
                ),
            }
        return loss, metrics
