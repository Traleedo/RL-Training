from __future__ import annotations

import torch
import torch.nn.functional as torch_f

from core import interfaces as F
from core.base_loss import LossTerm
from core.batch import Batch
from core.registry import register
from core.tensor_ops import masked_sum

__all__ = ["DPOLoss"]


@register("loss", "dpo")
class DPOLoss(LossTerm):
    """直接偏好优化。需要 Reference 模型 —— 这是它唯一昂贵的部分。"""

    requires = frozenset({
        F.CURR_LOGPROBS, F.REF_LOGPROBS, F.RESPONSE_MASK, F.PREFERENCE,
    })

    #: 只有 curr_logprobs 可微；ref_logprobs 是冻结副本产出的常数
    grad_fields = frozenset({F.CURR_LOGPROBS})

    #: 一个对必须整对进同一个 mini-batch —— trainer 据此改用 split_by_group
    needs_intact_groups = True

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0) -> None:
        super().__init__()
        if beta < 0.0:
            raise ValueError(f"beta 必须 >= 0，收到 {beta}")
        if not 0.0 <= label_smoothing < 0.5:
            raise ValueError(
                f"label_smoothing 应在 [0, 0.5) 内，收到 {label_smoothing}。"
                f"0.5 意味着 chosen 与 rejected 完全对称，损失恒为 log 2、梯度恒为 0。"
            )
        self.beta = float(beta)
        self.label_smoothing = float(label_smoothing)

    def name(self) -> str:
        return "dpo"

    # ------------------------------------------------------------------
    def _check_pairing(self, signs: torch.Tensor) -> None:

        zero = torch.zeros(signs.shape[0], dtype=signs.dtype, device=signs.device)
        two = torch.full_like(zero, 2.0)
        balanced = torch.isclose(signs.sum(dim=1), zero).all()
        binary = torch.isclose(signs.abs().sum(dim=1), two).all()
        if not (bool(balanced) and bool(binary)):
            raise ValueError(
                "DPO 的每个偏好对必须恰好包含一个 chosen(+1) 与一个 rejected(-1)，"
                "但收到了不配对的分组。\n"
                "最常见的原因：mini-batch 是按行切开的（Batch.split），"
                "把一个对的两行分到了不同的块里。DPOLoss 已经声明了 "
                "needs_intact_groups=True，trainer 应当改用 split_by_group —— "
                "如果你在手工组装，请检查这一步。\n"
                "另一个原因是数据集的 group_size 不是 2，或 preference 字段写错。"
            )

    def compute(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        batch.require(
            F.CURR_LOGPROBS,
            F.REF_LOGPROBS,
            F.RESPONSE_MASK,
            F.PREFERENCE,
            who="DPOLoss",
        )
        mask = batch[F.RESPONSE_MASK]

        delta = masked_sum(
            batch[F.CURR_LOGPROBS] - batch[F.REF_LOGPROBS], mask, dim=-1
        )

        pair_delta = batch.group_view(delta, 2)            # [G, 2]
        pair_sign = batch.group_view(batch[F.PREFERENCE], 2)  # [G, 2]
        self._check_pairing(pair_sign)

        # (Δ · sign) 在组内求和 = Δ_w − Δ_l，与两行的先后无关
        margin = (pair_delta * pair_sign).sum(dim=1)       # [G]
        logits = self.beta * margin

        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            loss = -(
                (1.0 - eps) * torch_f.logsigmoid(logits)
                + eps * torch_f.logsigmoid(-logits)
            ).mean()
        else:
            loss = -torch_f.logsigmoid(logits).mean()

        with torch.no_grad():
            metrics = {
        
                "implicit_reward_gap": float(logits.mean()),
                "margin": float(margin.mean()),
                "accuracy": float((margin > 0).float().mean()),
                # 逐 token 的对数比均值 —— 排查「参考模型是不是与策略差太远」
                "mean_log_ratio": float(
                    ((batch[F.CURR_LOGPROBS] - batch[F.REF_LOGPROBS])
                     * mask.to(batch[F.CURR_LOGPROBS].dtype)).sum()
                    / mask.sum().clamp(min=1)
                ),
                "beta": self.beta,
            }
        return loss, metrics

    def weight_hint(self) -> float | None:
        return 1.0
