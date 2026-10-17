
from __future__ import annotations

import torch

from core import interfaces as F
from core.base_loss import LossTerm
from core.batch import Batch
from core.registry import register
from core.tensor_ops import masked_mean, masked_sum

__all__ = ["CrossEntropyLoss"]


@register("loss", "cross_entropy")
class CrossEntropyLoss(LossTerm):
    requires = frozenset({F.CURR_LOGPROBS, F.RESPONSE_MASK})

    #: 只有 curr_logprobs 是可微的 —— 这条声明同时是「SFT 不建 Reference」的依据
    grad_fields = frozenset({F.CURR_LOGPROBS})

    #: 成对数据才需要整组进同一个 mini-batch；SFT 每行独立，随便切。
    needs_intact_groups = False

    def __init__(self, reduction: str = "token_mean") -> None:
        super().__init__()
        if reduction not in ("token_mean", "seq_mean"):
            raise ValueError(
                f"未知的 reduction {reduction!r}；支持 token_mean / seq_mean"
            )
        self.reduction = reduction

    def name(self) -> str:
        return "cross_entropy"

    def compute(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        batch.require(
            F.CURR_LOGPROBS, F.RESPONSE_MASK, who="CrossEntropyLoss"
        )
        mask = batch[F.RESPONSE_MASK]
        logprobs = batch[F.CURR_LOGPROBS]

        if self.reduction == "token_mean":
            loss = -masked_mean(logprobs, mask)
        else:
            per_sequence = masked_sum(logprobs, mask, dim=-1) / mask.sum(dim=-1).clamp(min=1)
            loss = -per_sequence.mean()

        with torch.no_grad():
            # 困惑度是唯一一个跨模型、跨数据都能看懂的量 ——
            # token 平均的负对数似然取 exp 就是它。
            metrics = {
                "perplexity": float(torch.exp(-masked_mean(logprobs, mask).detach())),
                "response_tokens": float(mask.sum()),
            }
        return loss, metrics
