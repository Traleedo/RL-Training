"""SFT 的损失项 —— 就是负对数似然，但它值得单独说明三件事。

.. math::

    L = -\\frac{1}{\\sum_t m_t} \\sum_t m_t \\cdot \\log \\pi_\\theta(y_t \\mid x, y_{<t})

三件容易搞错的事
----------------

**一、它不需要 Reference。**
``requires`` 里没有 ``ref_logprobs``，于是 ``plan_assembly`` 推导出
``need_reference is False`` —— base.yaml 里 reference 那段配置还挂着，
模型就是不会被加载。这是「条件依赖」在离线家族里第一次成立，
也是 SFT 与 DPO 唯一的装配层差别。

**二、``curr_logprobs`` 已经是对齐好的 log 概率，不用再 gather 一次。**
``actor.logprobs(batch)`` 在训练相位的前向里已经做了 shift 与 gather
（见 ``core.tensor_ops.gather_token_logprobs``，全仓库唯一的 shift 实现）。
在这里再 gather 一次会得到一个形状对得上、数值全错的张量 ——
而 loss 照样下降，只是收敛到一个更低的下界。

**三、分母是 token 数，不是序列数。**
长回答的梯度天然更大。想让每条样本等权就得按长度归一化再平均，
那是另一个选择（也是另一种损失），不在这里偷偷做。
``reduction`` 参数把这个选择显式化，默认是标准的逐 token 平均。
"""

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
    """监督微调：让模型对给定回答的 log 概率尽量高。

    参数
    ----
    ``reduction``
        ``token_mean``（默认）—— 逐 token 平均，标准 SFT。
        ``seq_mean`` —— 每条序列先各自平均、再对序列平均。长回答不再占更大权重。
        两者的差别在长度方差大的数据上才明显，但那时它很明显。
    """

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
