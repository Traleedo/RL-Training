"""组内归一化 —— GRPO 一类的核心旋钮。

.. math::

    \\tilde{r}_i = \\frac{r_i - \\mu_g}{\\sigma_g + \\epsilon}

同一条 prompt 采样 ``num_generations`` 条，组内比较得到一个**相对**优势。
这是 GRPO 相对 PPO 的关键简化：不需要价值网络来定义「基线」，
组内均值就是基线。

``divide_std``：GRPO 与 Dr.GRPO 的分界线
---------------------------------------
- ``divide_std: true``（默认）—— 标准 GRPO。
- ``divide_std: false`` —— 只减均值、不除标准差，即 Dr.GRPO 的做法。

为什么想去掉 σ：**σ 里混进了难度信息**。一个「几乎全对」的组 σ 很小，
除掉它会把那点微小差异放大成一个满量级的优势 —— 于是模型被推着去拟合
噪声。去掉 σ 之后，难组（奖励分布散）自然得到更大的优势，
这正是我们想要的优先级。

为什么归一化归 processor 而不归 advantage
----------------------------------------
见 ``core/base_processor.py`` 的模块 docstring：做成 processor 才能
可组合、可单独关闭（DAPO 要单独调它）。代价是复现标准 GRPO 要写两个配置项。

**``normalizes_rewards`` 在 ``divide_std=False`` 时仍然是 True。** 因为减均值
本身就已经改变了奖励的语义（把「绝对好坏」变成「组内相对好坏」），
如果某个 advantage 又自己内部做一次归一化，两处叠加依然是错的。
这个标志表达的是「我这一层动过奖励」，不是「我除了标准差」。
"""

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
                f"只有一条采样时组内标准差恒为 0，归一化会把奖励整组压成 0，"
                f"梯度信号消失而训练照跑不误。单采样场景请改用 global_normalize。"
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

        # 「有多少组被 σ 近似零除」是一个值得盯的量：它偏大说明这批 prompt
        # 对该策略几乎全对或全错，而此时 divide_std=true 会把噪声放大成信号。
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
