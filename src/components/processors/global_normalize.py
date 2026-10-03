"""全局归一化 —— 单采样（PPO 一类）场景的奖励标准化。

.. math::

    \\tilde{r}_i = \\frac{r_i - \\mu}{\\sigma + \\epsilon}

μ 与 σ 取**整个 batch**，不分 prompt，也不看 ``num_generations``。

它和 ``group_normalize`` 是互补的两件事，而不是强弱两个版本
----------------------------------------------------------
``group_normalize`` 要求 ``num_generations >= 2``：它的基线是**同一条 prompt
的其它采样**，所以采样数为 1 时组内标准差恒为 0，归一化会把奖励整组压成 0
—— 梯度信号消失而训练照跑不误。它的报错信息里就写着「单采样场景请改用
global_normalize」，这个文件就是那句话指向的东西。

PPO 是单采样的（优势由 GAE 从价值函数算出来，不靠组内比较），所以它需要的是
这一个。两者的 ``normalizes_rewards`` 都是 True，而那面标志的含义是
「我这一层动过奖励」—— 只要动了，advantage 那边就不能再自己归一化一次。

与「优势级归一化」的区别
------------------------
教科书 PPO 常写 ``(adv - μ) / σ``，作用在**优势**上；这里是奖励级的。
之所以只能做到奖励级，是相位的硬约束：processor 链跑在
``advantage.compute()`` **之前**（见 ``engine/rl_trainer.py`` 的相位 B 与 D），
那一刻 advantages 还不存在。想要优势级归一化，得在 AdvantageEstimator
里做，不是在 processor 里。

.. warning::

   奖励级与优势级不等价。对 GAE 而言，缩放奖励会同时改变 δ 与自举项，
   最终的优势不是简单地被线性缩放的。这在单步（``gamma=0``）或
   ``lam=1`` 且价值函数准确的极限下才趋于一致。
"""

from __future__ import annotations

import torch

from core import interfaces as F
from core.base_processor import RewardProcessor
from core.batch import Batch
from core.registry import register

__all__ = ["GlobalNormalizeProcessor"]


@register("processor", "global_normalize")
class GlobalNormalizeProcessor(RewardProcessor):
    """整个 batch 上的 ``(r - μ) / σ``。单采样（``num_generations=1``）用这个。"""

    normalizes_rewards = True

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self._stats: dict[str, float] = {}

    def process(self, batch: Batch) -> Batch:
        rewards = batch[F.REWARDS]
        n = rewards.numel()

        # 空批次：放行，不归一化。
        #
        # 这**不是**在容忍错误，而是在把「本步没测到东西」交给唯一该管它的
        # 那一层 —— trainer 的空批次守卫（engine/rl_trainer.py 的相位 B'）。
        # 它会把这一步整个跳过：不推进 global_step、不存盘、不调控制器钩子。
        # 如果这里先抛错，得到的是一个更难看的失败，而信息量没有增加。
        #
        # 空批次是**会真的发生**的：DAPO 的 dynamic_filter 有权丢掉全部样本，
        # 而按 plan_assembly 的顺序约束，filter 必须排在归一化之前。
        if n == 0:
            return batch

        # 单样本：必须报错。
        #
        # 与空批次相反 —— 这一条是**静默失效**。一个样本的标准差恒为 0
        # （不管它取什么值），归一化后奖励变成 0，梯度随之为 0，而训练循环
        # 照常推进 global_step、照常写日志、loss 看着还挺稳。
        # 没有比「什么都没学到却报告在学」更糟的失败模式了。
        if n == 1:
            raise ValueError(
                f"全局归一化需要至少 2 个样本才能定义标准差，收到 {n} 个。"
                f"单样本归一化会把奖励压成 0，梯度恒为 0 而训练照跑不误。"
                f"请调大 batch（rollout.num_generations × prompt 条数），"
                f"或在单样本场景下改用不做归一化的 advantage。"
            )

        # unbiased=False：这是**总体**标准差。我们要的是「这一批奖励的离散度」
        # 这个描述量本身，不是对某个更大总体的估计 —— 用无偏估计（除以 n-1）
        # 会让归一化后的尺度依赖于 batch 大小，而 batch 大小是个工程参数，
        # 不该悄悄改变优化目标的量纲。
        std = rewards.std(unbiased=False)
        batch[F.REWARDS] = (rewards - rewards.mean()) / (std + self.eps)

        with torch.no_grad():
            self._stats = {
                "global_normalize/raw_std": float(std),
                "global_normalize/raw_mean": float(rewards.mean()),
                # σ 近似为 0 意味着整批奖励几乎一样（全对或全错）。此时
                # 归一化会把数值噪声放大成满量级的信号 —— 这个比例值得盯。
                "global_normalize/degenerate_frac": float(
                    (std <= self.eps * 10.0).to(std.dtype)
                ),
            }
        return batch

    def metrics(self) -> dict[str, float]:
        return dict(self._stats)

    def describe(self) -> str:
        return f"{super().describe()}  eps={self.eps}"
