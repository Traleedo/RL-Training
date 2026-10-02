"""GSPO —— 把 ratio 从「逐 token」改成「逐序列」，再裁一次。

.. math::

    s_i(\\theta) = \\left(\\frac{\\pi_\\theta(y_i|x_i)}{\\pi_{\\text{old}}(y_i|x_i)}\\right)^{1/|y_i|}
                 = \\exp\\left(\\frac{1}{|y_i|}\\sum_t \\log\\frac{\\pi_\\theta}{\\pi_{\\text{old}}}\\right)

    J = \\mathbb{E}_i\\Big[\\min\\big(s_i A_i,\\ \\mathrm{clip}(s_i, 1-\\epsilon, 1+\\epsilon)\\, A_i\\big)\\Big]

与 ``policy_gradient`` 的本质区别
--------------------------------
**裁剪的粒度不同，而不是裁剪的幅度不同。**

先把三个量分清楚，它们是三个东西：

===========================  ===============================  ==============================
量                           定义                             量级
===========================  ===============================  ==============================
逐 token ratio ``r_t``       ``π_θ(a_t)/π_old(a_t)``          ``O(1)``
序列重要性权重 ``R_i``        ``Π_t r_t``                      ``O(1)^{|y_i|}`` —— 随长度爆炸
序列比值 ``s_i``（本项用）    ``R_i^{1/|y_i|}``（几何平均）     ``O(1)``
===========================  ===============================  ==============================

逐 token 裁剪（PPO/GRPO/DAPO）把每个 ``r_t`` 各自夹进 ``[1-ε, 1+ε]``。
它**管不住 ``R_i``**：``|y_i|`` 个 1.01 逐个看去都在界内，而
``R_i = 1.01^512 ≈ 163`` —— 那个才是乘在序列奖励上的系数。序列越长缺口越大，
所以 PPO 系在长 response 上会出现「权重已经飞出天际，却没有任何一个 token
被判为 clipped」。

GSPO 换成对 ``s_i`` 裁剪。``s_i`` 是 ``R_i`` 的**长度归一化**版本，按构造就是
``O(1)``；把 ``ε`` 放在这个尺度上，一条序列「整体偏了多少」才是被直接约束
的量。这也解释了 GSPO 论文里 ``ε = 3e-4`` 为什么这么小：它和 PPO 的 ``0.2``
**不在同一个尺度上**，两者的大小不可直接比较。

代价是 ``clip_frac`` 的含义变了（见下）。

为什么``log_ratio`` 要先按 token 求平均再取 exp
---------------------------------------------
数学上 ``exp(mean_t log r_t) = (Π_t r_t)^{1/|y_i|}``，即 token ratio 的**几何平均**
—— 正是 ``s_i`` 的定义。之所以不写成连乘，除了溢出（``R_i`` 在几百个 token 上
直接变 inf），还有一个更基本的原因：**这里要的就是几何平均，不是连乘**。
写成连乘得到的 ``R_i`` 是另一个量，不是 GSPO。

三个必须写对的地方
------------------
1. **只裁一次，裁的是 ``s_i``。** 若写成对逐 token ratio 裁剪再平均，得到的
   就是另一个算法 —— 而且它在短 response 上看起来几乎一样，只有在长
   response 上才暴露。``tests/test_gspo.py`` 用「每个 token 都在界内、
   序列比值却是 10^50」这个构造把它钉住了。

2. **优势必须是序列级的。** ``A_i`` 在论文里是组内归一化后的奖励，是一个
   序列一个数。本框架的 ``advantage`` 契约统一写 ``[B, L]``，所以这里用
   ``response_mask`` 上的按行均值把它还原成 ``[B]``。**如果优势本来就是逐
   token 的（例如把 advantage 配成了 ``gae``），这个还原会把它压成一个
   平均值，算法就不是 GSPO 了** —— 而且不会报错。所以本项在运行期检查
   优势在 response_mask 内是否恒定，不恒定就直接报错。

3. **不要自己 shift。** 同 ``policy_gradient``：``curr_logprobs`` 与
   ``rollout_logprobs`` 已经逐位置对齐，``gather_token_logprobs`` 是全仓库
   唯一的 shift 出口。``tests/test_no_manual_shift.py`` 会静态扫描这件事。

``clip_frac`` 的语义与 ``policy_gradient`` 不同
--------------------------------------------
这里上报的是 ``seq_clip_frac`` —— **被裁剪的序列占比**，不是 token 占比。
故意用不同的键名：同一个 ``clip_frac`` 在两个项里一个按 token 算、一个按
序列算，是很容易看错的一类混淆。论文里 GSPO 的 ε 取得很小（量级 3e-4），
所以这个比例通常远低于 PPO 的 0.05 ~ 0.2。
"""

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
        """把 ``[B, L]`` 的优势还原成 ``[B]``，并拒绝逐 token 的优势。

        拒绝而不是悄悄取均值：GSPO 的 ``A_i`` 是一个序列一个数，而本框架的
        advantage 契约统一写 ``[B, L]``。把 ``gae`` 那种真正的逐 token 优势
        平均掉之后，损失仍然算得出来、曲线仍然好看，但优化的目标已经不是
        GSPO 了 —— 这类「配方对了但料不对」的错必须报出来。
        """
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
