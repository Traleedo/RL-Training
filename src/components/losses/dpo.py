"""DPO —— 直接从偏好对比里学，不需要奖励模型。

.. math::

    L = -\\mathbb{E}_{(x, y_w, y_l)}\\left[\\log \\sigma\\left(\\beta (\\Delta_w - \\Delta_l)\\right)\\right]

    \\Delta_y = \\sum_t \\left(\\log \\pi_\\theta(y_t \\mid x, y_{<t}) - \\log \\pi_{\\text{ref}}(y_t \\mid x, y_{<t})\\right)

``Δ`` 是「策略相对参考模型给这条回答加了多少分」，标准 DPO 论文里记作
``β log(π_θ/π_ref)`` 的序列和。这里把 ``β`` 提到 sigmoid 外面写，
两者是同一个式子。

三件在别处容易写错的事
----------------------

**一、 ``masked_sum`` 而不是 ``masked_mean``。**
``Δ_y`` 一个 token 一个 token 地累加，分母是 1 不是 token 数。写成 mean 会得到
「每条回答的平均对数比」—— 一个形状对、数值小 L 倍的量。它不会报错，
loss 照样降，只是训练的其实是另一个目标。

**二、配对的依据是 ``preference`` 字段，不是行在组里的位置。**
``(Δ · sign)`` 在组内求和 —— ``sign`` 是 ``+1``（chosen）或 ``-1``（rejected），
所以结果恒等于 ``Δ_w − Δ_l``，与两行的先后无关。靠位置（「第 0 行是 chosen」）
的写法在 ``split`` / ``filter`` 打乱一次行序之后会**静默地训反**。

**三、一个对必须在同一次前向里。**
``needs_intact_groups = True``，trainer 据此改用 ``split_by_group``。
切散了的话 ``group_view`` 会抛错；如果实现成「切散了也照算」，损失会基于两个
不相干的样本，梯度方向随机而 loss 曲线照样好看。

四个参数长得像，含义完全不同
----------------------------
=========================  ==================================================
``weight``（YAML）          这一项在总损失里的**组合权重**，由 ``LossComposition`` 管
``beta``（本类）            sigmoid 内部的**温度**，控制偏离参考模型的惩罚力度
``label_smoothing``（本类） 把标签从 hard 0/1 软化，抗噪声偏好
``coef``（KL 项里的）       KL 惩罚的强度，DPO 没有这个参数
=========================  ==================================================

``beta`` 不是 `weight`：把 ``beta`` 调成 0.01 是「允许偏离参考模型很多」，
把它写成 ``weight: 0.01`` 是「这一项在总损失里只占 1%」—— 后者会让梯度变得
很小、看起来像是学习率出了问题。
"""

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

    #: ``ref_logprobs`` 在这里 —— 仅凭这一行，plan_assembly 就会推导出
    #: 「需要构建 Reference」。offline 家族里 SFT 与 DPO 的装配层差别就是它。
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
        """每个对必须恰好一个 ``+1`` 一个 ``-1``。

        这不是防御性编程，而是**「mini-batch 被切散了」的守卫**：``split`` 按行
        打乱之后，一个对的两行会分到不同的块里，而块的边界恰好可能凑出一个
        ``(+1, +1)`` 的组合 —— 那样算出来的 ``Δ_w − Δ_l`` 是零，loss 恒为
        ``log 2``、梯度恒为 0，训练悄悄停摆。
        """
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

        # 序列级的「策略相对参考模型加了多少分」。[B]
        # 注意是 sum 不是 mean —— 见模块 docstring 第一条。
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
                # 隐式奖励差（β·margin）：正数 = 模型确实更偏好 chosen。
                # 它是 DPO 版的「奖励」，也是唯一能看出训练有没有效果的量。
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
        """DPO 通常是**唯一**的损失项，权重取 1.0。

        它是自包含的（β 已经写在 sigmoid 里），不需要再与 KL 项叠加 ——
        再加一个 kl_penalty 相当于把参考约束算了两遍。
        """
        return 1.0
