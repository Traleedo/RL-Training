"""比例裁剪策略梯度 —— 一个实现覆盖 GRPO / Dr.GRPO / DAPO / PPO。

.. math::

    L = -\\ \\mathrm{reduce}\\Big(\\min\\big(r_t A_t,\\ \\mathrm{clip}(r_t,\\ 1-\\epsilon_l,\\ 1+\\epsilon_h)\\, A_t\\big)\\Big),
    \\quad r_t = \\exp(\\log\\pi_\\theta - \\log\\pi_{\\text{old}})

这个文件是「换算法 = 改配置」这句话最直接的证据：GRPO、Dr.GRPO、DAPO、PPO
**用的是同一个公式**，差别只在三个旋钮上。

三个旋钮
--------
1. **裁不裁**（``clip_eps`` / ``clip_eps_low`` + ``clip_eps_high``）
   - 不裁 —— GRPO / Dr.GRPO。原论文认为组内归一化已经提供了足够的约束。
   - 对称裁 —— PPO，``clip_eps: 0.2``。
   - 非对称裁 —— DAPO 的 **clip-higher**，``clip_eps_low: 0.2`` 配
     ``clip_eps_high: 0.28``。抬高上界是为了松开「低概率 token 的概率
     涨不上去」的束缚：熵塌缩往往就是从这些 token 开始的。

2. **怎么归约**（``reduction``）—— 见下节。这是最容易被忽略、却实实在在
   改变梯度的一项。

3. **优势从哪来**（不在这里，在 ``advantage`` + ``processor``）
   有 critic 用 GAE，没有就用 ``broadcast`` 广播组内归一化后的奖励。

``reduction`` 的三种模式
-----------------------
记 ``s_i = Σ_t (x_{i,t} · m_{i,t})`` 为第 i 条的逐 token 和，``|o_i|`` 为其
response token 数，``B`` 为非空序列数。

===============  ==========================  ====================================
模式              公式                        用它的算法
===============  ==========================  ====================================
``seq_mean``     ``(1/B) Σ_i s_i / |o_i|``   GRPO 原论文
``token_mean``   ``Σ_i s_i / Σ_i |o_i|``     DAPO
``seq_sum``      ``(1/B) Σ_i s_i``           Dr.GRPO 的「去掉长度归一化」
===============  ==========================  ====================================

三者的差别不是常数倍，而是**权重分配**：``seq_mean`` 里每条序列说话一样重，
于是长序列里的每个 token 说话更轻（这就是「长度偏差」）；``token_mean`` 里每个
token 说话一样重，长序列自然权重更大；``seq_sum`` 里根本不除长度。
在 response 长度参差不齐时它们给出**不同**的梯度，
``tests/test_policy_gradient.py`` 会把这个差异钉住。

关于 Dr.GRPO 的诚实说明
----------------------
Dr.GRPO 想要的是「去掉 σ 归一化」这一半 —— 那一半在
``processor/group_normalize`` 的 ``divide_std: false`` 上，与本文件无关。
它的另一半是「不按 ``|o_i|`` 归一化」：论文用一个**常数**除数（最大生成长度）
而非常见的 ``|o_i|``。本框架对应的模式是 ``seq_sum``。

严格复刻论文的话，``seq_sum`` 与它只差一个 ``1/L_max`` —— 而 ``L_max`` 来自
配置（``rollout.max_new_tokens``），是个**固定常数**，不是逐批次的量。常数因子
等价于重新缩放有效学习率，用 ``weight`` 或 ``optim.lr`` 吸收即可，所以这里
不必再加一个 ``divisor`` 旋钮。

（值得留意的是 ``seq_sum`` 与 ``token_mean`` 之间差的才是**逐批次的**标量
—— ``(1/B)·Σ_i|o_i|``，即本批平均长度。所以这两者给出**相同方向**的梯度，
只有 ``seq_mean`` 真的改变了各序列的权重分配。）

怎么用 YAML 表达
----------------
.. code-block:: yaml

    # GRPO：不裁、每条序列等权
    - {type: policy_gradient, weight: 1.0, reduction: seq_mean}

    # DAPO：clip-higher、每个 token 等权
    - {type: policy_gradient, weight: 1.0, reduction: token_mean,
       clip_eps_low: 0.2, clip_eps_high: 0.28}

两个必须写对的地方
------------------
1. **不要自己 shift。** ``curr_logprobs`` 与 ``rollout_logprobs`` 都已经与
   ``input_ids`` 逐位置对齐 —— ``core.tensor_ops.gather_token_logprobs`` 是全仓库
   唯一的 shift 出口。直接逐元素相减即可，再切一次 ``[:, :-1]`` 会让形状仍然对得上
   而语义错位一位。``tests/test_no_manual_shift.py`` 会静态扫描这件事。

2. **``min`` 里的两项都乘 ````原```` advantages。** 正确形式是「未裁剪项」与
   「裁剪后的 ratio 乘**同一个** advantages」取逐元素最小值。这个 ``min``
   本身就实现了非对称性（``A > 0`` 时只在上界截断，``A < 0`` 时只在下界截断），
   所以非对称裁剪**不需要**另写一套逻辑 —— 只要把上下界设成不一样的即可。
   如果改写成对 advantage 也做裁剪的对称形式，梯度会错。

关于 ``approx_kl``
------------------
用 k3 估计 ``E[(r - 1) - log r]`` 而不是 ``E[-log r]``。后者样本均值可正可负
（期望为 0），在监控曲线上分不清「KL 很小」和「符号写反了」；k3 恒非负。
"""

from __future__ import annotations

import torch

from core import interfaces as F
from core.base_loss import LossTerm
from core.batch import Batch
from core.registry import register
from core.tensor_ops import masked_mean, masked_sum

__all__ = ["PolicyGradientLoss", "REDUCTIONS"]

#: 合法的归约模式 -> 一句话说明。报错时直接列出来，省得去翻源码。
REDUCTIONS = {
    "seq_mean": "先按序列求均值再对序列取均值（每条序列等权，有长度偏差）",
    "token_mean": "全局 token 均值（每个 token 等权，固定分母）",
    "seq_sum": "按序列求和再对序列取均值（不除长度，Dr.GRPO 原意）",
}


@register("loss", "policy_gradient")
class PolicyGradientLoss(LossTerm):
    """通用比例裁剪策略梯度项。"""

    requires = frozenset(
        {F.CURR_LOGPROBS, F.ROLLOUT_LOGPROBS, F.ADVANTAGES, F.RESPONSE_MASK}
    )

    #: ``rollout_logprobs`` 出现在 ``requires`` 里是因为本项**读**它，
    #: 但它必须作为常量参与（它已被 freeze 保护）。只有 ``curr_logprobs`` 可微。
    grad_fields = frozenset({F.CURR_LOGPROBS})

    def __init__(
        self,
        clip_eps: float | None = None,
        clip_eps_low: float | None = None,
        clip_eps_high: float | None = None,
        reduction: str = "seq_mean",
    ) -> None:
        super().__init__()

        # ---- 裁剪参数：对称与非对称互斥 ----
        # 两者同时给会让人猜「哪个赢」，而猜错的表现是梯度悄悄变了 ——
        # 这类歧义在启动时就该报错，不该留到看曲线的时候。
        asymmetric = clip_eps_low is not None or clip_eps_high is not None
        if clip_eps is not None and asymmetric:
            raise ValueError(
                "clip_eps 与 clip_eps_low/clip_eps_high 不能同时给："
                "前者是对称裁剪，后者是非对称裁剪（DAPO 的 clip-higher），"
                "两套参数同时存在时无法判断哪个生效。请二选一。"
            )
        if asymmetric and (clip_eps_low is None or clip_eps_high is None):
            # 报错必须指出**真正缺的那个** —— 说反了会把人引向已经写对的那一行。
            missing, given = (
                ("clip_eps_low", "clip_eps_high")
                if clip_eps_low is None
                else ("clip_eps_high", "clip_eps_low")
            )
            raise ValueError(
                f"非对称裁剪必须同时给出上下界，缺 {missing}（只给了 {given}）。"
                f"如果本意是对称裁剪，请改用 clip_eps。"
            )

        if clip_eps is not None:
            if clip_eps < 0:
                raise ValueError(f"clip_eps 必须非负，收到 {clip_eps}")
            self.clip_eps_low = float(clip_eps)
            self.clip_eps_high = float(clip_eps)
            self.clips = True
        elif asymmetric:
            if clip_eps_low < 0 or clip_eps_high < 0:
                raise ValueError(
                    f"裁剪上下界必须非负，收到 low={clip_eps_low} high={clip_eps_high}"
                )
            self.clip_eps_low = float(clip_eps_low)
            self.clip_eps_high = float(clip_eps_high)
            self.clips = True
        else:
            # 不裁 —— GRPO / Dr.GRPO。不是「忘了配」，是一个正当的选择。
            self.clip_eps_low = 0.0
            self.clip_eps_high = 0.0
            self.clips = False

        if reduction not in REDUCTIONS:
            raise ValueError(
                f"未知的 reduction {reduction!r}；合法取值："
                + "；".join(f"{k}（{v}）" for k, v in REDUCTIONS.items())
            )
        self.reduction = reduction

    def name(self) -> str:
        return "policy_gradient"

    def weight_hint(self) -> float:
        return 1.0

    def describe(self) -> str:
        clip = (
            f"clip=[1-{self.clip_eps_low}, 1+{self.clip_eps_high}]"
            if self.clips
            else "clip=off"
        )
        return f"{super().describe()}  {clip} reduction={self.reduction}"

    # ------------------------------------------------------------------
    def _reduce(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """把逐 token 张量归约成标量。三种模式的唯一实现处。"""
        if self.reduction == "token_mean":
            return masked_mean(values, mask)

        active = (mask.sum(dim=-1) > 0).to(values.dtype)
        # 只数「真的有 response token」的行：全 padding 的行不该稀释均值，
        # 否则一个坏样本就能把整步的梯度按比例压小。
        n_rows = active.sum().clamp(min=1.0)

        if self.reduction == "seq_mean":
            return masked_mean(values, mask, dim=-1).sum() / n_rows
        # seq_sum
        return masked_sum(values, mask) / n_rows

    def compute(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        batch.require(
            F.CURR_LOGPROBS,
            F.ROLLOUT_LOGPROBS,
            F.ADVANTAGES,
            F.RESPONSE_MASK,
            who="PolicyGradientLoss",
        )
        mask = batch[F.RESPONSE_MASK]
        log_ratio = batch[F.CURR_LOGPROBS] - batch[F.ROLLOUT_LOGPROBS]
        ratio = torch.exp(log_ratio)
        advantages = batch[F.ADVANTAGES]

        if self.clips:
            clipped_ratio = torch.clamp(
                ratio, 1.0 - self.clip_eps_low, 1.0 + self.clip_eps_high
            )
            objective = torch.min(ratio * advantages, clipped_ratio * advantages)
        else:
            objective = ratio * advantages

        loss = -self._reduce(objective, mask)

        with torch.no_grad():
            if self.clips:
                is_clipped = (
                    (ratio > 1.0 + self.clip_eps_high)
                    | (ratio < 1.0 - self.clip_eps_low)
                ).to(mask.dtype)
            else:
                # 不裁时这个量恒为 0，但仍然上报 —— 指标键在所有配置间保持一致，
                # 曲线可以直接叠着看，也免得有人以为「没打印就是坏了」。
                is_clipped = torch.zeros_like(ratio)
            metrics = {
                # ratio 偏离 1 的平均幅度。第一个 mini-batch 必须精确为 1，
                # 之后应当缓慢偏离 —— 它暴涨就说明 lr 太大或者 old logprob 被刷新了。
                "mean_ratio": float(masked_mean(ratio, mask)),
                # 被裁剪的 token 比例。健康的 PPO 一般在 0.05 ~ 0.2。
                "clip_frac": float(masked_mean(is_clipped, mask)),
                "approx_kl": float(masked_mean((ratio - 1.0) - log_ratio, mask)),
                # 负优势占比。若恒为 0，说明奖励没有区分度或归一化出了问题。
                "frac_neg_adv": float(
                    masked_mean((advantages < 0).to(mask.dtype), mask)
                ),
                # 平均 response 长度。seq_mean 与 token_mean 的差别完全由它决定，
                # 不记下来就永远看不出两者为什么给出不同的梯度。
                "mean_seq_len": float(mask.sum(dim=-1).to(torch.float32).mean()),
            }
        return loss, metrics
