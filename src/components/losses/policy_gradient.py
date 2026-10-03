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
