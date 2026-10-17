
from __future__ import annotations

from core.registry import register

from components.losses.policy_gradient import PolicyGradientLoss

__all__ = ["PPOClippedLoss"]


@register("loss", "ppo_clip")
class PPOClippedLoss(PolicyGradientLoss):
    """对称裁剪（ε=0.2）、全局 token 均值的策略梯度项。"""

    def __init__(self, clip_eps: float = 0.2) -> None:
        super().__init__(clip_eps=clip_eps, reduction="token_mean")

    def name(self) -> str:
        # 覆盖基类的 "policy_gradient" —— 指标前缀用它，改名会让历史曲线断掉。
        return "ppo_clip"
