"""PPO 的裁剪策略损失。

这个文件现在只是一层**默认值**。真正实现是 ``policy_gradient`` ——
两者用的是同一个公式，``ppo_clip`` 把两个旋钮钉在 PPO 论文的取值上：

- ``clip_eps = 0.2``（对称裁剪）
- ``reduction = "token_mean"``（全局 token 均值）

保留这个名字有两个实际理由：

1. ``configs/ppo.yaml`` 以及所有从它复制出来的配置**一个字都不用改**；
2. 日志里的指标前缀仍然是 ``loss/ppo_clip/...``，历史曲线可以直接对比。

想调非对称裁剪或换归约方式，直接用 ``policy_gradient``，别在这里加参数 ——
一个名字对应一组固定语义，是这个仓库让配置可读的方式。参数细节与三个必须
写对的地方（不自己 shift、``min`` 里的两项都乘原 advantages、``grad_fields``
只含 ``curr_logprobs``）都写在 ``policy_gradient.py`` 的模块 docstring 里。
"""

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
