"""损失项实现。

本目录提供 PPO 的三项（``ppo_clip`` / ``value_loss`` / ``kl_k3``），
加上通用策略梯度（``policy_gradient``）与另外两个 KL 估计量
（``kl_k1`` / ``kl_k2``）。它们是按「新增一个损失项」的 SOP 写出来的样板 ——
也就是说，下面这段话描述的不只是约定，而是这些文件的实际结构。

离线家族的两项（``cross_entropy`` / ``dpo``）也在这里。
它们与上面那些**住在同一个目录、走同一条注册路径、由同一个
``LossComposition`` 组合** —— 区别只有 ``requires`` 写了什么：

- ``cross_entropy`` 只要 ``curr_logprobs`` + ``response_mask``；
- ``dpo`` 还要 ``ref_logprobs``（于是 Reference 被构建）与 ``preference``
  （于是只有成对数据集能配上它）。

「显式信号 / 隐式信号 / 监督」这个划分因此**没有对应任何基类或分支** ——
它就是六个损失项各自 ``requires`` 的差别。

新增一个损失项
--------------
新建 ``src/components/losses/my_custom.py``：

.. code-block:: python

    import torch
    from core import interfaces as F
    from core.base_loss import LossTerm
    from core.batch import Batch
    from core.registry import register
    from core.tensor_ops import masked_mean

    @register("loss", "my_custom")          # 双参：category 显式写出
    class MyCustomLoss(LossTerm):
        # 我要读哪些字段。plan_assembly 靠这个推导要构建哪些模型。
        # 如果我要 ref_logprobs，写在这里 —— Reference 模型就会被自动加载。
        requires = frozenset({F.CURR_LOGPROBS, F.RESPONSE_MASK, F.ADVANTAGES})

        # 我依赖的字段里哪些必须是可微的。合法值只有 curr_logprobs / values。
        grad_fields = frozenset({F.CURR_LOGPROBS})

        def __init__(self, coef: float = 1.0) -> None:
            super().__init__()
            self.coef = coef
            # 条件依赖：coef 为 0 时我根本不需要 advantages
            if coef == 0.0:
                self.release(F.ADVANTAGES)

        def name(self) -> str:
            return "my_custom"

        def compute(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
            mask = batch[F.RESPONSE_MASK]
            lp = batch[F.CURR_LOGPROBS]
            adv = batch[F.ADVANTAGES]
            loss = -self.coef * masked_mean(lp * adv, mask)   # 必须归约成 0 维
            return loss, {"mean_adv": float(masked_mean(adv, mask))}

然后在 ``src/components/losses/__init__.py`` 里 import 它，最后在 YAML 里加一行。

两条硬约束
----------
1. **``compute`` 必须是纯函数** —— 不调模型、不 backward、不碰 optimizer。
   需要额外前向的场景走 ``Component.prepare(batch, ctx)`` 钩子。
2. **返回值必须是 0 维标量**，且已经归约到 ``response_mask`` 上。
   加权求和由 ``LossComposition`` 负责，它不做归约（它不知道该用哪个 mask）。
"""

from __future__ import annotations

# 新增的损失项在这里 import —— **import 就是注册**。少一行，YAML 里引用它就会
# 报「未注册的组件」（tests/test_configs.py 会在跑测试时抓到）。
#
# 注意 policy_gradient 必须在 ppo_clip **之前**：ppo_clip 是它的子类，
# 反过来 import 会成环。顺序在这里是有意义的，不是随手排的。
from components.losses.cross_entropy import CrossEntropyLoss
from components.losses.dpo import DPOLoss
from components.losses.gspo import GSPOLoss
from components.losses.kl_penalty import KLK1Loss, KLK2Loss, KLK3Loss, KLEstimator
from components.losses.policy_gradient import PolicyGradientLoss
from components.losses.ppo_clip import PPOClippedLoss
from components.losses.value_loss import ValueLoss

__all__ = [
    "CrossEntropyLoss",
    "DPOLoss",
    "GSPOLoss",
    "KLEstimator",
    "KLK1Loss",
    "KLK2Loss",
    "KLK3Loss",
    "PolicyGradientLoss",
    "PPOClippedLoss",
    "ValueLoss",
]
