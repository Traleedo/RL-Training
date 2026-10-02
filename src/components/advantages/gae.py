"""GAE —— PPO 一类「有 critic」算法的优势估计。

这个文件就是 PPO 与 GRPO 的分界线
---------------------------------
``requires`` 里的 ``ROLLOUT_VALUES`` 是全部差异。``core.interfaces.MODEL_FOR_FIELD``
把它映射到 ``critic``，所以 ``plan_assembly`` 只要看到有组件需要这个字段，就会把
Critic 构建出来；GRPO 的 advantage 不需要它，Critic 根本不会被加载。

换句话说：**「PPO 需要 Critic」这件事不是 trainer 里的一个 if，而是这一行声明。**

归一化不在这里
--------------
``assumes_normalized_rewards`` 保持默认的 ``False``。PPO 用 ``global_normalize``
processor 做归一化（``normalizes_rewards = True``），若这里也置 True，
``plan_assembly`` 会判定「归一化做了两次」并在启动时直接报错 —— 那确实是个
会静默毁掉训练的配置。

两个容易写错的地方
------------------
1. **padding 会污染反向递推。** ``Batch.from_rollout`` 只对 response 右侧补 pad
   （见 ``core/batch.py``），所以 response 区域内**存在**长度为 0 的列。递推时
   必须用 ``response_mask`` 把上一条序列的 delta 截断，否则 padding 位置会把
   尾巴上的值一路带回去，而且不会有任何报错。

2. **序列末尾要 bootstrap 成 0。** 这里的约定是「一条 response 就是一个完整
   episode」，所以最后一个 response token 的 ``next_value`` 取 0。
   如果生成是被 ``max_new_tokens`` 截断的，严格说应该 bootstrap 自截断状态的
   价值（``γ · V(s_T)``），但那需要把 ``values`` 在序列末尾之后的位置也喂进来 ——
   本项目按「截断即终止」处理，这是简化，不是错误。要改成真 bootstrap，
   在 ``_next_values`` 里把 ``* mask`` 换成对截断行的特殊处理即可。
"""

from __future__ import annotations

import torch

from core import interfaces as F
from core.base_advantage import AdvantageEstimator
from core.batch import Batch
from core.registry import register
from core.tensor_ops import masked_mean

__all__ = ["GAEAdvantage"]


@register("advantage", "gae")
class GAEAdvantage(AdvantageEstimator):
    """``δ_t = r_t + γ·V(t+1) - V(t)``，``A_t = δ_t + γλ·A_{t+1}``。

    即时奖励的约定：序列级奖励只落在**最后一个** response token 上，其余为 0。
    这与 ``fixture_response_mean`` 打分器的输出形状一致（``rewards`` 是 ``[B]``）。
    """

    # ROLLOUT_VALUES 是 PPO 的身份标识 —— 它一出现，Critic 就会被构建。
    requires = frozenset({F.REWARDS, F.ROLLOUT_VALUES, F.RESPONSE_MASK})

    provides = frozenset({F.ADVANTAGES, F.RETURNS})

    # 基类已声明 transient_requires = {ROLLOUT_VALUES}（用完即弃，省显存）。
    # 但 PPO 的 value_loss 需要拿它当裁剪基准，所以 trainer 算出的
    # _releasable = transient - loss_needs 会自动把它排除，不会误删。

    def __init__(self, gamma: float = 0.99, lam: float = 0.95) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.lam = float(lam)
        self._stats: dict[str, float] = {}

    # ------------------------------------------------------------------
    def compute(self, batch: Batch) -> Batch:
        batch.require(
            F.REWARDS, F.ROLLOUT_VALUES, F.RESPONSE_MASK, who="GAEAdvantage"
        )
        values = batch[F.ROLLOUT_VALUES]                      # [B, L]
        mask = batch[F.RESPONSE_MASK].to(values.dtype)        # [B, L]
        rewards = batch[F.REWARDS]                            # [B]

        deltas = self._deltas(rewards, values, mask)

        # 反向递推。`* mask[:, t]` 那一下是关键：它让 padding 位置把累积量清零，
        # 从而不会跨行泄漏。
        advantages = torch.zeros_like(deltas)
        running = torch.zeros_like(rewards)
        for t in range(deltas.shape[-1] - 1, -1, -1):
            running = deltas[:, t] + self.gamma * self.lam * running * mask[:, t]
            advantages[:, t] = running

        batch[F.ADVANTAGES] = advantages
        batch[F.RETURNS] = (advantages + values) * mask

        # 记下来给 metrics() 用。指标必须在 compute 里算完就 detach ——
        # 这个相位在 no_grad 下跑，但 detach 的意图要写出来，别依赖上下文。
        with torch.no_grad():
            self._stats = {
                "advantage/mean": float(masked_mean(advantages, mask)),
                # advantage 的标准差是「归一化是否做了两次」最快的可观测信号：
                # 归一化两次会把它压到 1e-3 量级甚至更低。
                "advantage/std": float(advantages[mask > 0].std()) if mask.sum() > 1 else 0.0,
            }
        return batch

    # ------------------------------------------------------------------
    def _deltas(
        self, rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """TD 残差 ``δ_t = r_t + γ·V(t+1) - V(t)``，padding 位置为 0。"""
        # ---- 即时奖励：只落在最后一个 response token 上 ----
        # cumsum 在最后一个 mask=1 的位置正好等于该行的 token 总数。
        # `& (mask > 0)` 是必要的：其后的 padding 列 cumsum 不再增长，
        # 也会等于总数，不排掉就会把奖励重复撒到 padding 上。
        lengths = mask.sum(-1, keepdim=True)
        is_last = ((mask.cumsum(-1) == lengths) & (mask > 0)).to(values.dtype)
        r_t = rewards.unsqueeze(-1) * is_last

        # ---- 下一个状态的价值：右移一位 ----
        # 序列末尾（及之后的 padding）取 0，即「截断视作终止」。见模块 docstring。
        #
        # 注意 mask 是跟着一起右移的：`values[:, 1:] * mask[:, 1:]`。
        # 写成 `values[:, 1:]` 之后再乘 `mask`（不右移）是**错的** ——
        # 那样位置 t 拿到的是 mask[t]，而它管的是位置 t 自己。
        # 最后一个 response token 处 mask[t] = 1，于是紧邻的 padding 列里
        # 未被清理的价值会直接漏进 TD 残差，把整条序列的优势抬高一个量级。
        #
        # 这个 bug 只在 response **变长**（response 区域内存在 padding 列）时出现，
        # 等长数据的测试完全看不到它 —— tests/test_gae.py 专门用变长 mask 守这一条。
        next_values = torch.zeros_like(values)
        next_values[:, :-1] = values[:, 1:] * mask[:, 1:]

        return (r_t + self.gamma * next_values - values) * mask

    # ------------------------------------------------------------------
    def metrics(self) -> dict[str, float]:
        return dict(self._stats)

    def describe(self) -> str:
        return f"{super().describe()}  gamma={self.gamma} lam={self.lam}"
