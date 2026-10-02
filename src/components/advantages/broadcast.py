"""序列级奖励直接广播成逐 token 优势 —— 无 critic 算法（GRPO 一类）的优势。

为什么这样是对的
----------------
GRPO 的假设是「一整个回答共享同一个优势」。所以 ``A_{i,t} = R_i`` 对所有
response token ``t`` 成立，不需要价值网络估计基线 —— 基线由组内均值提供
（见 ``components/processors/group_normalize``）。

这也是 GRPO 与 PPO 的**唯一结构性差异**：本类的 ``requires`` 里没有
``ROLLOUT_VALUES``，于是 ``plan_assembly`` 推导出「不需要 Critic」，
那一整个价值网络就不会被加载。PPO 用 ``advantage/gae``，它的 ``requires``
里有 ``ROLLOUT_VALUES``，于是 Critic 被建出来。

**「PPO 需要 Critic」不是 trainer 里的一个 if，而是这一行声明。**

与 PPO 共享的细节
----------------
``ADVANTAGES`` 一律写逐 token 的 ``[B, L]``，序列级的量用
``core.tensor_ops.broadcast_sequence_to_tokens`` 广播到 ``response_mask`` 上。
这样所有 loss term 拿到的形状约定是一致的，loss 那边不需要知道
优势是来自组内比较还是来自 GAE。
"""

from __future__ import annotations

import torch

from core import interfaces as F
from core.base_advantage import AdvantageEstimator
from core.batch import Batch
from core.registry import register
from core.tensor_ops import broadcast_sequence_to_tokens, masked_mean

__all__ = ["BroadcastAdvantage"]


@register("advantage", "broadcast")
class BroadcastAdvantage(AdvantageEstimator):
    """``A_{i,t} = R_i`` —— 一个回答内的所有 token 共享同一个优势。"""

    requires = frozenset({F.REWARDS, F.RESPONSE_MASK})

    #: 不写 RETURNS：没有价值网络，也就没有「价值回归的目标」这回事。
    #: 硬塞一个 returns 只会让 plan_assembly 以为有人需要它。
    provides = frozenset({F.ADVANTAGES})

    #: 用完即弃的字段：这里没有。声明一个不存在的 transient 会让释放逻辑
    #: 去 drop 一个从未存在的字段 —— 无害但会误导读代码的人。
    transient_requires = frozenset()

    def __init__(self) -> None:
        super().__init__()
        self._stats: dict[str, float] = {}

    def compute(self, batch: Batch) -> Batch:
        batch.require(F.REWARDS, F.RESPONSE_MASK, who="BroadcastAdvantage")
        mask = batch[F.RESPONSE_MASK]
        advantages = broadcast_sequence_to_tokens(batch[F.REWARDS], mask)
        batch[F.ADVANTAGES] = advantages

        with torch.no_grad():
            # 优势的标准差是「归一化是否漏了」最快的可观测信号：
            # 它应当接近奖励的组内尺度；恒为 0 说明组内归一化之后全被压平了。
            self._stats = {
                "advantage/mean": float(masked_mean(advantages, mask)),
                "advantage/std": (
                    float(advantages[mask > 0].std()) if mask.sum() > 1 else 0.0
                ),
            }
        return batch

    def metrics(self) -> dict[str, float]:
        return dict(self._stats)
