"""优势估计实现。

骨架阶段这里是空的。新增时继承 ``core.base_advantage.AdvantageEstimator``，
加 ``@register("advantage", "<名字>")``，然后在本文件 import。

写之前务必想清楚的一件事：**归一化放在哪**
------------------------------------------
组内 ``(r - μ) / σ`` 这类归一化**归 RewardProcessor**，不归这里。
本类只做「奖励 -> 逐 token 优势」的转换。

这样切分是为了让归一化可组合、可单独关闭（DAPO 需要单独调它）。
代价是复现「标准 GRPO」要写两个配置项（一个 group_normalize processor +
一个广播式的 advantage），这个代价是划算的。

为了拦住「两处都做导致归一化两次」这种会把优势压成噪声级的静默失效：

- 如果某个 advantage **自己内部会做归一化**（例如组内 ``(r - μ) / σ``），
  把 ``assumes_normalized_rewards = True``；
- 如果某个 processor 会归一化，把 ``normalizes_rewards = True``。

两者同时成立时 ``plan_assembly`` 会在启动时直接报错。

⚠️ 前一个属性名读起来像「我假定奖励已被归一化」，**语义正好相反** ——
它表示「我这一层做归一化」。按约定归一化归 processor，所以绝大多数 advantage
（包括本目录下的 ``GAEAdvantage``）应当保持默认的 ``False``。

形状约定：``ADVANTAGES`` 一律写逐 token 的 ``[B, L]``。序列级的量用
``core.tensor_ops.broadcast_sequence_to_tokens`` 广播到 ``response_mask`` 上。
"""

from __future__ import annotations

from components.advantages.broadcast import BroadcastAdvantage
from components.advantages.gae import GAEAdvantage

__all__ = ["BroadcastAdvantage", "GAEAdvantage"]
