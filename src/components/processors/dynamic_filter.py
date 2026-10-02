"""DAPO 的动态采样 —— 丢掉「组内奖励全同」的组。

为什么值得丢
------------
组内奖励全同（全对或全错）时，组内归一化之后优势恒为 0：

- 全对：``r - μ = 0``，这一组不提供任何梯度信号；
- 全错：同理。

这些组占了算力却不产生梯度。DAPO 的做法是直接把它们丢掉，
腾出的预算去采样更有信息量的 prompt。这是**效率**优化，不是正确性修正 ——
所以它默认不开，需要它在 YAML 里显式加一行。

三个必须做对的地方
------------------
1. **过滤之后必须重写 ``GROUP_IDS``。**

   ``filter_groups`` 只按行选，留下来的组的 id 会带空洞（原 id ``[1,1,3,3]``）。
   而 ``Batch.group_by_prompt`` 的散射回退路径是
   ``[select(gids == g) for g in range(n_groups)]`` —— 用 ``n_groups = 2``
   去匹配 id ``1`` 和 ``3``：

   - ``gids == 0`` 匹配不到任何行 -> **返回一个空组**；
   - id ``3`` 那两行**永远不会被访问** -> 整组静默消失。

   它只打一条 warning，不报错。这是本项目里「不重写 id」最坏的结果，
   比「破坏连续块契约」严重得多，所以这里无条件重写。

2. **必须排在 processor 链的第一位**（在 ``global_normalize`` 之类之前）。

   否则全局统计量会把**即将被丢掉的样本**也算进均值和标准差 ——
   那些样本本来就不该影响这一批的基线。这是一个不会报错的统计错误。
   ``plan_assembly`` 会用 ``RewardProcessor.changes_batch_size`` 检查这条顺序。

3. **它可能把整个批次丢空**，而空批次在 trainer 里是**不能直接跑**的
   （见 ``engine.rl_trainer`` 的空批次守卫）。所以本组件在丢空时不报错，
   只是让批次变空 —— 由 trainer 决定怎么处理（记 warning、跳过、不推进步数）。
   「丢空」在策略已经很擅长这批 prompt 时是正常现象，不是异常。
"""

from __future__ import annotations

import torch

from core import interfaces as F
from core.base_processor import RewardProcessor
from core.batch import Batch
from core.registry import register

__all__ = ["DynamicFilterProcessor"]


@register("processor", "dynamic_filter")
class DynamicFilterProcessor(RewardProcessor):
    """丢掉组内奖励极差 <= ``eps`` 的组（DAPO 动态采样）。"""

    #: 会让 batch 的**行数变少**。plan_assembly 靠这个标志检查它在链上的位置。
    changes_batch_size = True

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self._stats: dict[str, float] = {}

    def process(self, batch: Batch) -> Batch:
        rewards = batch[F.REWARDS]
        group_size = int(batch.meta.get(F.NUM_GENERATIONS, 1))

        if group_size < 2:
            raise ValueError(
                f"动态采样需要 num_generations >= 2，收到 {group_size}。"
                f"单采样时每一「组」只有一个样本，组内极差恒为 0，"
                f"结果是整个批次都被丢掉。"
            )
        if len(batch) % group_size:
            raise ValueError(
                f"batch 维 {len(batch)} 不能被 num_generations {group_size} 整除，"
                f"分组会残缺 —— 残缺的组上算出的极差没有意义。"
            )

        groups = rewards.reshape(-1, group_size)
        spread = groups.max(dim=1).values - groups.min(dim=1).values
        keep = spread > self.eps

        kept = batch.filter_groups(keep, group_size)

        # 无条件重写 —— 理由见模块 docstring 第 1 条。过滤后组仍是连续块，
        # 所以 group_ids() 生成的 arange(G').repeat_interleave(n) 恰好成立，
        # 既恢复了契约，也让 group_by_prompt 走回快路径。
        kept[F.GROUP_IDS] = kept.group_ids(group_size)

        n_groups = int(keep.numel())
        n_kept = int(keep.sum())
        with torch.no_grad():
            self._stats = {
                "dynamic_filter/groups_in": float(n_groups),
                "dynamic_filter/groups_kept": float(n_kept),
                # 偏大说明这批 prompt 对该策略几乎全对或全错。它是「要不要换一批
                # prompt」的第一个信号 —— 比等到 reward 曲线变平要早得多。
                "dynamic_filter/drop_frac": float((n_groups - n_kept) / max(n_groups, 1)),
            }
        return kept

    def metrics(self) -> dict[str, float]:
        return dict(self._stats)

    def describe(self) -> str:
        return f"{super().describe()}  eps={self.eps}"
