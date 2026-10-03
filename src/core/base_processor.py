"""RewardProcessor 抽象 —— 奖励的可插拔后处理链。

Scorer 产出原始分之后、算优势之前，中间会串一条 processor 链，顺序就是 YAML
列表顺序。典型成员：

===================  =========================================================
实现                 作用
===================  =========================================================
``dynamic_filter``   DAPO 的动态采样：丢掉组内奖励全同的组
``group_normalize``  GRPO 的组内 ``(r - μ) / σ``（需 ``num_generations >= 2``）
``global_normalize`` 单采样（PPO 一类）的整批 ``(r - μ) / σ``
===================  =========================================================

**这一层能做的是「奖励级」加工，不是「优势级」。** 教科书 PPO 常写
``(adv - μ) / σ``，但 processor 链跑在 ``advantage.compute()`` **之前**
（见 ``engine/rl_trainer.py`` 的相位 B 与 D），那一刻 advantages 还不存在。
想要优势级归一化，只能在 ``AdvantageEstimator`` 里做 —— 做成 processor
在架构上就够不着。

**关于「归一化归属」这条容易踩的坑**

组内归一化放在 processor 而不放在 ``AdvantageEstimator`` 里，是为了让它可组合、
可关闭（DAPO 需要单独调它）。代价是复现「标准 GRPO」时要写两个配置项。
为了避免「同时开了 group_normalize 和一个自带归一化的 advantage，于是归一化两次、
advantage 被压成噪声级」这种静默失效，本类提供 ``normalizes_rewards`` 类属性，
advantage 侧有对应的 ``assumes_normalized_rewards``（语义是「advantage 那一层也做
归一化」，与属性名字面读法相反），trainer 启动时会检查冲突并报错。
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from core import interfaces as F
from core.batch import Batch
from core.component import Component

__all__ = ["RewardProcessor"]


class RewardProcessor(Component):
    """奖励后处理链上的一环。"""

    category = "processor"

    requires = frozenset({F.REWARDS})

    #: 大多数 processor 就地改 REWARDS，不新增字段
    provides = frozenset()

    #: 这个 processor 是否对奖励做了归一化（标准差缩放一类）。
    #: 与 ``AdvantageEstimator.assumes_normalized_rewards`` 冲突时启动即报错。
    normalizes_rewards: ClassVar[bool] = False

    #: 这个 processor 是否会**改变样本数量**（丢掉一些行）。目前只有 DAPO 的
    #: ``dynamic_filter`` 会。
    #:
    #: 声明它有实际作用，不是文档：``plan_assembly`` 会检查它排在链上所有
    #: ``normalizes_rewards=True`` 的 processor **之前**。反过来的话，
    #: 全局（或组内）统计量会把**即将被丢掉的样本**也算进基线与标准差 ——
    #: 那些样本本来就不该影响这一批的统计量。这是一个不会报错的统计错误。
    changes_batch_size: ClassVar[bool] = False

    @abstractmethod
    def process(self, batch: Batch) -> Batch:
        """就地改造 batch 并返回（通常是改 ``REWARDS``）。

        约定：

        - 不要悄悄使用 ``requires`` 里没声明的字段 —— 如果要用 ``ref_logprobs``，
          必须在 ``requires`` 或 ``__init__`` 里的 ``self.require(...)`` 声明，
          否则 ``plan_assembly`` 不会加载 Reference 模型，运行期会拿到 ``None``。
        - 是纯 batch 变换，不要在这里调模型（需要模型的场景请考虑做成 Scorer）。
        """
        raise NotImplementedError
