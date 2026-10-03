"""奖励后处理链实现。

链的顺序就是 YAML 列表顺序。Scorer 出原始分，然后依次经过这些 processor，
最后才去算优势。

典型的链长这样::

    processors:
      - {type: dynamic_filter}            # 会丢样本 -> 必须排在最前面
      - {type: group_normalize}           # GRPO 的组内归一化
      - {type: global_normalize}          # PPO 的全局归一化（单采样）

写 processor 时注意三件事
------------------------
1. **用了什么字段就要声明**。比如要用 ``ref_logprobs``，就必须在
   ``requires`` 里写出来，否则 ``plan_assembly`` 不会加载 Reference 模型，
   运行期你会拿到 ``None`` 而不是一个明确的报错。这引出一个反直觉但正确的
   结论：**奖励处理链上的组件可以决定哪些模型被构建。**
   启动时打印的装配计划会写明是谁导致了哪个模型被加载。

2. **做了归一化就要自报**：把 ``normalizes_rewards = True`` 写进类属性。
   否则它和 ``AdvantageEstimator.assumes_normalized_rewards`` 的冲突检查失效，
   归一化两次会把优势压成噪声级，而 loss 曲线看起来一切正常。

3. **会丢样本就要自报**：把 ``changes_batch_size = True`` 写进类属性。
   否则 ``plan_assembly`` 检查不了「它必须排在会做统计的 processor 之前」——
   而顺序反了会把即将被丢掉的样本算进基线，同样不会报错。
"""

from __future__ import annotations

# 新增的 processor 在这里 import —— **import 就是注册**。
from components.processors.dynamic_filter import DynamicFilterProcessor
from components.processors.global_normalize import GlobalNormalizeProcessor
from components.processors.group_normalize import GroupNormalizeProcessor

__all__ = [
    "DynamicFilterProcessor",
    "GlobalNormalizeProcessor",
    "GroupNormalizeProcessor",
]
