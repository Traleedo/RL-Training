"""控制器实现 —— 有状态的、每个训练步调整别处系数的组件。

与其它子包的区别：这里的组件**不碰 Batch**。它们的 ``requires`` / ``provides``
都是空的，因为它们的动作是「改另一个组件的 Python 属性」，而不是读写字段。
把需求声明留空是必须的：一旦声明了 ``ref_logprobs``，控制器就会错误地成为
Reference 模型被构建的原因（见 ``core/base_controller.py`` 的模块 docstring）。

新增一个控制器的步骤
--------------------
1. 在本目录建文件，继承 ``core.base_controller.Controller``；
2. 加 ``@register("controller", "<名字>")``；
3. 实现 ``on_train_step_end(metrics)``（基类里它是 ``NotImplementedError``）；
4. 在本文件 import 它。

注意：**不要** import 兄弟子包（``components.losses`` 之类）。
``tests/test_no_cross_imports.py`` 会静态拦截。需要对目标损失项做类型判断时，
用鸭子类型 —— 控制器只应该依赖「目标项有一个可写的 ``coef``」这个约定，
而不是具体的类。
"""

from __future__ import annotations

from components.controllers.adaptive_kl import AdaptiveKLController
from components.controllers.fixed_kl import FixedKLController

__all__ = ["AdaptiveKLController", "FixedKLController"]
