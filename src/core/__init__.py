"""核心抽象层。

这个包只放三样东西：**抽象接口、注册表、Batch 容器**。
它绝不 import ``components`` / ``models`` / ``engine`` —— 依赖方向只能是单向的。

因此新增组件时的铁律是：

    components/** 之间的实现文件**不许互相 import**，只能 ``from core import ...``

这条由 ``tests/test_no_cross_imports.py`` 静态守护。它保证了「把任意组件拔掉、
换上另一个同 category 的实现」永远可行 —— 也就是整个框架存在的理由。

关于下面的 ``__getattr__``
-------------------------
``Batch`` / ``tensor_ops`` / 三个携带张量的基类都要 ``import torch``，而
``import torch`` 是这个仓库最贵的一次导入（约两秒）。但 ``core.config``、
``core.registry``、``core.interfaces`` 都是纯 Python，脚本想只读一个 YAML
或只查一个名字时不该付这个钱。

所以只有 ``interfaces`` 与 ``registry`` 是急切导入的（``interfaces`` 必须在
最前面：其它子模块会 ``from core import interfaces as F``），其余按需加载。
``from core import Batch`` 与 ``from core.batch import Batch`` 两种写法都照常可用。
"""

from __future__ import annotations

import importlib
from typing import Any

# interfaces 必须最先导入：其他子模块会 `from core import interfaces as F`
from core import interfaces, registry

#: 名字 -> 定义它的子模块。命中后才真正 import 那个子模块。
_LAZY: dict[str, str] = {
    # 张量容器与算子（需要 torch）
    "Batch": "core.batch",
    "gather_token_logprobs": "core.tensor_ops",
    "masked_mean": "core.tensor_ops",
    "masked_sum": "core.tensor_ops",
    "broadcast_sequence_to_tokens": "core.tensor_ops",
    # 组件基建
    "Component": "core.component",
    "ForwardContext": "core.component",
    "register": "core.registry",
    "build": "core.registry",
    "get": "core.registry",
    "list_registered": "core.registry",
    "is_registered": "core.registry",
    "categories": "core.registry",
    "load_config": "core.config",
    "find_config": "core.config",
    "iter_algorithm_configs": "core.config",
    # 抽象基类
    "RolloutEngine": "core.base_rollout",
    "Scorer": "core.base_scorer",
    "RewardProcessor": "core.base_processor",
    "AdvantageEstimator": "core.base_advantage",
    "LossTerm": "core.base_loss",
    "LossComposition": "core.base_loss",
    "Actor": "core.base_model",
    "Critic": "core.base_model",
    "Reference": "core.base_model",
    "Logger": "core.base_logger",
    "Checkpointer": "core.base_checkpoint",
    "TrainerState": "core.base_checkpoint",
}

__all__ = [
    # 数据契约
    "Batch",
    "interfaces",
    # 组件基建
    "Component",
    "ForwardContext",
    "register",
    "build",
    "get",
    "list_registered",
    "is_registered",
    "categories",
    # 配置
    "load_config",
    "find_config",
    "iter_algorithm_configs",
    # 张量算子
    "gather_token_logprobs",
    "masked_mean",
    "masked_sum",
    "broadcast_sequence_to_tokens",
    # 抽象基类
    "RolloutEngine",
    "Scorer",
    "RewardProcessor",
    "AdvantageEstimator",
    "LossTerm",
    "LossComposition",
    "Actor",
    "Critic",
    "Reference",
    "Logger",
    "Checkpointer",
    "TrainerState",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # 缓存，后续访问不再走 __getattr__
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})
