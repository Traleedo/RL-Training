"""组件实现层 —— **注册的唯一扇出点**。

只有被 import 过的实现才会出现在注册表里。所以新增一个组件之后，除了写文件和加
装饰器，还必须在这个文件里加一行 import。忘了加的症状是：
报错说「未注册 xxx」，但文件明明存在。``core/registry.py`` 的报错信息里会提示这一点。

新增组件的完整步骤（README 里也有一份）
---------------------------------------
1. 在对应 category 的子包里建文件，继承抽象基类；
2. 加 ``@register("<category>", "<名字>")`` 装饰器（双参，category 显式写出）；
3. **在这里 import 它**；
4. 在 YAML 里加一行 ``{type: <名字>, weight: ...}``。

trainer 一行都不用改。

目录约定
--------
``components/`` 放**纯逻辑**组件（只依赖 core，不碰 transformers）：

    losses/        损失项
    advantages/    优势估计
    controllers/   每个训练步调整别处系数的有状态组件（自适应 KL 等）
    scorers/       奖励来源（规则验证器、外部 API 等）
    processors/    奖励后处理链
    loggers/       日志后端
    checkpointers/ checkpoint 后端

**这里的实现文件之间不许互相 import**，只能 ``from core import ...``。
这条由 ``tests/test_no_cross_imports.py`` 静态守护 —— 它保证了「任意组件都能被
拔掉换成另一个同 category 的实现」，也就是整个框架存在的理由。

需要 transformers 的实现（HF 策略模型、价值模型、rollout 引擎）放在
``src/models/``，本文件也会顺便 import 它。那是唯一一处跨层的 wiring，
不是实现之间的依赖。
"""

from __future__ import annotations

# ---- 纯逻辑组件的扇出 ----
from components import (
    advantages,
    checkpointers,
    controllers,
    loggers,
    losses,
    processors,
    scorers,
)

# ---- HF 实现层（可选：transformers 不可用时会自己降级）----
import models  # noqa: F401  触发 actor/critic/reference/rollout 的注册

# 数据层（``data/``）**不在这里**扇出。它的组件同样要在启动时注册，但这条边
# 由 ``engine.trainer.ensure_components_registered()`` 负责 —— components 不该
# 认识 data（两个平级的实现层之间没有依赖关系），而 trainer 作为 wiring 的那一层
# 认识它们两个是自然的。tests/test_no_cross_imports.py 守着这条边界。

__all__ = [
    "losses",
    "advantages",
    "controllers",
    "scorers",
    "processors",
    "loggers",
    "checkpointers",
    "models",
]
