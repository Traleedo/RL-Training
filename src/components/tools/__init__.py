"""具体工具实现（P0 阶段为空占位）。

P1 会加入 ``calculator``（纯本地函数）与 ``python_exec``（走
``core.sandbox_client.SandboxClient``）。扇出 import 放在这里：
``components.tools`` 一被 import，所有工具就完成注册 —— 与其它子包的约定一致。

工具不进 ``plan_assembly``：它们由 rollout 引擎（``rollout.type: tool_agent``）
按配置持有并驱动，不参与 Batch 字段依赖推导。见 ``core/base_tool.py`` 的说明。
"""

from __future__ import annotations

__all__: list[str] = []
