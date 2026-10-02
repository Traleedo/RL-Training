"""固定 β 控制器 —— 不调整任何东西，但把 β 记进日志。

它为什么存在
------------
功能上它等价于「不挂控制器」：β 就是 YAML 里那个 KL 项的 ``coef``，从头到尾不变。

写出来的价值在于两点：

1. **配置对称。** 从自适应 β 切回固定 β，只需要把 ``type`` 改一个词，
   不用把整段 ``controllers:`` 删掉。删掉的话，下次想再打开又得重新查一遍
   参数名和嵌套层级。
2. **日志对称。** 两条曲线（固定 / 自适应）都会打出 ``controller/*/beta``，
   可以直接画在一起比较。如果固定 β 那条没有这个键，对比就得手工记 YAML 里的数。

换句话说：这是一个**为了可观测性和可切换性**而存在的组件，不是因为固定 β
需要什么计算。它的 ``on_train_step_end`` 确实是空操作 —— 但那是它的语义，
不是没写完。
"""

from __future__ import annotations

from core.base_controller import Controller
from core.registry import register

__all__ = ["FixedKLController"]


@register("controller", "fixed_kl")
class FixedKLController(Controller):
    """β 保持 YAML 里写的值。只上报，不修改。"""

    def on_train_step_end(self, metrics: dict[str, float]) -> None:
        # 刻意不写 self.current_coef —— 固定 β 的全部语义就是「没人改它」。
        prefix = f"controller/{self.target_name}"
        metrics[f"{prefix}/beta"] = self.current_coef
