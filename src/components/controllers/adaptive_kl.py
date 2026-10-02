"""自适应 KL 控制器 —— β 自己去找一个让 KL 落在目标附近的量级。

.. math::

    \\beta \\leftarrow \\mathrm{clip}\\bigl(\\beta + \\eta\\,(\\mathrm{KL}_{\\text{obs}} - \\mathrm{KL}_{\\text{target}}),\\ \\beta_{\\min},\\ \\beta_{\\max}\\bigr)

为什么需要它
------------
固定 β 有一个说不出口的问题：**合适的量级完全取决于任务**。KL 惩罚太弱则策略
跑飞（奖励涨、KL 爆炸、生成退化成重复），太强则策略被钉死在参考模型上（KL 很小、
奖励不涨）。而「合适的 β」在不同任务间能差三个数量级 —— 于是每换一个任务，
调 β 就变成一轮新的玄学。

自适应 β 把这个超参数换成两个更好指定的量：**你想要的 KL 是多少**（``target_kl``），
以及**调整多快**（``step_size``）。前者是任务语义（「允许多大程度偏离参考模型」），
后者是控制论参数。

初值从哪来：**绑定项的 coef**
-----------------------------
控制器**不**自己存一份初始 β。起始值就是 YAML 里那个 KL 项的 ``coef``。
这样只有一个真相来源 —— 两处各写一份（``beta_init`` 与 ``coef``）迟早会不一致，
而不一致的表现是「控制器以为自己在 0.01 起步、实际损失项用的是 0.1」，
从曲线上完全看不出来。

这也是为什么 ``coef`` 必须 > 0：``coef == 0`` 会让 Reference 模型根本不被构建
（见 ``core.base_controller.check_target`` 的长注释）。

β 的生效时机
------------
本钩子在一个 train_step 的**最后**跑，而损失是在 epoch × mini-batch 循环**内部**
算的。所以第 k 步写进去的 β 从第 **k+1** 步开始生效。理由见 ``core.base_controller``
的模块 docstring：要让 β 在当前步的剩余 mini-batch 生效，就得把它塞进
``LossComposition`` 的循环里，那样每个 mini-batch 会读到不同的 β，
梯度不再对应同一个目标函数。

读的是**绑定项自己**的 ``mean_kl``
--------------------------------
控制器调节的是它所绑定那个估计量的 ``mean_kl``，不是某个独立的 KL 度量。
所以绑 ``kl_k3`` 时它把 k3 估计拉到 ``target_kl`` 附近，绑 ``kl_k2`` 时拉 k2。
这两个量的数值并不相等（k3 是 KL 的无偏-ish 估计，k2 是二阶矩），
所以**换估计量就要重设 ``target_kl``** —— 这是语义上的必然，不是实现细节。
"""

from __future__ import annotations

from core.base_controller import Controller
from core.registry import register

__all__ = ["AdaptiveKLController"]


@register("controller", "adaptive_kl")
class AdaptiveKLController(Controller):
    """按「观测 KL 与目标 KL 的差」成比例地调整 β。"""

    def __init__(
        self,
        term: str | None = None,
        target_kl: float = 0.01,
        step_size: float = 0.1,
        beta_min: float = 0.0,
        beta_max: float = 1.0,
    ) -> None:
        super().__init__(term=term)
        self.target_kl = float(target_kl)
        self.step_size = float(step_size)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

        if self.beta_min > self.beta_max:
            raise ValueError(
                f"beta_min={self.beta_min} 大于 beta_max={self.beta_max}，"
                f"夹取区间为空，β 会被钉在 beta_min 上。"
            )
        if self.step_size <= 0.0:
            raise ValueError(
                f"step_size={self.step_size} 必须为正，否则 β 永远不会变化。"
                f"如果想保持 β 恒定，请改用 controller/fixed_kl。"
            )

    # ------------------------------------------------------------------
    def on_train_step_end(self, metrics: dict[str, float]) -> None:
        key = f"loss/{self.target_name}/mean_kl"
        if key not in metrics:
            available = sorted(k for k in metrics if k.startswith("loss/"))
            raise KeyError(
                f"AdaptiveKLController 需要指标 {key!r}，但本步的指标里没有它。\n"
                f"    它绑定的是损失项 {self.target_name!r}，所以那一项必须在 compute() "
                f"返回的指标字典里报 'mean_kl'（kl_k1 / kl_k2 / kl_k3 都报）。\n"
                f"    本步可用的 loss/* 指标：{available}"
            )

        kl = float(metrics[key])
        previous = self.current_coef
        updated = previous + self.step_size * (kl - self.target_kl)
        beta = min(max(updated, self.beta_min), self.beta_max)
        self.current_coef = beta

        # 上报写进 metrics（而不是走 metrics() 方法）—— trainer 轮询 metrics()
        # 的时机在这个钩子**之前**，那时 β 还是上一步的旧值。就地写才能保证
        # 日志里这一行的 β 就是刚刚写进损失项的 β。
        prefix = f"controller/{self.target_name}"
        metrics[f"{prefix}/beta"] = beta
        metrics[f"{prefix}/kl"] = kl
        metrics[f"{prefix}/kl_error"] = kl - self.target_kl
        # 顶到边界 = 控制器已经失去控制权。β 卡在下界说明 target_kl 定得太低，
        # 卡在上界说明定得太高（或 step_size 太小追不上）。没有这个信号的话，
        # 「β 一直线性往上爬」和「KL 根本没在收敛」在曲线上长得一模一样。
        metrics[f"{prefix}/at_bound"] = (
            1.0 if beta >= self.beta_max else (-1.0 if beta <= self.beta_min else 0.0)
        )

    def describe(self) -> str:
        return (
            f"{super().describe()}  target_kl={self.target_kl} "
            f"step_size={self.step_size} bounds=[{self.beta_min}, {self.beta_max}]"
        )
