"""所有组件的公共基类 —— 依赖声明协议。

``requires`` / ``provides`` 的意义
----------------------------------
每个组件静态声明自己「读哪些字段」和「写哪些字段」。Trainer 把所有已配置组件的
``needed`` 求并集，就能推导出**这个算法到底需要哪些模型**：

- 没有任何组件需要 ``ref_logprobs``  -> 不加载 Reference 模型
- 没有任何组件需要 ``values``        -> 不构建 Critic（GRPO 场景）

于是「换算法」退化成「改 YAML」，而不是「改 trainer 里的 if-else」。

为什么 ``needed`` 是实例属性而不是类属性
----------------------------------------
类属性 ``requires`` 只能表达**无条件**依赖。但真实需求里有一大类条件依赖：

    kl_penalty(coef=0.0)  不应该需要 ref_logprobs
    ppo_clip(clip_eps=None) 退化成纯策略梯度，不需要 old_logprobs

这些条件依赖只有在**看到配置之后**才知道。所以组件在 ``__init__`` 里读完自己的
参数后，调用 ``self.require(...)`` / ``self.release(...)`` 来调整。

Trainer 求并集时遍历的是 ``component.needed``（实例属性），
而组件构建本身很便宜（纯逻辑，不加载权重），所以这个顺序是可行的：
**先建便宜的组件 -> 问它们要什么 -> 再建贵的模型**。
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from core.batch import Batch

__all__ = ["Component", "ForwardContext"]


class ForwardContext:
    """传给 ``Component.prepare`` 的只读上下文。

    绝大多数组件用不到它 —— 它们应该只依赖 ``Batch``，因为 ``LossTerm.compute``
    的契约是「Batch 的纯函数」。但确实存在少数需要额外前向的组件，例如：

    - 用不同 mask 对同一序列再跑一次前向的对比项
    - 需要当前 reward model 打分的 loss term
    - 双前向 KL 估计

    给这些情况留一个受控出口，好过让它们绕过框架直接摸 ``self.actor``。

    **重要**：``prepare`` 里对模型的访问**不参与** ``plan_assembly`` 的模型构建
    推导（它是运行期动态的，静态推不出来）。所以如果你在 ``prepare`` 里用了
    ``ctx.reference``，必须在 ``requires`` 里也声明 ``ref_logprobs``，
    或者确保配置里已经让 Reference 被构建，否则运行期会拿到 ``None``。
    """

    __slots__ = ("_actor", "_critic", "_reference", "_device")

    def __init__(self, actor, critic, reference, device) -> None:
        self._actor = actor
        self._critic = critic
        self._reference = reference
        self._device = device

    @property
    def actor(self):
        return self._actor

    @property
    def critic(self):
        return self._critic

    @property
    def reference(self):
        return self._reference

    @property
    def device(self):
        return self._device


class Component(ABC):
    """所有可插拔组件的基类。"""

    #: 静态依赖：这个组件要读 Batch 里的哪些字段
    requires: ClassVar[frozenset[str]] = frozenset()

    #: 静态产出：这个组件会写 Batch 里的哪些字段
    provides: ClassVar[frozenset[str]] = frozenset()

    #: 所属注册表 category（与 @register 的第一个参数一致）
    category: ClassVar[str] = ""

    #: 只在**自己执行期间**需要、之后可以被释放的字段。
    #:
    #: 存在的理由是显存：这类框架第一次跑就可能 OOM，因为 B×n×L 的中间张量
    #: 有十几个。典型例子是 GAE 需要的 ``rollout_values`` —— 它在优势相位用完
    #: 就再也用不到了，但如果不声明，它会一路跟到训练相位占着显存。
    #:
    #: trainer 的释放规则是「声明为 transient 的字段，减去**训练相位确实还需要**
    #: 的字段」。所以 ``rollout_logprobs`` 即便被某个 advantage 声明为 transient，
    #: 只要 PPO 的 loss term 还需要它，就不会被释放。
    transient_requires: ClassVar[frozenset[str]] = frozenset()

    def __init__(self) -> None:
        self._extra_requires: set[str] = set()

    # ------------------------------------------------------------------
    # 实例级依赖调整
    # ------------------------------------------------------------------
    def require(self, *fields: str) -> None:
        """追加实例级依赖。在 ``__init__`` 里根据配置调用。"""
        self._extra_requires.update(fields)

    def release(self, *fields: str) -> None:
        """移除依赖（包括类属性里的）。在 ``__init__`` 里根据配置调用。"""
        self._extra_requires.difference_update(fields)
        # 用一个「移除集合」记录需要从类属性里挖掉的字段
        removed = getattr(self, "_removed_requires", None)
        if removed is None:
            removed = set()
            self._removed_requires = removed
        removed.update(fields)

    @property
    def needed(self) -> frozenset[str]:
        """这个组件实例实际需要的字段集合。

        Trainer 求并集时用的是**这个**属性，不是类属性 ``requires``。
        """
        removed = getattr(self, "_removed_requires", frozenset())
        return (self.requires | frozenset(self._extra_requires)) - removed

    # ------------------------------------------------------------------
    # 可选钩子
    # ------------------------------------------------------------------
    def metrics(self) -> dict[str, float]:
        """组件自报的标量指标，会被 logger 记录。默认无。"""
        return {}

    def prepare(self, batch: "Batch", ctx: ForwardContext) -> "Batch":
        """可选钩子：需要额外前向的组件在这里做。默认恒等。

        调用时机在 trainer 训练相位的 F1（可微前向）之后、F2（loss 计算）之前。
        参见 ``ForwardContext`` 的说明了解它的能力边界与注意事项。
        """
        return batch

    def on_train_step_end(self, metrics: dict[str, float]) -> None:
        """可选钩子：一个 train_step 完全结束后被调用。默认无。

        只有 ``category == "controller"`` 的组件会被 trainer 调用 —— 它们靠这个
        钩子根据本步观测到的指标去调整别的组件的参数（adaptive KL 就是典型）。
        其余 category 实现这个方法不会被执行，因为它们没有「每步调整一次系数」
        的语义，多调用一次只会制造出「谁在什么时候改了什么」的困惑。

        钩子可以**就地修改** ``metrics``：改动会进入本步的日志。
        实现这个钩子时注意它跑在 ``global_step`` 自增之前，所以往 ``metrics``
        里写的键要带自己的命名前缀，免得和别的组件的指标撞名。
        """

    def state_dict(self) -> dict[str, Any]:
        """需要进 checkpoint 的额外状态（优化器状态之外的）。默认空。"""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """恢复 ``state_dict()`` 存下的状态。默认什么都不做。"""

    def describe(self) -> str:
        """一行摘要，启动时打印，方便确认配置到底装配出了什么。"""
        key = getattr(type(self), "_registry_key", None)
        label = f"{key[0]}/{key[1]}" if key else type(self).__name__
        return f"{label}  needs={sorted(self.needed)}  provides={sorted(self.provides)}"

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.describe()}>"
