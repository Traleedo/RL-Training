from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from core.batch import Batch

__all__ = ["Component", "ForwardContext"]


class ForwardContext:
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
    requires: ClassVar[frozenset[str]] = frozenset()
    provides: ClassVar[frozenset[str]] = frozenset()
    category: ClassVar[str] = ""
    transient_requires: ClassVar[frozenset[str]] = frozenset()

    def __init__(self) -> None:
        self._extra_requires: set[str] = set()

    def require(self, *fields: str) -> None:
        """追加实例级依赖。在 ``__init__`` 里根据配置调用。"""
        self._extra_requires.update(fields)

    def release(self, *fields: str) -> None:
        """移除依赖（包括类属性里的）。在 ``__init__`` 里根据配置调用。"""
        self._extra_requires.difference_update(fields)
        removed = getattr(self, "_removed_requires", None)
        if removed is None:
            removed = set()
            self._removed_requires = removed
        removed.update(fields)

    @property
    def needed(self) -> frozenset[str]:
        removed = getattr(self, "_removed_requires", frozenset())
        return (self.requires | frozenset(self._extra_requires)) - removed
    
    def metrics(self) -> dict[str, float]:
        """组件自报的标量指标，会被 logger 记录。默认无。"""
        return {}

    def prepare(self, batch: "Batch", ctx: ForwardContext) -> "Batch":
        return batch

    def on_train_step_end(self, metrics: dict[str, float]) -> None:
        """可选钩子：一个 train_step 完全结束后被调用。默认无。"""

    def state_dict(self) -> dict[str, Any]:
        """需要进 checkpoint 的额外状态（优化器状态之外的）。默认空。"""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """恢复 ``state_dict()`` 存下的状态。默认什么都不做。"""

    def to(self, device: Any) -> "Component":
        """把底层 ``module`` 搬到 ``device``。返回 ``self``，便于链式调用。

        默认实现走 ``self.module`` —— 对绝大多数持有 ``nn.Module`` 的组件够用。
        **但 ``module`` 只是「这个组件暴露出来的那个模块」，不等于「它的全部
        参数」。** 最典型的反例是共享骨架的 critic：它的 backbone 与 value head
        是两个平级的 ``nn.Module``，而 ``module`` 只返回 backbone。那种组件必须
        覆盖本方法，否则 value head 会独自留在 CPU 上 —— 直到前向时才报
        device mismatch，而报错指向的是 ``nn.Linear``，离真正的原因很远。

        纯逻辑组件（processor / advantage / loss / controller）没有 module，
        这里是空操作。
        """
        module = getattr(self, "module", None)
        if module is not None:
            module.to(device)
        return self

    def describe(self) -> str:
        """一行摘要，启动时打印，方便确认配置到底装配出了什么。"""
        key = getattr(type(self), "_registry_key", None)
        label = f"{key[0]}/{key[1]}" if key else type(self).__name__
        return f"{label}  needs={sorted(self.needed)}  provides={sorted(self.provides)}"

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.describe()}>"
