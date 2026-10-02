from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar, Iterator

import torch
from torch import nn

from core import interfaces as F
from core.batch import Batch
from core.component import Component

__all__ = ["Actor", "Critic", "Reference"]


class _Weighted(Component):
    """有可训练参数的组件的公共部分。"""

    @abstractmethod
    def parameters(self) -> Iterator[nn.Parameter]:
        """可训练参数，供 trainer 建优化器。"""
        raise NotImplementedError

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """带名字的参数，供梯度裁剪与调试。默认从 ``parameters()`` 合成。"""
        for i, param in enumerate(self.parameters()):
            yield f"param_{i}", param

    def train(self, mode: bool = True) -> None:
        """切换训练模式。默认无操作，持有 nn.Module 的实现应该转发。"""

    def eval(self) -> None:
        self.train(False)

    @property
    def module(self) -> nn.Module | None:
        """底层 ``nn.Module``，供 Checkpointer / LoRA 等需要摸权重的地方使用。"""
        return None

    def state_dict(self) -> dict[str, Any]:
        """要进 checkpoint 的权重。

        默认语义是「全部参数」。**LoRA 实现应该覆盖它**，只返回 adapter 的权重，
        外加一个基座模型的指纹 —— 否则每次 checkpoint 都会存下一份完整基座。
        """
        module = self.module
        if module is None:
            return {}
        return {k: v.detach().cpu() for k, v in module.state_dict().items()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        module = self.module
        if module is None:
            if state:
                raise RuntimeError(
                    f"{type(self).__name__} 没有底层 module，无法加载 {len(state)} 项权重"
                )
            return
        module.load_state_dict(state)

    @property
    def base_model_hash(self) -> str | None:
        """基座模型的指纹。LoRA 场景下用于校验 resume 时基座没被换掉。"""
        return None


class Actor(_Weighted):
    """策略模型：算当前策略下的 logprob，并持有可训练参数。"""

    category = "actor"

    requires = frozenset({F.INPUT_IDS, F.ATTENTION_MASK, F.RESPONSE_MASK})

    provides = frozenset({F.CURR_LOGPROBS})

    @abstractmethod
    def logprobs(self, batch: Batch) -> Batch:
        """前向，写**可微的** ``CURR_LOGPROBS: [B, L]``。
        """
        raise NotImplementedError


class Critic(_Weighted):
    """价值模型。只在算法声明需要 ``values`` 时才被构建。"""

    category = "critic"

    requires = frozenset({F.INPUT_IDS, F.ATTENTION_MASK, F.RESPONSE_MASK})

    provides = frozenset({F.VALUES, F.ROLLOUT_VALUES})

    @abstractmethod
    def forward_values(self, batch: Batch, *, detach: bool = False) -> Batch:
        raise NotImplementedError


class Reference(_Weighted):
    """参考模型：冻结的策略，用于算 KL。只在需要 ``ref_logprobs`` 时才被构建。"""

    category = "reference"

    requires = frozenset({F.INPUT_IDS, F.ATTENTION_MASK, F.RESPONSE_MASK})

    provides = frozenset({F.REF_LOGPROBS})

    @abstractmethod
    def logprobs(self, batch: Batch) -> Batch:
        raise NotImplementedError
