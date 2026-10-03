from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, Sequence

import torch

from core import interfaces as F
from core.batch import Batch
from core.component import Component

__all__ = ["LossTerm", "LossComposition"]


class LossTerm(Component):
    """一项损失。"""

    category = "loss"

    requires = frozenset({F.CURR_LOGPROBS, F.RESPONSE_MASK})
    provides = frozenset()

    grad_fields: ClassVar[frozenset[str]] = frozenset({F.CURR_LOGPROBS})
    needs_intact_groups: ClassVar[bool] = False

    @abstractmethod
    def compute(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        """日志里的名字，会作为指标前缀。默认实现建议返回类名的小写下划线形式。"""
        raise NotImplementedError

    def weight_hint(self) -> float | None:
        """给使用者看的推荐权重，仅用于文档/自省，不影响计算。"""
        return None


class LossComposition:
    def __init__(self, terms: Sequence[tuple[LossTerm, float]]) -> None:
        self.terms: list[tuple[LossTerm, float]] = list(terms)

    def __len__(self) -> int:
        return len(self.terms)

    def needed(self) -> frozenset[str]:
        fields: set[str] = set()
        for term, _ in self.terms:
            fields |= term.needed
        return frozenset(fields)

    def grad_fields(self) -> frozenset[str]:
        fields: set[str] = set()
        for term, _ in self.terms:
            fields |= term.grad_fields
        return frozenset(fields)

    def __call__(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        mask = batch[F.RESPONSE_MASK]
        total = torch.zeros((), device=mask.device, dtype=torch.float32)
        metrics: dict[str, float] = {}

        for term, weight in self.terms:
            if weight == 0.0:
                continue
            loss, term_metrics = term.compute(batch)
            if loss.dim() != 0:
                raise ValueError(
                    f"loss term {term.name()!r} 返回了 {loss.dim()} 维张量，"
                    f"必须归约成 0 维标量。"
                )
            total = total + weight * loss
            prefix = f"loss/{term.name()}"
            metrics[f"{prefix}"] = float(loss.detach())
            metrics[f"{prefix}/weighted"] = float((weight * loss).detach())
            for key, value in term_metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.detach()
                metrics[f"{prefix}/{key}"] = float(value)

        if not torch.isfinite(total):
            raise FloatingPointError(
                f"总损失不是有限值（{float(total)}）。"
            )
        return total, metrics

    def __repr__(self) -> str:
        items = ", ".join(f"{t.name()}×{w}" for t, w in self.terms)
        return f"<LossComposition [{items}]>"
