from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from core import interfaces as F
from core.batch import Batch
from core.component import Component

__all__ = ["AdvantageEstimator"]


class AdvantageEstimator(Component):
    """计算优势。"""

    category = "advantage"

    requires = frozenset({F.REWARDS})
    provides = frozenset({F.ADVANTAGES})
    transient_requires = frozenset({F.ROLLOUT_VALUES})
    assumes_normalized_rewards: ClassVar[bool] = False

    @abstractmethod
    def compute(self, batch: Batch) -> Batch:
        raise NotImplementedError
