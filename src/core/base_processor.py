from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar
from core import interfaces as F
from core.batch import Batch
from core.component import Component

__all__ = ["RewardProcessor"]


class RewardProcessor(Component):
    """奖励后处理链上的一环。"""

    category = "processor"
    requires = frozenset({F.REWARDS})
    provides = frozenset()
    normalizes_rewards: ClassVar[bool] = False
    changes_batch_size: ClassVar[bool] = False

    @abstractmethod
    def process(self, batch: Batch) -> Batch:
        raise NotImplementedError
