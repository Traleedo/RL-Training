from __future__ import annotations

from abc import abstractmethod

from core import interfaces as F
from core.batch import Batch
from core.component import Component

__all__ = ["Scorer"]


class Scorer(Component):
    """把生成结果打成分数。"""

    category = "scorer"
    requires = frozenset({F.PROMPT_TEXTS, F.RESPONSE_TEXTS})
    provides = frozenset({F.REWARDS})
    @abstractmethod
    def score(self, batch: Batch) -> Batch:
        raise NotImplementedError
