from __future__ import annotations

from abc import abstractmethod
from typing import Any

from core import interfaces as F
from core.batch import Batch
from core.component import Component

__all__ = ["RolloutEngine"]


class RolloutEngine(Component):
    """生成序列并给出行为策略的 logprob。"""

    category = "rollout"

    requires = frozenset({F.PROMPT_TEXTS})

    provides = frozenset({
        F.INPUT_IDS,
        F.ATTENTION_MASK,
        F.RESPONSE_MASK,
        F.ROLLOUT_LOGPROBS,
        F.GROUP_IDS,
        F.RESPONSE_TEXTS,
    })

    def __init__(
        self,
        num_generations: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> None:
        super().__init__()
        if num_generations < 1:
            raise ValueError(f"num_generations 必须 >= 1，收到 {num_generations}")
        self.num_generations = int(num_generations)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

    def resolve(self, **overrides: Any) -> dict[str, Any]:
        resolved = {
            "num_generations": self.num_generations,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        for key, value in overrides.items():
            if key in resolved and value is not None:
                resolved[key] = type(resolved[key])(value)
        if resolved["num_generations"] < 1:
            raise ValueError(f"num_generations 必须 >= 1，收到 {resolved['num_generations']}")
        return resolved

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        *,
        num_generations: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **gen_kwargs: Any,
    ) -> Batch:
        raise NotImplementedError
