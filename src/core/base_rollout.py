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

    # ------------------------------------------------------------------
    # 权重生命周期钩子
    #
    # 单机 HF rollout 与 actor 共享**同一个模块对象**（见 RLTrainer._assemble），
    # 所以优化器一 step，rollout 的权重就同时更新了，不需要任何同步。
    # 而外部推理引擎（sglang）持有一份**独立的**权重副本：不同步的话，
    # 训练到第 10 步时 rollouts 还是第 0 步的策略采出来的，而
    # rollout_logprobs 却是「谁采的」—— PPO 的 ratio 从此恒等于 1，
    # 重要性采样悄悄失效，loss 曲线完全正常。
    #
    # 框架只负责「在对的时机调用」，具体怎么做由实现决定：
    # 纯本地实现保持 no-op（默认），外部引擎实现覆盖。
    # ------------------------------------------------------------------
    @property
    def uses_external_weights(self) -> bool:
        """rollout 是否持有独立于 actor 的权重副本。"""
        return False

    def sync_weights(self, actor_module: Any = None) -> None:
        """优化器更新之后，把 actor 的新权重推给 rollout。默认无操作。"""

    def close(self) -> None:
        """释放外部资源（子进程 / HTTP 连接 / 引擎）。默认无操作。"""

    def describe(self) -> str:
        """一行摘要，启动日志用。"""
        return f"{type(self).__name__}(num_generations={self.num_generations})"

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
