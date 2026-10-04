from __future__ import annotations

from typing import Any

import torch

from core import interfaces as F
from core.base_scorer import Scorer
from core.batch import Batch
from core.registry import register

from models.hf_common import load_reward_model

__all__ = ["HFOutcomeRewardScorer"]


@register("scorer", "hf_reward_model")
class HFOutcomeRewardScorer(Scorer):
    """用一个序列分类模型给「完整对话」打一个标量分数。"""

    def __init__(
        self,
        model_config: Any,
        tokenizer: Any = None,
        model: Any = None,
        batch_size: int = 16,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if model is None:
            model, tokenizer = load_reward_model(model_config, tokenizer)
        self._model = model
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.scale = float(scale)

    # ------------------------------------------------------------------
    @property
    def module(self):
        return self._model

    def score(self, batch: Batch) -> Batch:
        # 依赖的是**文本**而不是 actor 的 input_ids。基类 Scorer 的 requires
        # 声明的是 prompt_texts / response_texts，就是为这件事 —— 奖励模型
        # 几乎必然与 actor 不是同一个词表（Skywork 是 Llama-3.1，actor 是
        # Qwen2.5）。batch 里的 input_ids 是 actor 的 tokenizer 产出的，
        # 直接喂给奖励模型的 embedding 会 IndexError: index out of range in self。
        # 所以必须用奖励模型自己的 tokenizer 把文本重新编码一遍。
        batch.require(
            F.PROMPT_TEXTS, F.RESPONSE_TEXTS, who="HFOutcomeRewardScorer"
        )
        device = next(self._model.parameters()).device

        full_texts = [
            str(prompt) + str(response)
            for prompt, response in zip(batch[F.PROMPT_TEXTS], batch[F.RESPONSE_TEXTS])
        ]

        scores: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(full_texts), self.batch_size):
                encoded = self.tokenizer(
                    full_texts[start : start + self.batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                outputs = self._model(
                    input_ids=encoded["input_ids"].to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                )
                logits = outputs.logits
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)
                scores.append(logits.float())

        rewards = torch.cat(scores, dim=0) * self.scale
        batch[F.REWARDS] = rewards.to(batch[F.INPUT_IDS].device)
        return batch

    def metrics(self) -> dict[str, float]:
        return {}
