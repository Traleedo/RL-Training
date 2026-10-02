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

        # 需要张量而非仅文本 —— 声明出来，装配阶段才认识这个依赖
        self.require(F.INPUT_IDS, F.ATTENTION_MASK)

    # ------------------------------------------------------------------
    @property
    def module(self):
        return self._model

    def score(self, batch: Batch) -> Batch:
        batch.require(F.INPUT_IDS, F.ATTENTION_MASK, who="HFOutcomeRewardScorer")
        device = next(self._model.parameters()).device

        scores: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(batch), self.batch_size):
                chunk = batch.select(slice(start, start + self.batch_size))
                outputs = self._model(
                    input_ids=chunk[F.INPUT_IDS].to(device),
                    attention_mask=chunk[F.ATTENTION_MASK].to(device),
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
