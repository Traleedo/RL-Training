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
        batch.require(
            F.PROMPT_TEXTS, F.RESPONSE_TEXTS, who="HFOutcomeRewardScorer"
        )
        device = next(self._model.parameters()).device

        # Skywork-Reward-V2-Qwen3-8B 是序列分类模型，官方用法要求把「完整对话」
        # 用 chat 模板包成 [user, assistant] 再送进去，**不是**裸文本直接拼接。
        # 裸拼接会喂出一堆越界 token，打分退化成几乎常数 -> group_normalize 的
        # σ≈0 -> advantage≈0 -> 策略梯度≈0，这就是「训练几乎不动」的根因。
        full_texts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": str(prompt)},
                    {"role": "assistant", "content": str(response)},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            for prompt, response in zip(
                batch[F.PROMPT_TEXTS], batch[F.RESPONSE_TEXTS]
            )
        ]

        scores: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(full_texts), self.batch_size):
                encoded = self.tokenizer(
                    full_texts[start : start + self.batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    add_special_tokens=False,
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
