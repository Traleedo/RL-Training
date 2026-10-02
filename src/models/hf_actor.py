"""基于 HuggingFace transformers 的策略模型封装。"""

from __future__ import annotations

from typing import Any, Iterator

import torch
from torch import nn

from core import interfaces as F
from core.base_model import Actor
from core.batch import Batch
from core.registry import register
from core.tensor_ops import gather_token_logprobs

from models.hf_common import load_causal_lm

__all__ = ["HFPolicyActor"]


@register("actor", "hf_causal_lm")
class HFPolicyActor(Actor):
    def __init__(
        self,
        model_config: Any,
        tokenizer: Any = None,
        model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        if model is None:
            model, tokenizer = load_causal_lm(model_config, tokenizer)
        self._model = model
        self.tokenizer = tokenizer

        self._base_hash: str | None = None
        declared = model_config.get("name_or_path") if hasattr(model_config, "get") else None
        if declared:
            self._base_hash = str(declared)

    @property
    def module(self) -> nn.Module:
        return self._model

    @property
    def base_model_hash(self) -> str | None:
        return self._base_hash

    def logprobs(self, batch: Batch) -> Batch:
        batch.require(F.INPUT_IDS, F.ATTENTION_MASK, who="HFPolicyActor")
        outputs = self._model(
            input_ids=batch[F.INPUT_IDS],
            attention_mask=batch[F.ATTENTION_MASK],
        )
        batch[F.CURR_LOGPROBS] = gather_token_logprobs(outputs.logits, batch[F.INPUT_IDS])
        return batch

    # ------------------------------------------------------------------
    def parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self._model.parameters() if p.requires_grad)

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        return (
            (name, p) for name, p in self._model.named_parameters() if p.requires_grad
        )

    def train(self, mode: bool = True) -> None:
        self._model.train(mode)

    def eval(self) -> None:
        self._model.eval()

    # ------------------------------------------------------------------
    def enable_lora(self, **lora_kwargs: Any):
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover - 依赖可选
            raise ImportError(
                "需要 peft 才能启用 LoRA：pip install peft"
            ) from exc

        lora_config = LoraConfig(task_type="CAUSAL_LM", **lora_kwargs)
        self._model = get_peft_model(self._model, lora_config)
        self._model.print_trainable_parameters()
        return self._model
