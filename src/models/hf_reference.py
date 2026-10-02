from __future__ import annotations

import logging
from typing import Any, Iterator

import torch
from torch import nn

from core import interfaces as F
from core.base_model import Reference
from core.batch import Batch
from core.registry import register
from core.tensor_ops import gather_token_logprobs

from models.hf_common import load_causal_lm

__all__ = ["HFFrozenReference"]

logger = logging.getLogger(__name__)


@register("reference", "hf_frozen")
class HFFrozenReference(Reference):
    """独立加载一份权重并冻结。"""

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

        for param in self._model.parameters():
            param.requires_grad_(False)
        self._model.eval()

    # ------------------------------------------------------------------
    @property
    def module(self) -> nn.Module:
        return self._model

    def logprobs(self, batch: Batch) -> Batch:
        """前向，写 **detached** 的 ``ref_logprobs``。"""
        batch.require(F.INPUT_IDS, F.ATTENTION_MASK, who="HFFrozenReference")
        with torch.no_grad():
            outputs = self._model(
                input_ids=batch[F.INPUT_IDS],
                attention_mask=batch[F.ATTENTION_MASK],
            )
            logprobs = gather_token_logprobs(outputs.logits, batch[F.INPUT_IDS])
        batch[F.REF_LOGPROBS] = logprobs.detach()
        return batch

    # ------------------------------------------------------------------
    def parameters(self) -> Iterator[nn.Parameter]:
        return iter(())          # 冻结模型没有可训练参数

    def train(self, mode: bool = True) -> None:
        self._model.eval()       # 永远保持 eval（dropout 关闭）

    def state_dict(self) -> dict[str, Any]:
        """参考模型的权重**不进 checkpoint**。

        它是由基座模型加载出来的冻结副本，可以从配置重新构造，
        存一份纯属浪费磁盘与时间。
        """
        return {}
