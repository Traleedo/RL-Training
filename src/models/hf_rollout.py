from __future__ import annotations

import logging
from typing import Any

import torch

from core import interfaces as F
from core.base_rollout import RolloutEngine
from core.batch import Batch
from core.registry import register
from core.tensor_ops import gather_token_logprobs

__all__ = ["HFRolloutEngine"]

logger = logging.getLogger(__name__)


@register("rollout", "hf")
class HFRolloutEngine(RolloutEngine):
    """用 HF 的 ``model.generate`` 采样，然后重新前向算 logprob。"""

    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
        num_generations: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        recompute_logprobs: bool = True,
    ) -> None:
        super().__init__(
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        self._model = model
        self.tokenizer = tokenizer
        self.do_sample = bool(do_sample)
        self.recompute_logprobs = bool(recompute_logprobs)

    # ------------------------------------------------------------------
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
        params = self.resolve(
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        num_generations = params["num_generations"]
        if not prompts:
            raise ValueError("generate 收到空的 prompt 列表")

        expanded = [p for p in prompts for _ in range(num_generations)]

        device = next(self._model.parameters()).device
        encoded = self.tokenizer(
            expanded,
            return_tensors="pt",
            padding=True,          # tokenizer.padding_side 已在加载时设为 left
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        prompt_width = input_ids.shape[1]

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            sequences = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=params["max_new_tokens"],
                do_sample=self.do_sample,
                temperature=params["temperature"],
                top_p=params["top_p"],
                pad_token_id=pad_token_id,
                **gen_kwargs,
            )

        generated = sequences[:, prompt_width:]           # [B, Lr]
        eos_id = self.tokenizer.eos_token_id
        prompt_ids: list[torch.Tensor] = []
        response_ids: list[torch.Tensor] = []
        response_texts: list[str] = []
        for i in range(generated.shape[0]):
            real_prompt = input_ids[i][attention_mask[i].bool()]
            row = generated[i]
            if eos_id is not None:
                hits = (row == eos_id).nonzero(as_tuple=False)
                if hits.numel():
                    row = row[: int(hits[0])]
            prompt_ids.append(real_prompt)
            response_ids.append(row)
            response_texts.append(self.tokenizer.decode(row, skip_special_tokens=True))
        batch = Batch.from_rollout(
            prompt_ids,
            response_ids,
            prompt_texts=expanded,
            response_texts=response_texts,
            num_generations=num_generations,
            pad_token_id=int(pad_token_id),
        )

        batch = self._fill_logprobs(batch)
        return batch

    # ------------------------------------------------------------------
    def _fill_logprobs(self, batch: Batch) -> Batch:
        """算行为策略的 logprob。
        ``recompute_logprobs=True``（默认）时对完整序列重新前向，这是推荐路径。
        """
        model = self._model
        if hasattr(model, "module"):        # DataParallel / DDP 包装
            model = model.module

        if not self.recompute_logprobs:
            raise NotImplementedError(
                "从 generate 的 scores 里取 logprob 的路径被有意不实现。\n"
            )

        with torch.no_grad():
            outputs = model(
                input_ids=batch[F.INPUT_IDS],
                attention_mask=batch[F.ATTENTION_MASK],
            )
            logprobs = gather_token_logprobs(outputs.logits, batch[F.INPUT_IDS])

        batch[F.ROLLOUT_LOGPROBS] = logprobs.detach()
        return batch
