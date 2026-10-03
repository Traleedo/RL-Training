from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch
from omegaconf import DictConfig

from core import interfaces as F
from core.batch import Batch
from core.component import ForwardContext
from core.registry import build
from engine.assembly import AssemblyPlan, plan_assembly
from engine.logic import build_controllers, build_loss_terms
from engine.trainer import Trainer

__all__ = ["OfflineTrainer"]

logger = logging.getLogger(__name__)


class OfflineTrainer(Trainer):
    """按配置装配组件，然后跑离线训练循环（SFT / DPO / RFT）。"""

    def __init__(self, config: DictConfig) -> None:
        # 数据游标：单调前进，从不取模回绕（一个 epoch 结束后继续读下去），
        # 且随 checkpoint 落盘 —— 它是离线家族唯一跨进程存活的状态。
        self._cursor = 0
        super().__init__(config)

    def _build_components(self) -> None:
        self.dataset = build("dataset", self.cfg.data)

        # 与 RLTrainer 共用 engine/logic.py 的实现，两份循环只有一份。
        self.loss_terms = build_loss_terms(self.cfg.algorithm.losses)
        self.controllers = build_controllers(
            self.cfg.algorithm.get("controllers", []) or []
        )

    def _assemble(self) -> None:
        self.plan: AssemblyPlan = plan_assembly(
            processors=(),
            advantage=None,
            loss_terms=self.loss_terms,
            extra_components=[*self.controllers, self.dataset],
            phases=F.OFFLINE_PHASES,
        )
        self.plan.raise_if_invalid()
        self._bind_controllers()
        logger.info("\n%s", self.plan.describe())
        logger.info("%s", self.dataset.describe())

        model_cfg = self.cfg.model
        self.actor = build("actor", self.cfg.actor, model_config=model_cfg)
        self.dataset.bind_tokenizer(getattr(self.actor, "tokenizer", None))

        if self.plan.need_reference:
            self.reference = build(
                "reference",
                self.cfg.reference,
                model_config=model_cfg,
                tokenizer=getattr(self.actor, "tokenizer", None),
            )
        else:
            self.reference = None

        # 离线家族没有优势相位，也就不需要价值函数。
        self.critic = None

    def _build_forward_context(self) -> ForwardContext:
        return ForwardContext(self.actor, None, self.reference, self.device)

    def _train_phase_writers(self) -> list[Any]:
        return [self.actor]

    def _needs_intact_groups(self) -> bool:
        return any(getattr(term, "needs_intact_groups", False) for term, _ in self.loss_terms)

    def _split_for_train(self, batch: Batch, size: int, *, seed: int) -> list[Batch]:
        if self._needs_intact_groups():
            return batch.split_by_group(
                size,
                shuffle=True,
                seed=seed,
                drop_last=False,
                group_size=self.dataset.group_size,
            )
        # 逐行的损失对行序无所谓 —— 按行打乱反而更好（每个 mini-batch 更同质）
        return batch.split(size, shuffle=True, seed=seed, drop_last=False)

    def _cfg_hash_payload(self) -> dict[str, Any]:
        return {
            "data": self.cfg.data,
            "algorithm": self.cfg.algorithm,
            "model": self.cfg.model,
        }

    def _validate_domain(self) -> None:
        if self.dataset.batch_size > len(self.dataset):
            raise ValueError(
                f"data.batch_size={self.dataset.batch_size} 超过了数据集大小 "
                f"{len(self.dataset)}（{self.cfg.data.get('type')}）。"
            )
        mini_batch = int(self.cfg.algorithm.get("mini_batch_size", 0))
        group_size = self.dataset.group_size
        if mini_batch and self._needs_intact_groups() and mini_batch % group_size != 0:
            raise ValueError(
                f"algorithm.mini_batch_size={mini_batch} 不是数据组大小 {group_size} "
                f"的整数倍，而当前损失项声明了 needs_intact_groups。\n"
                f"按组切分时这会切出半个组 —— 对成对样本（DPO）就是丢掉半个对。"
            )

    def _data_cursor(self) -> int | None:
        return self._cursor

    def _restore_data_cursor(self, state) -> None:
        self._cursor = int(state.sampler_pos or 0)

    def train_step(self) -> dict[str, float]:
        collected: dict[str, list[float]] = defaultdict(list)

        indices, self._cursor = self.dataset.next_batch_indices(
            self._cursor, self.dataset.batch_size
        )

        # ================= P. 准备 =================
        batch = self.dataset.build_batch(indices)
        batch.require(*F.DATASET_OUTPUT_FIELDS, who="train_step 在数据集之后")
        batch.meta[F.NUM_GENERATIONS] = self.dataset.group_size

        with torch.no_grad():
            if self.reference is not None:
                batch = self.reference.logprobs(batch)

        # 冻结：数据集提供的字段全程只读。教育意义大于防护意义 ——
        # 但「某个损失项顺手把 preference 改了」这种事，只有冻结能拦住。
        batch = batch.detach().freeze(F.FROZEN_BEFORE_TRAIN).to(self.device)

        # ================= T. 训练（与 RL 共用基类的循环）=================
        self._train_epochs(batch, collected)

        # ================= G. 汇总与记录 =================
        aggregated = self._aggregate(collected)
        aggregated.update(self._batch_metrics(batch))
        aggregated.update(self.dataset.metrics())
        for term, weight in self.loss_terms:
            aggregated[f"loss/{term.name()}/weight"] = float(weight)
        return self._finish_step(aggregated)

    def _batch_metrics(self, batch: Batch) -> dict[str, float]:
        out: dict[str, float] = {}
        if F.RESPONSE_MASK in batch.tensors:
            lengths = batch[F.RESPONSE_MASK].sum(dim=1).float()
            out["data/response_len_mean"] = float(lengths.mean())
            out["data/batch_rows"] = float(len(batch))
        return out
