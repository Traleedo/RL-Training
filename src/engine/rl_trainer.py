from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from typing import Any

import torch
from omegaconf import OmegaConf

from core import interfaces as F
from core.batch import Batch
from core.component import ForwardContext
from core.registry import build, get
from core.tensor_ops import masked_mean
from engine.assembly import AssemblyPlan, plan_assembly
from engine.logic import build_controllers, build_loss_terms
from engine.trainer import Trainer

__all__ = ["RLTrainer"]

logger = logging.getLogger(__name__)


class RLTrainer(Trainer):
    """按配置装配组件，然后跑 RL 训练循环。"""

    def _build_components(self) -> None:
        scorer_cfg = self.cfg.reward.scorer
        scorer_kwargs: dict[str, Any] = {}
        if str(scorer_cfg.get("type")) == "hf_reward_model":
            reward_model_cfg = self.cfg.get("reward_model", None)
            if not reward_model_cfg or not reward_model_cfg.get("name_or_path"):
                raise ValueError(
                    f"reward.scorer.type 是 hf_reward_model，但配置里没有可用的 "    
                )
            scorer_kwargs["model_config"] = reward_model_cfg
        self.scorer = build("scorer", scorer_cfg, **scorer_kwargs)

        self.processors = [
            build("processor", node)
            for node in self.cfg.reward.get("processors", []) or []
        ]

        self.advantage = build("advantage", self.cfg.algorithm.advantage)

        self.loss_terms = build_loss_terms(self.cfg.algorithm.losses)
        self.controllers = build_controllers(
            self.cfg.algorithm.get("controllers", []) or []
        )

    def _assemble(self) -> None:
        self.plan: AssemblyPlan = plan_assembly(
            processors=self.processors,
            advantage=self.advantage,
            loss_terms=self.loss_terms,
            extra_components=self.controllers,
            phases=F.RL_PHASES,
        )
        self.plan.raise_if_invalid()
        self._bind_controllers()
        logger.info("\n%s", self.plan.describe())

        model_cfg = self.cfg.model

        # actor 模型只加载一次，rollout 与 actor 共享同一个对象。
        self.actor = build("actor", self.cfg.actor, model_config=model_cfg)
        actor_module = self.actor.module
        tokenizer = getattr(self.actor, "tokenizer", None)

        self.rollout = build(
            "rollout", self.cfg.rollout, model=actor_module, tokenizer=tokenizer
        )

        if self.plan.need_reference:
            # 默认加载**独立的冻结副本**。共享 actor 权重（关掉 LoRA adapter）
            self.reference = build(
                "reference", self.cfg.reference, model_config=model_cfg, tokenizer=tokenizer
            )
        else:
            self.reference = None

        if self.plan.need_critic:
            # 「共享骨架」的 critic（hf_shared_value）声明了 needs_actor_backbone，
            # 于是这里把 actor 的底座注进去。沿用已有的「组件声明 -> 框架推导」
            kwargs: dict[str, Any] = {}
            if getattr(get("critic", self.cfg.critic.type), "needs_actor_backbone", False):
                kwargs["backbone"] = actor_module
            self.critic = build(
                "critic",
                self.cfg.critic,
                model_config=model_cfg,
                tokenizer=tokenizer,
                **kwargs,
            )
        else:
            self.critic = None

    def _finalize(self) -> None:
        super()._finalize()

        # 优势相位之后可以释放的字段。省显存，且规则是自动推导的 ——
        # 「声明为 transient」减去「训练相位确实还需要」，所以 PPO 需要的
        # rollout_logprobs 不会被误删。
        loss_needs: set[str] = set()
        for term, _ in self.loss_terms:
            loss_needs |= term.needed
        transient: set[str] = set()
        for component in [*self.processors, self.advantage]:
            transient |= component.transient_requires
        self._releasable: set[str] = transient - loss_needs
        if self._releasable:
            logger.info("优势相位之后将释放字段以节省显存：%s", sorted(self._releasable))

    # ==================================================================
    # 基类钩子
    # ==================================================================
    def _build_forward_context(self) -> ForwardContext:
        return ForwardContext(self.actor, self.critic, self.reference, self.device)

    def _device_components(self) -> list[Any]:
        # scorer 是 RL 家族独有的第四个持有权重的组件（奖励模型，8B）。
        # 它不参与基类的 actor/critic/reference 三件套，得显式补上。
        return [*super()._device_components(), self.scorer]

    def _train_phase_writers(self) -> list[Any]:
        return [c for c in (self.actor, self.critic) if c]

    def _optimizer_param_groups(self) -> list[dict[str, Any]]:
        """actor 与 critic 用同一个优化器的不同参数组。

        critic 用单独的 ``critic_lr``（价值函数通常需要更大的学习率）。
        这样只有一个优化器状态要存进 checkpoint。
        """
        groups: list[dict[str, Any]] = [
            {"params": list(self.actor.parameters()), "lr": float(self.cfg.optim.lr)},
        ]
        if self.critic is not None:
            groups.append({
                "params": list(self.critic.parameters()),
                "lr": float(self.cfg.optim.get("critic_lr", self.cfg.optim.lr)),
            })
        return groups

    def _cfg_hash_payload(self) -> dict[str, Any]:
        """指纹覆盖 algorithm / reward / model 三块。

        """
        return {
            "algorithm": self.cfg.algorithm,
            "reward": OmegaConf.create(
                {"scorer": self.cfg.reward.scorer,
                 "processors": self.cfg.reward.get("processors", [])}
            ),
            "model": self.cfg.model,
        }

    def _extra_weights(self) -> dict[str, Any]:
        return {"critic": self.critic.state_dict() if self.critic is not None else None}

    def _load_extra_weights(self, state) -> None:
        if self.critic is not None:
            self.critic.load_state_dict(state.critic or {})

    def _forward_trainable(self, batch: Batch) -> Batch:
        batch = self.actor.logprobs(batch)
        if self.critic is not None:
            batch = self.critic.forward_values(batch, detach=False)
        return batch

    def _trainable_parameters(self) -> list[Any]:
        params = list(self.actor.parameters())
        if self.critic is not None:
            params += list(self.critic.parameters())
        return params

    def _collect_microbatch_metrics(
        self, batch: Batch, collected: dict[str, list[float]], *, first: bool
    ) -> None:
        self._logprob_metrics(batch, collected, first=first)

    def _validate_domain(self) -> None:
        # num_generations 与 mini_batch_size 的整除性
        num_gen = int(self.cfg.rollout.get("num_generations", 1))
        mini_batch = int(self.cfg.algorithm.get("mini_batch_size", 0))
        if num_gen > 1 and mini_batch and mini_batch % num_gen != 0:
            warnings.warn(
                f"mini_batch_size={mini_batch} 不是 num_generations={num_gen} 的整数倍，"
                f"mini-batch 内会包含残缺的组。如果组内统计在 split 之前已经算完，"
                f"这无害；否则请调整 mini_batch_size。",
                stacklevel=2,
            )

    # ==================================================================
    # 训练
    # ==================================================================
    def train_step(self, prompts: list[str]) -> dict[str, float]:
        """跑一个完整的训练步骤。

        整步是**原子**的：checkpoint 只能存在 step 与 step 之间，
        不能存在 ``train_step`` 内部（那是个半完成状态，无法正确恢复）。
        """
        num_generations = int(self.cfg.rollout.get("num_generations", 1))
        collected: dict[str, list[float]] = defaultdict(list)

        # ================= A. 生成 =================
        with torch.no_grad():
            batch = self.rollout.generate(
                prompts,
                num_generations=num_generations,
                max_new_tokens=int(self.cfg.rollout.get("max_new_tokens", 512)),
                temperature=float(self.cfg.rollout.get("temperature", 1.0)),
                top_p=float(self.cfg.rollout.get("top_p", 1.0)),
            )
        batch.meta[F.NUM_GENERATIONS] = num_generations
        batch.require(*F.ROLLOUT_OUTPUT_FIELDS, who="train_step 在 rollout 之后")

        # ================= B. 打分 -> 奖励处理链 =================
        raw_reward_stats: dict[str, float] = {}
        with torch.no_grad():
            batch = self.scorer.score(batch)
            # 在归一化**之前**记一份原始奖励。归一化后的均值恒为 0，
            # 只看它的话「奖励到底在什么量级」「有没有退化」完全看不出来。
            if F.REWARDS in batch.tensors:
                raw = batch[F.REWARDS]
                raw_reward_stats["reward/raw_mean"] = float(raw.mean())
                raw_reward_stats["reward/raw_std"] = (
                    float(raw.std()) if raw.numel() > 1 else 0.0
                )
            for processor in self.processors:
                batch = processor.process(batch)
        batch.require(F.REWARDS, who="train_step 在打分之后")

        if len(batch) == 0:
            logger.warning(
                "本步的批次为空（奖励处理链把全部样本都过滤掉了），跳过这一步，"
                "不推进 global_step。常见原因：DAPO 的动态采样丢掉了所有"
                "「组内奖励全同」的组 —— 通常说明这批 prompt 对该策略太容易或太难。"
                "调用方应当换一批 prompt 重新采样。"
            )
            skipped_metrics = {"train/skipped_step": 1.0}
            for lg in self.loggers:
                lg.log(skipped_metrics, step=self.global_step)
            return skipped_metrics

        # ================= C. 准备相位前向（只算一次，detached）=================
        with torch.no_grad():
            if self.reference is not None:
                batch = self.reference.logprobs(batch)
            if self.critic is not None and F.ROLLOUT_VALUES in self.plan.needed:
                batch = self.critic.forward_values(batch, detach=True)

        # ================= D. 优势 =================
        with torch.no_grad():
            batch = self.advantage.compute(batch)
        batch.require(F.ADVANTAGES, who="train_step 在优势之后")

        # 释放只有优势相位需要的字段（例如 GAE 的 rollout_values）
        if self._releasable:
            batch.drop(*sorted(self._releasable & set(batch.tensors)))

        # ================= E. 冻结 + 切梯度 =================
        # freeze 让 rollout_logprobs 等字段变成只读。这是保证 PPO 第 2 个 epoch
        # 仍能拿到 rollout 时刻 old_logprobs 的第二道防线（第一道是字段名分离）
        batch = batch.detach().freeze(F.FROZEN_BEFORE_TRAIN).to(self.device)

        # ================= F. epoch × mini-batch =================
        self._train_epochs(batch, collected)

        # ================= G. 汇总与记录 =================
        aggregated = self._aggregate(collected)
        aggregated.update(self._batch_metrics(batch))
        aggregated.update(raw_reward_stats)
        aggregated.update(self.advantage.metrics())
        for component in [*self.processors, self.scorer]:
            aggregated.update(component.metrics())

        return self._finish_step(aggregated)

    def _logprob_metrics(
        self, mb: Batch, collected: dict[str, list[float]], *, first: bool
    ) -> None:
        """记录 ratio 与 logprob 漂移。

        ``train/ratio_first`` 是整份设计里最有价值的诊断量之一：它应该精确等于 1.0。
        如果它明显偏离 1，说明 rollout 的 logprob 与训练前向算出的不是同一个东西
        ——最可能的原因是有人用了 ``generate`` 的 scores 而没有重新前向计算。
        """
        if F.CURR_LOGPROBS not in mb.tensors or F.ROLLOUT_LOGPROBS not in mb.tensors:
            return
        mask = mb[F.RESPONSE_MASK]
        delta = mb[F.CURR_LOGPROBS] - mb[F.ROLLOUT_LOGPROBS]
        ratio = torch.exp(delta)
        # detach：这些只是诊断量，不该挂在计算图上（float(带梯度张量) 也会 warning）
        collected["train/ratio_mean"].append(float(masked_mean(ratio, mask).detach()))
        collected["train/logprob_drift"].append(float(masked_mean(delta, mask).detach()))
        if first:
            collected["train/ratio_first"].append(float(masked_mean(ratio, mask).detach()))

    def _batch_metrics(self, batch: Batch) -> dict[str, float]:
        """从 rollout 批次里提取诊断量。"""
        out: dict[str, float] = {}
        if F.REWARDS in batch.tensors:
            rewards = batch[F.REWARDS]
            out["rollout/reward_mean"] = float(rewards.mean())
            out["rollout/reward_std"] = float(rewards.std()) if rewards.numel() > 1 else 0.0
        if F.RESPONSE_MASK in batch.tensors:
            lengths = batch[F.RESPONSE_MASK].sum(dim=1).float()
            out["rollout/response_len_mean"] = float(lengths.mean())
        if F.ADVANTAGES in batch.tensors and F.RESPONSE_MASK in batch.tensors:
            out["rollout/advantage_std"] = float(
                batch[F.ADVANTAGES][batch[F.RESPONSE_MASK].bool()].std()
            ) if batch[F.RESPONSE_MASK].sum() > 1 else 0.0
        return out
