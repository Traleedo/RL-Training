
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from typing import Any, Iterable

import torch
from omegaconf import DictConfig, OmegaConf

from core import interfaces as F
from core.base_checkpoint import TrainerState
from core.base_loss import LossComposition
from core.batch import Batch
from core.component import ForwardContext
from core.registry import build

__all__ = ["Trainer"]

logger = logging.getLogger(__name__)


def ensure_components_registered() -> None:
    import components  # noqa: F401
    import data  # noqa: F401


class Trainer:

    def __init__(self, config: DictConfig) -> None:
        ensure_components_registered()

        self.cfg = config
        self.device = torch.device(config.trainer.get("device", "cpu"))
        self.seed = int(config.trainer.get("seed", 42))
        self.global_step = 0
        self.controllers: list[Any] = []

        self._set_seed(self.seed)

        # ==============================================================
        # 阶段 1：纯逻辑组件（便宜，不加载权重）
        # ==============================================================
        self._build_components()

        self.loggers = [
            build("logger", node) for node in config.get("loggers", []) or []
        ]
        self.checkpointer = build("checkpointer", config.checkpointer)

        # ==============================================================
        # 阶段 2 & 3：依赖并集 -> 模型清单 -> 构建模型
        # ==============================================================
        self._assemble()

        # ==============================================================
        # 阶段 4：优化器 / 调度器 / 组合器
        # ==============================================================
        self.optimizer = self._build_optimizer(config.optim)
        self.lr_scheduler = self._build_scheduler(config.optim)
        self._finalize()

        self._validate()

    # ==================================================================
    # 子类钩子 —— 装配
    # ==================================================================
    def _build_components(self) -> None:
        """构建便宜的纯逻辑组件（含 ``self.loss_terms``）。子类必须实现。"""
        raise NotImplementedError

    def _assemble(self) -> None:
        """求依赖并集、构建模型、设置 ``self.plan``。子类必须实现。"""
        raise NotImplementedError

    def _finalize(self) -> None:
        """装配的最后一步：组合器、前向上下文、释放集。默认只建组合器。"""
        self.composition = LossComposition(self.loss_terms)
        self._fwd_ctx = self._build_forward_context()

    def _build_forward_context(self) -> ForwardContext:
        """传给 ``Component.prepare`` 的上下文。默认只有 actor。"""
        return ForwardContext(self.actor, None, None, self.device)

    def _validate_domain(self) -> None:
        """子类自己的启动期校验。默认无。"""

    def _train_phase_writers(self) -> list[Any]:
        """在训练相位写字段的组件（用于校验「被需要的字段有人写」）。"""
        return [self.actor]

    def _optimizer_param_groups(self) -> list[dict[str, Any]]:
        """优化器的参数组。默认只有 actor。

        做成参数组而不是「一个 lr 走天下」，是因为 critic / 价值头通常需要
        比策略更大的步长。两者共用同一个优化器，所以 checkpoint 里只有一份
        优化器状态。
        """
        return [{"params": list(self.actor.parameters()), "lr": float(self.cfg.optim.lr)}]

    def _cfg_hash_payload(self) -> dict[str, Any]:
        """参与配置指纹的块。子类按需扩展。

        ⚠️ 改这个函数的返回值会让**全部已有 checkpoint 拒绝加载**。
        扩展可以，改动已有的键不行。
        """
        return {"model": self.cfg.model}

    def _extra_weights(self) -> dict[str, Any]:
        """actor 之外要进 checkpoint 的权重。默认无。"""
        return {}

    def _load_extra_weights(self, state: TrainerState) -> None:
        """恢复 ``_extra_weights`` 存下的权重。默认无。"""

    def _data_cursor(self) -> int | None:
        """数据加载游标。默认没有 —— RL 的数据由调用方喂进来。"""
        return None

    def _restore_data_cursor(self, state: TrainerState) -> None:
        """恢复数据加载游标。默认无。"""

    # ==================================================================
    # 子类钩子 —— 训练循环的差异点
    # ==================================================================
    def _split_for_train(self, batch: Batch, size: int, *, seed: int) -> list[Batch]:
        """把一个 batch 切成 mini-batch。默认按行切（会打断分组）。"""
        return batch.split(size, shuffle=True, seed=seed, drop_last=False)

    def _forward_trainable(self, batch: Batch) -> Batch:
        """mini-batch 上携带梯度的前向。默认只算 actor 的 logprob。"""
        return self.actor.logprobs(batch)

    def _trainable_parameters(self) -> list[Any]:
        """要交给 ``clip_grad_norm_`` 的参数。默认只有 actor 的。"""
        return list(self.actor.parameters())

    def _collect_microbatch_metrics(
        self, batch: Batch, collected: dict[str, list[float]], *, first: bool
    ) -> None:
        """每个 mini-batch 结束后的额外指标。默认无。"""

    # ==================================================================
    # 构建辅助
    # ==================================================================
    def _set_seed(self, seed: int) -> None:
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_optimizer(self, optim_cfg: DictConfig) -> torch.optim.Optimizer:
        weight_decay = float(optim_cfg.get("weight_decay", 0.0))
        return torch.optim.AdamW(self._optimizer_param_groups(), weight_decay=weight_decay)

    def _build_scheduler(self, optim_cfg: DictConfig):
        total = int(self.cfg.trainer.get("total_steps", 1000))
        kind = str(optim_cfg.get("lr_scheduler", "constant"))
        if kind == "constant":
            return torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1.0, total_iters=1
            )
        if kind == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total)
        if kind == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=total
            )
        raise ValueError(
            f"未知的 lr_scheduler {kind!r}；支持 constant / cosine / linear"
        )

    def _bind_controllers(self) -> None:
        """把每个控制器解析并锁到它的目标损失项上。

        绑定用 ``LossTerm.name()``（即日志里 ``loss/<名字>/...`` 的那个名字），
        不是 YAML 的 ``type``。理由见 ``core/base_controller.py``：用同一个标识符
        做绑定和读数，才能消掉「绑对了项、却读错了键」这类 bug。
        """
        if not self.controllers:
            return
        terms = {term.name(): term for term, _ in self.loss_terms}
        for controller in self.controllers:
            controller.bind(terms)
            logger.info("控制器已绑定：%s", controller.describe())

    def _validate(self) -> None:
        """启动即失败，而不是跑到一半才炸。"""
        # 训练相位字段：**被需要的**必须有人写。
        # 注意是「被需要的」而不是「全部」—— GRPO 配置下没有任何组件需要 values，
        # 那时没有 critic 是正确的，不该报错。
        writers = self._train_phase_writers()
        for field_name in F.TRAIN_PHASE_FIELDS:
            if field_name not in self.plan.needed:
                continue
            if not [c for c in writers if field_name in c.provides]:
                raise ValueError(
                    f"字段 {field_name!r} 被组件需要（需求方："
                    f"{self.plan.consumers.get(field_name, [])}），"
                    f"但没有任何模型提供它，训练相位没人会写这个字段。"
                )

        for lg in self.loggers:
            if not callable(getattr(lg, "log", None)):
                raise TypeError(f"logger {lg!r} 没有可调用的 log 方法")

        self._validate_domain()

    # ==================================================================
    # 训练循环骨架
    # ==================================================================
    def _train_epochs(
        self, batch: Batch, collected: dict[str, list[float]]
    ) -> None:
        """epoch × mini-batch 的可微循环。两个家族共用。

        ``collected`` 是就地累加的指标桶（每个 key 一个 list，之后取均值）。
        """
        epochs = int(self.cfg.algorithm.get("epochs", 1))
        mini_batch_size = int(self.cfg.algorithm.get("mini_batch_size", len(batch)))
        trainable = self._trainable_parameters()
        grad_clip = float(self.cfg.optim.get("grad_clip", 1.0))

        for epoch in range(epochs):
            minibatches = self._split_for_train(
                batch, mini_batch_size, seed=self.seed + epoch
            )
            for mb_index, mb in enumerate(minibatches):
                # clone 防止上一轮的 autograd 图残留在 batch 上
                mb = mb.clone()

                # F1. 本 mini-batch 的可微前向。每个 epoch 重算 —— 这是 PPO 的正确语义。
                # 注意这一句只写 CURR_LOGPROBS，不碰 ROLLOUT_LOGPROBS。
                mb = self._forward_trainable(mb)

                # F2. 需要额外前向的损失项的钩子（默认恒等）
                for term, _ in self.loss_terms:
                    mb = term.prepare(mb, self._fwd_ctx)

                # F3. 加权求和
                total_loss, term_metrics = self.composition(mb)
                if not total_loss.requires_grad:
                    raise RuntimeError(
                        "总损失不携带梯度。检查 loss term 的 grad_fields 声明，"
                        "以及是否有组件在训练相位错误地 detach 了张量。"
                    )

                # F4. 反向 + 更新
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                self.optimizer.step()
                self.lr_scheduler.step()

                # 指标
                collected["train/loss"].append(float(total_loss.detach()))
                collected["train/grad_norm"].append(float(grad_norm))
                for key, value in term_metrics.items():
                    collected[key].append(value)
                self._collect_microbatch_metrics(
                    mb, collected, first=(epoch == 0 and mb_index == 0)
                )

    @staticmethod
    def _aggregate(collected: dict[str, list[float]]) -> dict[str, float]:
        """把逐步累加的指标取均值。

        ``if v`` 同时过滤了空列表 —— 某个 mini-batch 没报的指标不会变成 NaN。
        """
        return {k: float(sum(v) / len(v)) for k, v in collected.items() if v}

    def _finish_step(self, aggregated: dict[str, float]) -> dict[str, float]:
        """相位 G + H：控制器回调、记录、推进步数、按需存盘。"""
        # 控制器回调。这个位置有三条约束，缺一条就会出错：
        #   * 在聚合**之后** —— 它要读本步的 loss/<name>/mean_kl；
        #   * 在 logger.log **之前** —— 它把新的 β 就地写回 aggregated，
        #     于是日志里这一行的 β 就是刚刚写进损失项的那个值；
        #   * 在 global_step += 1 **之前** —— 否则记到错误的 step 上。
        #
        # 注意控制器**不**走 metrics() 轮询：那个在回调之前，轮询到的是上一步的
        # 旧 β。上报一律通过就地修改 aggregated 完成。
        for controller in self.controllers:
            controller.on_train_step_end(aggregated)

        if self.loggers:
            for lg in self.loggers:
                lg.log(aggregated, step=self.global_step)

        # ================= H. 存盘（只在 step 边界）=================
        self.global_step += 1
        if (
            self.checkpointer.every_n_steps
            and self.global_step % self.checkpointer.every_n_steps == 0
        ):
            self.save_checkpoint()
        return aggregated

    # ==================================================================
    # Checkpoint
    # ==================================================================
    def _cfg_hash(self) -> str:
        """配置指纹。只取会改变训练语义的部分，避免 run_name 之类的无关改动
        让 resume 无谓失败。"""
        relevant = OmegaConf.to_yaml(
            OmegaConf.create(self._cfg_hash_payload()), resolve=True
        )
        return hashlib.sha1(relevant.encode("utf-8")).hexdigest()[:16]

    def state_dict(self) -> TrainerState:
        return TrainerState(
            step=self.global_step,
            actor=self.actor.state_dict(),
            optimizer=self.optimizer.state_dict(),
            lr_scheduler=self.lr_scheduler.state_dict(),
            rng=TrainerState.capture_rng(),
            cfg_hash=self._cfg_hash(),
            sampler_pos=self._data_cursor(),
            # 控制器的可调系数（自适应 β）。不存的话 resume 会把 β 弹回 YAML 初值，
            # 惩罚强度不连续，续跑的结果与不中断的训练不等价 —— 一种从曲线上
            # 看不出来的静默偏差。用 TrainerState 现成的 extra，零 schema 改动。
            extra={"controllers": [c.state_dict() for c in self.controllers]},
            **self._extra_weights(),
        )

    def save_checkpoint(self, path: str | None = None) -> str:
        if path is None:
            path = self.checkpointer.path_for_step(None, self.global_step)
        self.checkpointer.save(self.state_dict(), path)
        logger.info("已保存 checkpoint：%s", path)
        return path

    def resume(self, path: str) -> None:
        """恢复训练状态。

        配置指纹不符会直接报错 —— 从 grpo 换成 ppo 却加载旧 checkpoint，
        权重与配置对不上，会以极难排查的方式失效。
        """
        state = self.checkpointer.load(path)
        if state.cfg_hash and state.cfg_hash != self._cfg_hash():
            raise ValueError(
                f"checkpoint 的配置指纹 {state.cfg_hash!r} 与当前配置 "
                f"{self._cfg_hash()!r} 不一致，拒绝加载。\n"
                f"指纹覆盖的配置块见 Trainer._cfg_hash_payload。"
                f"如果你确实换了算法，请从新的模型权重开始训练，"
                f"而不是续跑一个语义不同的存档。"
            )
        self.actor.load_state_dict(state.actor or {})
        self._load_extra_weights(state)
        if state.optimizer:
            self.optimizer.load_state_dict(state.optimizer)
        if state.lr_scheduler:
            self.lr_scheduler.load_state_dict(state.lr_scheduler)
        self._restore_controllers(state)
        self._restore_data_cursor(state)
        TrainerState.restore_rng(state.rng)
        self.global_step = state.step
        logger.info("已从 %s 恢复到第 %d 步", path, self.global_step)

    def _restore_controllers(self, state: TrainerState) -> None:
        """恢复控制器系数（自适应 β）。

        数量对不上就直接报错，不用 ``zip`` 悄悄截断 —— zip 会在数量不一致时
        静默地少恢复一个，于是某个控制器的 β 悄悄留在初值上，而另一个恢复正确。
        那种错不会报任何异常，只会让两次训练的 KL 曲线对不上。
        """
        saved = (getattr(state, "extra", None) or {}).get("controllers", [])
        if not self.controllers and not saved:
            return
        if len(saved) != len(self.controllers):
            raise ValueError(
                f"checkpoint 里存了 {len(saved)} 个控制器的状态，"
                f"当前配置有 {len(self.controllers)} 个。拒绝猜测对应关系。"
            )
        for controller, item in zip(self.controllers, saved):
            controller.load_state_dict(item)

    def close(self) -> None:
        for lg in self.loggers:
            lg.close()
