"""Checkpoint 抽象。

真正的痛点不是「存不存得下」，而是**多对象的原子性**。一次断点续训要恢复：

- actor 权重、critic 权重（可能没有）
- actor 优化器状态、critic 优化器状态（可能没有）、lr scheduler 状态
- 四路随机数状态（torch / cuda / numpy / random）—— 少了它们，续跑出来的
  采样序列与不中断时不同，实验不可复现
- 数据加载游标
- 配置指纹

所以设计成「``TrainerState`` 一次打包、``Checkpointer`` 一个点替换」。
Checkpointer 不直接摸模型 —— 它只搬运 ``TrainerState``，由 trainer 负责把各组件
序列化进去。这样「全参存盘」和「LoRA 只存 adapter」的差异只体现在
``Actor.state_dict()`` 一个点上。

两条必须遵守的约束
------------------
1. **只能在 step 边界存。** ``train_step`` 内部（rollout 之后、``optimizer.step()``
   之前）是个半完成状态，存下来无法正确恢复。整个 ``train_step`` 是原子的。
2. **resume 时校验配置指纹。** 从 grpo 换到 ppo 却加载了旧 checkpoint，
   权重与配置对不上，会以极难排查的方式失效。``cfg_hash`` 不符直接报错。
"""

from __future__ import annotations

import random
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from core.component import Component

__all__ = ["TrainerState", "Checkpointer"]


@dataclass
class TrainerState:
    """一次断点续训需要的全部状态。"""

    step: int = 0

    #: 权重。LoRA 场景下只含 adapter + 基座指纹
    actor: dict[str, Any] | None = None
    critic: dict[str, Any] | None = None

    #: 优化器与调度器。注意 critic 可能根本没有优化器
    optimizer: dict[str, Any] | None = None
    critic_optimizer: dict[str, Any] | None = None
    lr_scheduler: dict[str, Any] | None = None

    #: 四路随机数状态，缺一不可复现
    rng: dict[str, Any] = field(default_factory=dict)

    #: 数据加载游标
    sampler_pos: int | None = None

    #: 配置指纹
    cfg_hash: str = ""

    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @staticmethod
    def capture_rng() -> dict[str, Any]:
        """抓取四路随机数状态。"""
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def restore_rng(state: dict[str, Any]) -> None:
        """恢复四路随机数状态。缺失的路径会被跳过。"""
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])


class Checkpointer(Component):
    """存/取 ``TrainerState``。

    ``save_dir`` 与 ``every_n_steps`` 是**公共参数**（由基类持有、trainer 读取），
    于是 YAML 里可以自然地把它们写在 ``checkpointer`` 节点下，

    .. code-block:: yaml

        checkpointer:
          type: local
          save_dir: ckpts/my_run
          every_n_steps: 50

    而不需要在 trainer 里把这些键从配置中剥掉再传给构造函数 ——
    「悄悄剥掉配置键」会让使用者对「哪个键到底属于谁」产生困惑。
    """

    category = "checkpointer"

    requires = frozenset()

    def __init__(self, save_dir: str = "checkpoints", every_n_steps: int = 0) -> None:
        super().__init__()
        self.save_dir = str(save_dir)
        #: 每多少步自动存一次；0 表示只手动存
        self.every_n_steps = int(every_n_steps)

    @abstractmethod
    def save(self, state: TrainerState, path: str) -> None:
        """把状态写到 ``path``。"""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> TrainerState:
        """从 ``path`` 读回状态。"""
        raise NotImplementedError

    @abstractmethod
    def latest(self, save_dir: str) -> str | None:
        """返回 ``save_dir`` 下最新的 checkpointer 路径，没有则返回 ``None``。

        用于 ``resume: latest`` 这种配置。
        """
        raise NotImplementedError

    def path_for_step(self, save_dir: str | None, step: int) -> str:
        """第 ``step`` 步的存档路径。子类可以覆盖以改变命名规则。"""
        base = (save_dir or self.save_dir).rstrip("/")
        return f"{base}/step_{step:08d}"
