"""训练引擎：装配规划 + 两个算法家族的训练主循环。"""

from __future__ import annotations

from engine.assembly import AssemblyPlan, plan_assembly
from engine.build import TRAINER_KINDS, build_trainer
from engine.offline_trainer import OfflineTrainer
from engine.rl_trainer import RLTrainer
from engine.trainer import Trainer

__all__ = [
    "Trainer",
    "RLTrainer",
    "OfflineTrainer",
    "build_trainer",
    "TRAINER_KINDS",
    "AssemblyPlan",
    "plan_assembly",
]
