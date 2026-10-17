from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from core import interfaces as F
from core.base_advantage import AdvantageEstimator
from core.base_loss import LossTerm
from core.registry import build
from engine.assembly import AssemblyPlan, plan_assembly

__all__ = [
    "LogicComponents",
    "build_loss_terms",
    "build_controllers",
    "build_logic_components",
    "plan_from_config",
]

logger = logging.getLogger(__name__)


def build_loss_terms(nodes: Sequence[Any]) -> list[tuple[LossTerm, float]]:
    terms: list[tuple[LossTerm, float]] = []
    skipped: list[str] = []
    for node in nodes:
        weight = float(node.get("weight", 1.0))
        if weight == 0.0:
            skipped.append(str(node.get("type", "?")))
            continue
        terms.append((build("loss", node), weight))
    if skipped:
        logger.info("跳过权重为 0 的损失项：%s", skipped)
    if not terms:
        raise ValueError("没有任何权重非零的损失项，无法训练。")
    return terms


def build_controllers(nodes: Sequence[Any]) -> list[Any]:
    return [build("controller", node) for node in nodes]


@dataclass
class LogicComponents:
    """一个配置里**不涉及模型**的那部分组件，外加据此推出的装配计划。"""

    plan: AssemblyPlan
    loss_terms: list[tuple[LossTerm, float]] = field(default_factory=list)
    controllers: list[Any] = field(default_factory=list)
    processors: list[Any] = field(default_factory=list)
    advantage: AdvantageEstimator | None = None
    #: 离线家族才有。注意它读的是磁盘上的 jsonl，所以它**不是**模型。
    dataset: Any | None = None


def build_logic_components(cfg: Any) -> LogicComponents:
    kind = str(cfg.get("trainer", {}).get("kind", "rl")).lower()
    loss_terms = build_loss_terms(cfg.algorithm.losses)
    controllers = build_controllers(cfg.algorithm.get("controllers", []) or [])

    if kind == "rl":
        processors = [
            build("processor", node) for node in cfg.reward.get("processors", []) or []
        ]
        advantage = build("advantage", cfg.algorithm.advantage)
        plan = plan_assembly(
            processors=processors,
            advantage=advantage,
            loss_terms=loss_terms,
            extra_components=controllers,
            phases=F.RL_PHASES,
        )
        return LogicComponents(
            plan=plan,
            loss_terms=loss_terms,
            controllers=controllers,
            processors=processors,
            advantage=advantage,
        )

    if kind == "offline":
        dataset = build("dataset", cfg.data)
        plan = plan_assembly(
            processors=(),
            advantage=None,
            loss_terms=loss_terms,
            extra_components=[*controllers, dataset],
            phases=F.OFFLINE_PHASES,
        )
        return LogicComponents(
            plan=plan,
            loss_terms=loss_terms,
            controllers=controllers,
            dataset=dataset,
        )

    raise ValueError(
        f"未知的 trainer.kind {kind!r}；支持 'rl'（样本来自策略采样）与 "
        f"'offline'（样本来自数据集）。"
    )


def plan_from_config(cfg: Any) -> AssemblyPlan:
    """只算装配计划 —— ``scripts/plan.py --all`` 用它，不会下载任何模型。"""
    return build_logic_components(cfg).plan
