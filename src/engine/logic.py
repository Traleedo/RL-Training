"""只构建**纯逻辑**组件 —— 一个模型都不加载。

为什么需要单独一层
------------------
装配计划（``AssemblyPlan``）是从存活组件的 ``needed`` 并集推出来的，而那些
组件 —— processor / advantage / loss / controller / dataset —— 全是纯逻辑，
构造它们不碰任何权重。

但 ``build_trainer()`` 会在 ``_assemble()`` 里**顺着计划把模型建出来**：
actor 一个、reference 一个、PPO 还要加一个 critic。这在微模型时代无所谓
（秒级），换成真实的 Qwen2.5 之后就变成了「只想看一眼计划，先下载 3GB」。

``scripts/plan.py`` 需要的恰恰是「计划」，不是「训练器」，所以这里把那一半
拆出来。它同时是 `--plan` 这类只读操作的唯一正确入口。

顺带解决另一件事：``RLTrainer`` 与 ``OfflineTrainer`` 的损失项循环、
控制器循环本来是逐字重复的两份。放进这里之后只有一份，不会漂移。
"""

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
    """把 ``algorithm.losses`` 变成 ``(项, 权重)`` 列表。

    ``weight == 0.0`` 的项在这里就被剔除，**早于**求依赖并集 —— 否则 YAML 里
    留一行 ``weight: 0.0`` 会白白把 Reference 拉起来。注意它与 ``coef: 0.0``
    是两种不同的关法：后者仍在组合里、每步仍被调用，只是释放掉某个字段。
    """
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
    """按 ``trainer.kind`` 构建逻辑组件并推出装配计划。不加载任何模型。

    RL 家族不构建 scorer：scorer 是要加载奖励模型的（8B），而它在
    ``plan_assembly`` 里根本不出场 —— 框架侧的
    ``_FRAMEWORK_PROVIDERS_BY_PHASE`` 已经声明「rewards 由 scorer 提供」，
    计划不需要看见 scorer 对象本身。
    """
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
