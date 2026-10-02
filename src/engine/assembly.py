from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from core import interfaces as F
from core.base_advantage import AdvantageEstimator
from core.base_controller import Controller, check_target, resolve_target
from core.base_loss import LossTerm
from core.base_processor import RewardProcessor
from core.component import Component

__all__ = ["AssemblyPlan", "plan_assembly"]
_FRAMEWORK_PROVIDERS_BY_PHASE: dict[str, dict[str, str]] = {
    F.PHASE_DATASET: {f: "dataset" for f in F.DATASET_OUTPUT_FIELDS},
    F.PHASE_ROLLOUT: {
        **{f: "rollout" for f in F.ROLLOUT_OUTPUT_FIELDS},
        F.PROMPT_TEXTS: "rollout",
        F.RESPONSE_TEXTS: "rollout",
        F.PROMPT_LENGTHS: "rollout",
        F.ROLLOUT_VALUES: "critic",
    },
    F.PHASE_SCORE: {F.REWARDS: "scorer"},
    F.PHASE_ADVANTAGE: {F.ADVANTAGES: "advantage", F.RETURNS: "advantage"},
    F.PHASE_PREPARE: {F.REF_LOGPROBS: "reference"},
    F.PHASE_TRAIN: {F.CURR_LOGPROBS: "actor", F.VALUES: "critic"},
}


def _framework_providers(phases: Iterable[str]) -> dict[str, str]:
    unknown = sorted(set(phases) - set(_FRAMEWORK_PROVIDERS_BY_PHASE))
    if unknown:
        raise ValueError(
            f"未知的相位名 {unknown}；合法取值："
            f"{sorted(_FRAMEWORK_PROVIDERS_BY_PHASE)}"
        )
    out: dict[str, str] = {}
    # 按固定顺序遍历，让 providers 的内容与相位集合的传入顺序无关
    for phase, fields in _FRAMEWORK_PROVIDERS_BY_PHASE.items():
        if phase in phases:
            out.update(fields)
    return out


def _label(component: Component) -> str:
    """给组件起一个好认的短名字，用于报错与启动日志。"""
    key = getattr(type(component), "_registry_key", None)
    if key:
        return f"{key[0]}/{key[1]}"
    return type(component).__name__


@dataclass
class AssemblyPlan:
    """装配计划。由 ``plan_assembly`` 产出，trainer 照此构建模型。"""

    #: 所有存活组件声明的依赖并集
    needed: frozenset[str] = frozenset()

    #: 需要构建哪些模型
    need_critic: bool = False
    need_reference: bool = False

    #: 字段 -> 哪些组件声明了需要它。用于启动日志与报错定位。
    consumers: dict[str, list[str]] = field(default_factory=dict)

    #: 字段 -> 谁能提供它
    providers: dict[str, list[str]] = field(default_factory=dict)

    #: 被需要但无人能提供的字段 -> 需求方列表。这是启动即失败的主要依据。
    unknown: dict[str, list[str]] = field(default_factory=dict)

    #: 模型名 -> 导致它被构建的组件列表（可观测性）
    model_reasons: dict[str, list[str]] = field(default_factory=dict)

    #: 静态检查发现的问题（人类可读）
    conflicts: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def ok(self) -> bool:
        return not self.unknown and not self.conflicts

    def raise_if_invalid(self) -> None:
        """有问题就抛错，把所有问题一次性列全（而不是修一个报一个）。"""
        if self.ok:
            return
        lines: list[str] = []
        if self.unknown:
            lines.append("以下字段被组件需要，但没有任何东西能提供它：")
            for field_name, who in sorted(self.unknown.items()):
                lines.append(f"  - {field_name!r}  被 {who} 需要")
            lines.append(
                "  常见原因：字段名拼错（例如 'reward' 应为 'rewards'），"
                "或者组件忘了在 provides / requires 里声明它。"
            )
        if self.conflicts:
            lines.append("以下配置冲突会导致训练静默失效：")
            for item in self.conflicts:
                lines.append(f"  - {item}")
        raise ValueError("装配失败：\n" + "\n".join(lines))

    def describe(self) -> str:
        """多行摘要，启动时打印。"""
        models = []
        if True:  # actor 总是存在
            models.append("actor")
        if self.need_critic:
            models.append("critic")
        if self.need_reference:
            models.append("reference")

        lines = ["装配计划：", f"  将构建的模型：{models}"]
        for model in models:
            reasons = self.model_reasons.get(model)
            if reasons:
                lines.append(f"    {model} <- {reasons}")
        lines.append(f"  依赖字段并集：{sorted(self.needed)}")
        if self.need_reference is False:
            lines.append("    （没有组件需要 ref_logprobs，Reference 不会加载）")
        if self.need_critic is False:
            lines.append("    （没有组件需要 values，Critic 不会构建）")
        return "\n".join(lines)


def plan_assembly(
    *,
    processors: Sequence[RewardProcessor] = (),
    advantage: AdvantageEstimator | None = None,
    loss_terms: Sequence[tuple[LossTerm, float]],
    extra_components: Sequence[Component] = (),
    phases: Iterable[str] = F.RL_PHASES,
) -> AssemblyPlan:
    logic_components: list[Component] = [
        *processors, *([] if advantage is None else [advantage]), *extra_components
    ]
    all_components: list[Component] = [*logic_components, *(t for t, _ in loss_terms)]

    # ---- 依赖并集 + 需求方索引 ----
    consumers: dict[str, list[str]] = defaultdict(list)
    for component in all_components:
        label = _label(component)
        for field_name in component.needed:
            consumers[field_name].append(label)
    needed = frozenset(consumers)

    # ---- 提供方索引 ----
    providers: dict[str, list[str]] = defaultdict(list)
    for field_name, who in _framework_providers(phases).items():
        providers[field_name].append(who)
    for component in all_components:
        label = _label(component)
        for field_name in component.provides:
            providers[field_name].append(label)

    unknown = {
        field_name: who
        for field_name, who in consumers.items()
        if field_name not in providers
    }
    model_fields = needed - F.NON_MODEL_FIELDS
    models: set[str] = set()
    model_reasons: dict[str, list[str]] = defaultdict(list)
    for field_name in model_fields:
        model = F.MODEL_FOR_FIELD.get(field_name)
        if model is not None:
            models.add(model)
            model_reasons[model].extend(consumers[field_name])

    # ---- 冲突检查 ----
    conflicts: list[str] = []
    normalizing = [_label(p) for p in processors if p.normalizes_rewards]
    if normalizing and getattr(advantage, "assumes_normalized_rewards", False):
        conflicts.append(
            f"归一化冲突：processor {normalizing} 会对奖励做归一化，"
        )

    # 冲突二：grad_fields 声明了永远不会可微的字段
    for term, _ in loss_terms:
        label = _label(term)
        bad = set(term.grad_fields) - F.TRAIN_PHASE_FIELDS
        if bad:
            conflicts.append(
                f"{label} 的 grad_fields 声明了 {sorted(bad)}，"
            )
        # 冲突三：声明了 grad_fields 但根本没 requires 这个字段
        undeclared = set(term.grad_fields) - set(term.needed)
        if undeclared:
            conflicts.append(
                f"{label} 的 grad_fields 里有 {sorted(undeclared)}，"
                f"但它并没有在 requires / needed 里声明需要这些字段。"
            )

    # 冲突五：会改变样本数的 processor 必须排在会做统计的 processor 之前
    filters_at = [
        index
        for index, processor in enumerate(processors)
        if getattr(processor, "changes_batch_size", False)
    ]
    if filters_at:
        early_normalizers = [
            _label(processor)
            for processor in processors[: filters_at[0]]
            if processor.normalizes_rewards
        ]
        if early_normalizers:
            conflicts.append(
                f"processor 链顺序有问题：{early_normalizers} 会对奖励做统计，"
                f"但它们排在会改变样本数的 processor（"
                f"{_label(processors[filters_at[0]])}）之前。"
                f"被过滤掉的样本本来就不该算进基线和标准差，"
                f"而它们被算进去了 —— 不会报错，只会让指标「有点不对」。"
                f"请把会过滤样本的 processor 挪到链的最前面。"
            )
    controllers = [c for c in extra_components if isinstance(c, Controller)]
    if controllers:
        terms_by_name: dict[str, LossTerm] = {term.name(): term for term, _ in loss_terms}
        claimed: dict[str, str] = {}
        for controller in controllers:
            label = _label(controller)
            name, problems = resolve_target(controller, terms_by_name)
            conflicts.extend(f"控制器 {label}：{problem}" for problem in problems)
            if name is None:
                continue
            if name in claimed:
                # 两个控制器写同一个系数：后执行的覆盖先执行的，而执行顺序是
                # YAML 顺序 —— 从指标上看不出哪个赢了。
                conflicts.append(
                    f"控制器 {label} 与 {claimed[name]} 都绑定了损失项 {name!r}，"
                    f"它们会互相覆盖对方写入的系数（后写的赢），"
                    f"而哪个赢取决于 YAML 里的顺序。请让每个控制器绑定不同的项。"
                )
                continue
            claimed[name] = label
            conflicts.extend(
                f"控制器 {label}：{problem}"
                for problem in check_target(name, terms_by_name[name])
            )

    return AssemblyPlan(
        needed=needed,
        need_critic=F.MODEL_CRITIC in models,
        need_reference=F.MODEL_REFERENCE in models,
        consumers=dict(consumers),
        providers=dict(providers),
        unknown=unknown,
        model_reasons={k: sorted(set(v)) for k, v in model_reasons.items()},
        conflicts=conflicts,
    )
