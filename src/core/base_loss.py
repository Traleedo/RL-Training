"""损失项抽象与加权组合器 —— 框架的核心解耦点。

设计要点
--------
用户的核心动机是「加一项自定义损失只需要按组件的方式添加进去」。
所以损失不是「一个算法一个大类」，而是一串可插拔的**项**：

.. code-block:: yaml

    losses:
      - {type: ppo_clip,      weight: 1.0,  clip_eps: 0.2}
      - {type: kl_penalty,    weight: 0.001}
      - {type: value_loss,    weight: 0.5,  clip_eps: 0.2}
      - {type: entropy_bonus, weight: 0.01}
      - {type: my_custom,     weight: 0.3}    # 你自己加的那一项

``total = Σ wᵢ · Lᵢ``。加权和**不在** ``LossTerm`` 里，由 ``LossComposition`` 统一
负责 —— 这样每一项都只关心自己怎么算，不需要知道别人的存在。

**自定义损失的三步 SOP**

1. 新建 ``src/components/losses/my_custom.py``，继承 ``LossTerm``，
   用 ``@register("loss", "my_custom")`` 装饰；
2. 在 ``src/components/losses/__init__.py`` 里 import 它（触发注册）；
3. YAML 里加一行。

trainer 一行都不用改。这就是整个框架想达到的效果。

``LossTerm.compute`` 是纯函数
------------------------------
不允许调模型、不允许 backward、不允许碰 optimizer。原因不只是洁癖：如果 term
自己调 actor，它就要自己决定 ``no_grad`` 与否、要不要 ``zero_grad``、和 trainer 的
``clip_grad_norm_`` 怎么配合 —— 这些是 trainer 的职责，散出去必然出现
「某个 term 多调了一次前向导致显存爆掉」或「梯度被意外的 zero_grad 清掉」。

但纯函数不够用。确实存在需要额外前向的合法需求（双前向 KL 估计、需要 reward model
打分的 term）。给这些情况留了 ``Component.prepare(batch, ctx)`` 钩子，
参见 ``core.component.ForwardContext``。

梯度从哪来
----------
``grad_fields`` 声明「我依赖的字段里哪些必须是可微的」。合法值只有
``curr_logprobs`` 与 ``values`` —— 因为只有这两个字段在训练相位由 actor / critic
产生（相位规则见 ``core.interfaces``）。trainer 启动时会静态校验这一点。

这条声明的必要性：``requires`` 只表达了「字段存在」，没表达「字段可微」。
如果某个 term 误把 detached 的 ``rollout_logprobs`` 当可微输入用，``backward()``
要么直接报错，要么（更糟）让 ratio 恒等于 1、策略梯度项恒为 0，
只剩 KL 项在下降，loss 曲线看起来完全正常。
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, Sequence

import torch

from core import interfaces as F
from core.batch import Batch
from core.component import Component

__all__ = ["LossTerm", "LossComposition"]


class LossTerm(Component):
    """一项损失。"""

    category = "loss"

    requires = frozenset({F.CURR_LOGPROBS, F.RESPONSE_MASK})

    #: 大多数 loss term 只产出标量，不往 batch 里写字段
    provides = frozenset()

    #: 本项依赖的字段里，哪些必须是可微的。合法取值只有
    #: ``{curr_logprobs, values}``（见 ``core.interfaces.TRAIN_PHASE_FIELDS``）。
    grad_fields: ClassVar[frozenset[str]] = frozenset({F.CURR_LOGPROBS})

    #: 这一项是否要求「同一组的所有行落在同一个 mini-batch 里」。
    #:
    #: 默认 False —— 逐 token 的损失（策略梯度、交叉熵）对行序无所谓。
    #: DPO 是 True：它的损失是 ``−log σ(β(Δ_w − Δ_l))``，一个对必须在同一次
    #: 前向里，切散了要么报错、要么算出一个看似合理的错值。
    #:
    #: 做成**组件声明**而不是配置开关，与 ``RewardProcessor.changes_batch_size``
    #: 同一套做法：写这个损失的人知道它需要什么，配 YAML 的人不一定。
    #: trainer 读它来决定用 ``split`` 还是 ``split_by_group``。
    needs_intact_groups: ClassVar[bool] = False

    @abstractmethod
    def compute(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        """返回 ``(标量损失, 指标字典)``。

        约定：

        - 损失必须是 **0 维**、可微、且已经归约到 ``response_mask`` 上。
          不要返回逐 token 的张量 —— ``LossComposition`` 只做加权求和，
          不做归约（它不知道该用哪个 mask）。
        - 指标字典里的值会被打成 ``loss/{名字}/{指标名}`` 记录下来。
          推荐放一些「不记下来就永远不知道自己在优化什么」的量，
          例如裁剪比例、平均 ratio、KL 的均值和最大值。
        - 纯函数：不调模型、不 backward、不碰 optimizer。
        """
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        """日志里的名字，会作为指标前缀。默认实现建议返回类名的小写下划线形式。"""
        raise NotImplementedError

    def weight_hint(self) -> float | None:
        """给使用者看的推荐权重，仅用于文档/自省，不影响计算。"""
        return None


class LossComposition:
    """把若干 ``(LossTerm, weight)`` 加权求和。

    这不是一个 Component，也不注册 —— 它是纯粹的算术组合器。
    之所以独立成类而不是塞进 trainer：让「怎么组合」和「怎么训练」两件事分开，
    以后想换成别的组合策略（例如自适应权重、不确定性加权）只改这一个类。
    """

    def __init__(self, terms: Sequence[tuple[LossTerm, float]]) -> None:
        self.terms: list[tuple[LossTerm, float]] = list(terms)

    def __len__(self) -> int:
        return len(self.terms)

    def needed(self) -> frozenset[str]:
        """整个组合的依赖并集。

        注意调用方（``engine.assembly``）应该**只把权重非零的项传进来** ——
        零权重项在 assembly 阶段就被剔除了，所以 YAML 里留着一行
        ``weight: 0.0`` 不会白白把 Reference 模型拉起来。
        """
        fields: set[str] = set()
        for term, _ in self.terms:
            fields |= term.needed
        return frozenset(fields)

    def grad_fields(self) -> frozenset[str]:
        fields: set[str] = set()
        for term, _ in self.terms:
            fields |= term.grad_fields
        return frozenset(fields)

    def __call__(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        """算出总损失与全部指标。

        权重为 0 的项会被跳过（连带跳过它的 ``prepare`` 需求）。
        """
        mask = batch[F.RESPONSE_MASK]
        total = torch.zeros((), device=mask.device, dtype=torch.float32)
        metrics: dict[str, float] = {}

        for term, weight in self.terms:
            if weight == 0.0:
                continue
            loss, term_metrics = term.compute(batch)
            if loss.dim() != 0:
                raise ValueError(
                    f"loss term {term.name()!r} 返回了 {loss.dim()} 维张量，"
                    f"必须归约成 0 维标量。"
                )
            total = total + weight * loss
            prefix = f"loss/{term.name()}"
            metrics[f"{prefix}"] = float(loss.detach())
            metrics[f"{prefix}/weighted"] = float((weight * loss).detach())
            for key, value in term_metrics.items():
                # 组件自报的指标可能是带梯度的张量（它们本来就是从可微量算出来的）。
                # 这里统一 detach —— 否则 float(带梯度张量) 会触发 UserWarning，
                # 而写 loss term 的人不该被这种细节绊住。
                if isinstance(value, torch.Tensor):
                    value = value.detach()
                metrics[f"{prefix}/{key}"] = float(value)

        if not torch.isfinite(total):
            raise FloatingPointError(
                f"总损失不是有限值（{float(total)}）。常见原因：\n"
                f"  - advantage 的方差为 0 导致除零（检查归一化是否做了两次）\n"
                f"  - ref_logprobs 没被加载（值为 None 或 0）导致 KL 异常\n"
                f"  - 学习率过大导致数值爆炸\n"
                f"各项明细：{ {t.name(): w for t, w in self.terms} }"
            )
        return total, metrics

    def __repr__(self) -> str:
        items = ", ".join(f"{t.name()}×{w}" for t, w in self.terms)
        return f"<LossComposition [{items}]>"
