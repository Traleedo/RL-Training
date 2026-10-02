"""基于 HuggingFace transformers 的价值模型封装。

两种 critic，差别只在**权重从哪来**：

    hf_value_head    独立 ``from_pretrained`` 一份自己的模型。
    hf_shared_value  从 actor 的底座 **复制** 一份，接一个 value head。

后者是 PPO 的标准做法：critic 从策略的初始权重出发（此时它对「哪些 token
重要」的判断已经比随机初始化好得多），之后两者独立演化。
"""

from __future__ import annotations

import copy
from typing import Any, Iterator

import torch
from torch import nn

from core import interfaces as F
from core.base_model import Critic
from core.batch import Batch
from core.registry import register

from models.hf_common import extract_transformer, find_hidden_size, load_value_model

__all__ = ["HFValueCritic", "HFSharedValueCritic"]


@register("critic", "hf_value_head")
class HFValueCritic(Critic):
    def __init__(
        self,
        model_config: Any,
        tokenizer: Any = None,
        model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        if model is None:
            model, tokenizer = load_value_model(model_config, tokenizer)
        self._model = model
        self.tokenizer = tokenizer

    @property
    def module(self) -> nn.Module:
        return self._model

    def forward_values(self, batch: Batch, *, detach: bool = False) -> Batch:
        batch.require(F.INPUT_IDS, F.ATTENTION_MASK, who="HFValueCritic")
        target = F.ROLLOUT_VALUES if detach else F.VALUES

        context = torch.no_grad() if detach else torch.enable_grad()
        with context:
            outputs = self._model(
                input_ids=batch[F.INPUT_IDS],
                attention_mask=batch[F.ATTENTION_MASK],
            )
            values = outputs.logits
            if values.dim() == 3:
                values = values.squeeze(-1)

        batch[target] = values.detach() if detach else values
        return batch

    def parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self._model.parameters() if p.requires_grad)

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        return (
            (name, p) for name, p in self._model.named_parameters() if p.requires_grad
        )

    def train(self, mode: bool = True) -> None:
        self._model.train(mode)

    def eval(self) -> None:
        self._model.eval()


@register("critic", "hf_shared_value")
class HFSharedValueCritic(Critic):
    """共享骨架的价值模型：``actor 的底座`` + ``nn.Linear(hidden, 1)``。

    必须 deepcopy，不能别名
    ----------------------
    如果这里写 ``self._base = backbone``（同一个对象），value loss 的梯度会
    从 critic 的地位直接流回策略的网络权重 —— 于是「策略用优势更新、critic
    用回报更新」这两套目标立刻纠缠在一起，而**不会报任何错**。loss 会下降，
    ratio 也正常，只是 PPO 早已不是 PPO 了。

    ``copy.deepcopy`` 之后，critic 从策略的**当前**权重出发，但两者各走各的。
    代价是一份额外的底座显存 —— 这是刻意的取舍。

    ``freeze_backbone``
    -------------------
    关掉骨架的梯度、只训 value head。省显存，但价值函数失去拟合复杂回报的
    能力，适合底座本来就很强的场景。默认 False（与 PPO 论文一致）。
    """

    #: 声明需要 actor 的底座 —— ``RLTrainer`` 据此在构建时注入 ``backbone=``。
    #:
    #: 它**不是** ``requires``：``requires`` 是 **Batch 字段**依赖，参与
    #: ``plan_assembly`` 的模型推导；这里要的是**组件对象**，属于构建期的事。
    #: 两者混在一起会让装配计划凭空多出一个不存在的字段依赖。
    needs_actor_backbone = True

    def __init__(
        self,
        model_config: Any = None,
        tokenizer: Any = None,
        backbone: nn.Module | None = None,
        freeze_backbone: bool = False,
        model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.tokenizer = tokenizer

        if model is not None:
            # 逃生口：直接给一个已经拼好的模块（测试 / 自定义组装用）。
            self._base = model
            self.value_head = getattr(model, "value_head", None)
            return

        if backbone is None:
            raise ValueError(
                f"{type(self).__name__} 需要 actor 的底座，但构建时没有拿到 "
                f"backbone=。\n"
                f"它由 RLTrainer 在装配时注入（见 needs_actor_backbone）。"
                f"如果你是手动构建的，请显式传 backbone=actor.module，"
                f"或改用独立加载的 critic: hf_value_head。"
            )

        config = getattr(backbone, "config", None)
        if config is None:
            raise ValueError(
                f"backbone（{type(backbone).__name__}）没有 .config，"
                f"无法确定 value head 的输入维度。"
            )
        hidden = find_hidden_size(config)

        # 复制，不是别名 —— 见类文档。deepcopy 会连 config 一起复制，
        # 这是无害的（config 是纯描述性的）。
        self._base = extract_transformer(copy.deepcopy(backbone))

        # head 必须跟骨架同一个 dtype：bf16 的骨架配 fp32 的 head 会在第一次
        # 前向时报 dtype mismatch，而那个报错指向的是 nn.Linear 而不是这里。
        dtype = next(self._base.parameters()).dtype
        self.value_head = nn.Linear(hidden, 1).to(dtype)

        if freeze_backbone:
            for param in self._base.parameters():
                param.requires_grad_(False)

    @property
    def module(self) -> nn.Module:
        return self._base

    def forward_values(self, batch: Batch, *, detach: bool = False) -> Batch:
        batch.require(F.INPUT_IDS, F.ATTENTION_MASK, who="HFSharedValueCritic")
        target = F.ROLLOUT_VALUES if detach else F.VALUES

        context = torch.no_grad() if detach else torch.enable_grad()
        with context:
            hidden = self._base(
                input_ids=batch[F.INPUT_IDS],
                attention_mask=batch[F.ATTENTION_MASK],
            ).last_hidden_state
            values = self.value_head(hidden).squeeze(-1)

        batch[target] = values.detach() if detach else values
        return batch

    def parameters(self) -> Iterator[nn.Parameter]:
        params = list(self._base.parameters())
        if self.value_head is not None:
            params += list(self.value_head.parameters())
        return (p for p in params if p.requires_grad)

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        merged = [(f"backbone.{n}", p) for n, p in self._base.named_parameters()]
        if self.value_head is not None:
            merged += [
                (f"value_head.{n}", p) for n, p in self.value_head.named_parameters()
            ]
        return ((n, p) for n, p in merged if p.requires_grad)

    def train(self, mode: bool = True) -> None:
        self._base.train(mode)

    def eval(self) -> None:
        self._base.eval()

    def state_dict(self) -> dict[str, Any]:
        """带 ``backbone.`` / ``value_head.`` 前缀。

        前缀不是为了好看：它让 critic 的键与 actor 的**必然不重叠**
        （actor 存的是 ``model.xxx`` / ``lm_head.xxx``）。两者共用同一个
        checkpoint 字典，键一旦撞上，后写的会静默覆盖先写的 ——
        而权重形状往往还对得上，于是错误直到 loss 不下降才被发现。
        """
        state = {
            f"backbone.{k}": v.detach().cpu()
            for k, v in self._base.state_dict().items()
        }
        if self.value_head is not None:
            state.update({
                f"value_head.{k}": v.detach().cpu()
                for k, v in self.value_head.state_dict().items()
            })
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        backbone = {
            k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")
        }
        head = {
            k[len("value_head."):]: v for k, v in state.items() if k.startswith("value_head.")
        }
        self._base.load_state_dict(backbone)
        if self.value_head is not None:
            self.value_head.load_state_dict(head)
