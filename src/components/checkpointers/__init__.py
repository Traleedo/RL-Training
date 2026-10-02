"""Checkpoint 后端实现。

新增一个后端（S3、HuggingFace Hub、对象存储……）：继承
``core.base_checkpoint.Checkpointer``，实现 ``save`` / ``load`` / ``latest``，
加 ``@register("checkpointer", "<名字>")``，然后在本文件 import。

要点：Checkpointer **不直接摸模型**，它只搬运 ``TrainerState``。这样
「全参存盘」和「LoRA 只存 adapter」的差异只体现在 ``Actor.state_dict()``
一个点上，不用改任何后端。
"""

from __future__ import annotations

from components.checkpointers.local import LocalCheckpointer

__all__ = ["LocalCheckpointer"]
