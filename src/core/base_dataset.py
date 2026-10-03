from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar, Sequence

from core import interfaces as F
from core.batch import Batch
from core.component import Component

__all__ = ["BaseDataset"]


class BaseDataset(Component):
    """离线样本的来源。一个样本 **不一定** 是一行 —— DPO 的一对是两行。"""

    category = "dataset"

    #: 数据集不读 Batch（它是 Batch 的**起点**），所以 requires 恒为空。
    #: 这一点很重要：它意味着数据集永远不会是「某个模型被加载」的理由。
    requires: ClassVar[frozenset[str]] = frozenset()

    provides: ClassVar[frozenset[str]] = frozenset(F.DATASET_OUTPUT_FIELDS)

    def __init__(self) -> None:
        super().__init__()
        self._tokenizer: Any | None = None

    # ------------------------------------------------------------------
    # 分词语料注入
    # ------------------------------------------------------------------
    def bind_tokenizer(self, tokenizer: Any) -> None:
        """由 trainer 在模型构建之后注入。

        单独一个方法而不是构造参数，见模块 docstring。
        """
        self._tokenizer = tokenizer

    def _require_tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise RuntimeError(
                f"{type(self).__name__} 还没有绑定分词器。\n"
                f"数据集在 plan_assembly 之前构建（它要参与依赖推导），而 tokenizer "
                f"来自 actor，actor 在 plan_assembly 之后才加载 —— 所以必须由 trainer "
                f"在两者之间调用 bind_tokenizer()。\n"
                f"如果你是在手工组装，别忘了这一步。"
            )
        return self._tokenizer

    # ------------------------------------------------------------------
    # 数据集接口
    # ------------------------------------------------------------------
    @abstractmethod
    def __len__(self) -> int:
        """样本数。

        SFT 的一条样本是一行，DPO 的一条样本是**一对**（两行）。所以
        ``len()`` 不一定等于 ``build_batch`` 出来的 batch 维 —— 差一个
        ``group_size`` 因子。见 ``group_size``。
        """

    @property
    def group_size(self) -> int:
        """一个样本占几行。SFT = 1，成对的偏好数据 = 2。

        这个数字是 ``Batch.split_by_group`` 的依据：成对数据的 mini-batch
        必须整对地切，否则半个对算不出损失。
        """
        return 1

    @abstractmethod
    def build_batch(self, indices: Sequence[int]) -> Batch:
        """把这些**样本**下标变成对齐好的 Batch。"""

    # ------------------------------------------------------------------
    # 取样顺序（游标进 checkpoint）
    # ------------------------------------------------------------------
    def next_batch_indices(self, cursor: int, size: int) -> tuple[list[int], int]:
        """从游标处取一批样本下标，返回 ``(indices, 新游标)``。

        顺序由 ``self.order`` 决定（构造时按 seed 定下来），所以游标只要一个**整数**
        —— 它可以直接存进 ``TrainerState.sampler_pos`` 并在 resume 时接上。

        **游标单调递增，不取模。** 它的含义是「到目前为止一共消费了多少个样本」，
        而绕回是**取下标时**做的（``cursor % total``）。两者必须分开，否则
        「刚好走完一整圈」的存档会记成 ``sampler_pos = 0`` —— 与「从没训过」
        完全无法区分，而这正是 resume 场景下最需要分辨的一刻。

        用「跑到末尾时重新 shuffle」那种做法还会更糟：resume 之后的数据顺序与
        不中断的训练不同，而那是从 loss 曲线**看不出来**的偏差。
        """
        order = self.order
        total = len(order)
        if total == 0:
            raise ValueError(f"{type(self).__name__} 是空的，无法取样")
        if size < 1:
            raise ValueError(f"batch size 必须 >= 1，收到 {size}")
        if size > total:
            raise ValueError(
                f"data.batch_size={size} 超过了数据集大小 {total}。"
                f"这不会报错，只会让每个 batch 里出现重复样本 —— 所以在这里拦住。"
            )
        start = int(cursor) % total
        indices = [order[(start + i) % total] for i in range(size)]
        return indices, int(cursor) + size

    #: 取样顺序：``order[i]`` = 第 i 个被取到的样本下标。构造时定下来。
    order: list[int]

    def _init_order(self, *, shuffle: bool, seed: int) -> None:
        """按 seed 定下取样顺序。构造末尾调用。"""
        self.order = list(range(len(self)))
        if shuffle:
            import random

            random.Random(int(seed)).shuffle(self.order)

    def describe(self) -> str:
        return (
            f"dataset/{self._name()}  样本数={len(self)}  组大小={self.group_size}  "
            f"provides={sorted(self.provides)}"
        )

    def _name(self) -> str:
        key = getattr(type(self), "_registry_key", None)
        return key[1] if key else type(self).__name__
