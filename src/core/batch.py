from __future__ import annotations

import warnings
from typing import Any, Iterable, Sequence

import torch

from core import interfaces as F
from core.tensor_ops import gather_token_logprobs

__all__ = ["Batch"]


class Batch:
    """统一批次容器。参见模块文档了解设计意图与方法语义。"""

    __slots__ = ("tensors", "non_tensors", "meta", "frozen")

    def __init__(
        self,
        tensors: dict[str, torch.Tensor] | None = None,
        non_tensors: dict[str, list[Any]] | None = None,
        meta: dict[str, Any] | None = None,
        frozen: Iterable[str] | None = None,
    ) -> None:
        self.tensors: dict[str, torch.Tensor] = dict(tensors or {})
        self.non_tensors: dict[str, list[Any]] = dict(non_tensors or {})
        self.meta: dict[str, Any] = dict(meta or {})
        self.frozen: set[str] = set(frozen or ())
        self._check_invariants()

    # ==================================================================
    # 不变量
    # ==================================================================
    def _check_invariants(self) -> None:
        """检查三条不变量。所有组件都可以依赖它们成立。"""
        # 类型检查放在取 shape 之前 —— 否则塞一个 list 进来会报
        # AttributeError: 'list' object has no attribute 'shape'，
        # 而真正的原因（放错了 dict）就被掩盖了。
        for key, value in self.tensors.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"tensors[{key!r}] 应为 torch.Tensor，收到 {type(value).__name__}。"
                    f"非张量数据请放进 non_tensors。"
                )

        lengths = {k: v.shape[0] for k, v in self.tensors.items()}
        for key, length in self.non_tensors.items():
            lengths[f"non_tensors[{key}]"] = len(length)
        distinct = set(lengths.values())
        if len(distinct) > 1:
            raise ValueError(
                f"Batch 内各字段的 batch 维长度不一致：{lengths}。"
                f"（tensors 与 non_tensors 必须等长）"
            )

    def _check_writable(self, key: str) -> None:
        if key in self.frozen:
            raise KeyError(
                f"字段 {key!r} 已冻结，禁止写入。\n"
                f"冻结发生在进入训练循环之前，用于保证 rollout 时刻的量"
                f"（尤其是 rollout_logprobs）不会被后续相位覆盖。\n"
                f"当前已冻结：{sorted(self.frozen)}"
            )

    # ==================================================================
    # 基本访问
    # ==================================================================
    def __len__(self) -> int:
        if self.tensors:
            return next(iter(self.tensors.values())).shape[0]
        if self.non_tensors:
            return len(next(iter(self.non_tensors.values())))
        return 0

    @property
    def batch_size(self) -> int:
        return len(self)

    def __getitem__(self, key: str) -> Any:
        if key in self.tensors:
            return self.tensors[key]
        if key in self.non_tensors:
            return self.non_tensors[key]
        if key in self.meta:
            return self.meta[key]
        raise KeyError(
            f"Batch 里没有字段 {key!r}。\n"
            f"tensors: {sorted(self.tensors)}\n"
            f"non_tensors: {sorted(self.non_tensors)}\n"
            f"meta: {sorted(self.meta)}"
        )

    def __setitem__(self, key: str, value: Any) -> None:
        self._check_writable(key)
        if isinstance(value, torch.Tensor):
            self.tensors[key] = value
        elif isinstance(value, (list, tuple)):
            self.non_tensors[key] = list(value)
        else:
            self.meta[key] = value
        self._check_invariants()

    def __contains__(self, key: str) -> bool:
        return key in self.tensors or key in self.non_tensors or key in self.meta

    def keys(self) -> set[str]:
        return set(self.tensors) | set(self.non_tensors) | set(self.meta)

    def require(self, *fields: str, who: str | None = None) -> "Batch":
        """断言这些字段存在，否则报错。组件入口处调用，让缺失依赖快速失败。"""
        missing = [f for f in fields if f not in self]
        if missing:
            prefix = f"{who} " if who else ""
            raise KeyError(
                f"{prefix}需要字段 {missing}，但 Batch 里没有。\n"
                f"现有 tensors: {sorted(self.tensors)}\n"
                f"现有 non_tensors: {sorted(self.non_tensors)}\n"
                f"你的组件应该在 requires 里声明这些字段，让 plan_assembly "
                f"提前检查（而不是跑到这里才炸）。"
            )
        return self

    def add_metric(self, key: str, value: float) -> None:
        """记录一个标量指标，供 logger 读取。"""
        self.meta.setdefault(F.METRICS, {})[key] = float(value)

    def get_metrics(self) -> dict[str, float]:
        return dict(self.meta.get(F.METRICS, {}))

    # ==================================================================
    # 行选择（一律返回拷贝，避免父子 batch 别名污染）
    # ==================================================================
    def select(self, idx: torch.Tensor | Sequence[int] | slice) -> "Batch":
        """按 dim0 选行。返回**拷贝**，因此改子 batch 不会影响父 batch。"""
        index = self._normalize_index(idx)
        tensors = {k: v.index_select(0, index.to(v.device)) for k, v in self.tensors.items()}
        non_tensors = {
            k: [v[i] for i in index.tolist()] for k, v in self.non_tensors.items()
        }
        return Batch(tensors, non_tensors, dict(self.meta), self.frozen)

    def _normalize_index(self, idx: torch.Tensor | Sequence[int] | slice) -> torch.Tensor:
        n = len(self)
        if isinstance(idx, slice):
            idx = list(range(*idx.indices(n)))
        if isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                if idx.shape[0] != n:
                    raise ValueError(
                        f"布尔索引长度 {idx.shape[0]} 与 batch 维 {n} 不一致"
                    )
                idx = idx.nonzero(as_tuple=False).flatten()
            return idx.detach().to("cpu", torch.long).flatten()
        return torch.as_tensor(list(idx), dtype=torch.long)

    def filter(self, mask: torch.Tensor) -> "Batch":
        """按布尔 mask（长度 = batch 维）筛选行。"""
        if mask.dtype != torch.bool:
            raise TypeError(f"filter 需要布尔 mask，收到 {mask.dtype}")
        return self.select(mask)

    def filter_groups(self, keep: torch.Tensor, group_size: int) -> "Batch":
        """按**组**筛选：``keep`` 是 ``[G]`` 的布尔张量，展开成逐样本 mask 再筛。
        DAPO 的动态采样（丢掉全对或全错的组）用这个。
        """
        n_groups = keep.shape[0]
        expected = n_groups * group_size
        if expected != len(self):
            raise ValueError(
                f"组数 {n_groups} × 组大小 {group_size} = {expected}，"
                f"与 batch 维 {len(self)} 不一致"
            )
        expanded = keep.to("cpu").repeat_interleave(group_size)
        return self.filter(expanded)

    # ==================================================================
    # 复制与拼接
    # ==================================================================
    def repeat_interleave(self, repeats: int, dim: int = 0) -> "Batch":
        """行 i 连续复制 ``repeats`` 次：``[a, b] -> [a, a, b, b]``。

        这正是 ``num_generations`` 的语义，也是组采样必须遵守的行序：
        组是连续块，于是 ``group_index = row_index // n``，
        ``group_by_prompt`` 退化成一次廉价切片。
        """
        if dim != 0:
            raise NotImplementedError("目前只支持沿 batch 维（dim=0）复制")
        if repeats < 1:
            raise ValueError(f"repeats 必须 >= 1，收到 {repeats}")
        tensors = {k: v.repeat_interleave(repeats, dim=0) for k, v in self.tensors.items()}
        non_tensors = {
            k: [item for item in v for _ in range(repeats)]
            for k, v in self.non_tensors.items()
        }
        return Batch(tensors, non_tensors, dict(self.meta), self.frozen)

    def repeat(self, n: int) -> "Batch":
        """整体平铺 ``n`` 次：``[a, b] -> [a, b, a, b]``。

        注意与 ``repeat_interleave`` 的区别 —— 这个会打乱分组，组采样要用前者。
        """
        if n < 1:
            raise ValueError(f"n 必须 >= 1，收到 {n}")
        tensors = {
            k: v.repeat(n, *([1] * (v.dim() - 1))) for k, v in self.tensors.items()
        }
        non_tensors = {k: v * n for k, v in self.non_tensors.items()}
        return Batch(tensors, non_tensors, dict(self.meta), self.frozen)

    @classmethod
    def concat(cls, batches: Sequence["Batch"], *, strict: bool = True) -> "Batch":
        """按行拼接多个 batch。

        ``strict=True`` 要求字段集合完全一致（DAPO 过滤后回并的场景）；
        ``False`` 时取字段交集，独有字段被丢弃并 warning。
        """
        batches = [b for b in batches if b is not None]
        if not batches:
            raise ValueError("concat 收到空的 batch 列表")
        if len(batches) == 1:
            return batches[0].clone()

        key_sets = [set(b.tensors) for b in batches]
        common = set.intersection(*key_sets)
        union = set.union(*key_sets)
        if strict and common != union:
            raise ValueError(
                f"concat(strict=True) 要求各 batch 字段一致，但存在独有字段："
                f"{sorted(union - common)}。"
            )
        if not strict and common != union:
            warnings.warn(
                f"concat(strict=False) 丢弃了非公共字段 {sorted(union - common)}",
                stacklevel=2,
            )

        tensors = {k: torch.cat([b.tensors[k] for b in batches], dim=0) for k in common}

        nt_keys = set.intersection(*(set(b.non_tensors) for b in batches))
        non_tensors = {
            k: [item for b in batches for item in b.non_tensors[k]] for k in nt_keys
        }

        meta = dict(batches[0].meta)
        frozen = set.intersection(*(set(b.frozen) for b in batches))
        out = cls(tensors, non_tensors, meta, frozen)
        out._check_invariants()
        return out

    # ==================================================================
    # 分组
    # ==================================================================
    def group_ids(self, group_size: int) -> torch.Tensor:
        """生成 ``[B]`` 的组编号：``arange(G).repeat_interleave(n)``。"""
        n = len(self)
        if n % group_size != 0:
            raise ValueError(
                f"batch 维 {n} 不能被组大小 {group_size} 整除，无法分组"
            )
        return torch.arange(n // group_size).repeat_interleave(group_size)

    def group_view(self, tensor: torch.Tensor, group_size: int) -> torch.Tensor:
        """把 ``[B, ...]`` reshape 成 ``[G, n, ...]``，用于组内 mean/std。

        在残缺的组上调用会直接抛错（快速失败），而不是给出错误的统计量。
        """
        if tensor.shape[0] != len(self):
            raise ValueError(
                f"张量首维 {tensor.shape[0]} 与 batch 维 {len(self)} 不一致"
            )
        if tensor.shape[0] % group_size != 0:
            raise ValueError(
                f"张量首维 {tensor.shape[0]} 不能被组大小 {group_size} 整除。\n"
                f"如果你是在 mini-batch 上做组内统计，那已经晚了 —— 组内统计必须在 "
                f"split() 之前完成（见 README 的「归一化归属」一节）。"
            )
        return tensor.reshape(-1, group_size, *tensor.shape[1:])

    def group_by_prompt(self, group_size: int | None = None) -> list["Batch"]:
        """切成 ``G`` 个子 batch，每个是一组同 prompt 的采样。

        优先走连续块快路径（rollout 契约保证的行序）；如果检测到 ``group_ids``
        不是连续块（例如某个 processor 打乱过顺序），退化为按 id 散射分组。
        """
        n = int(group_size if group_size is not None else self.meta.get(F.NUM_GENERATIONS, 1))
        if n < 1:
            raise ValueError(f"group_size 必须 >= 1，收到 {n}")
        total = len(self)
        if total == 0:
            return []
        if total % n != 0:
            raise ValueError(
                f"batch 维 {total} 不能被组大小 {n} 整除，无法分组"
            )
        n_groups = total // n

        if F.GROUP_IDS in self.tensors:
            gids = self.tensors[F.GROUP_IDS].detach().to("cpu")
            expected = torch.arange(n_groups).repeat_interleave(n)
            if gids.shape == expected.shape and torch.equal(gids, expected):
                # 快路径：组是连续块
                return [self.select(slice(g * n, (g + 1) * n)) for g in range(n_groups)]
            # 慢路径：group_ids 非连续，按 id 散射分组
            warnings.warn(
                "group_ids 不是连续块，group_by_prompt 退化为散射分组。"
                "这通常意味着某个组件打乱了行序。",
                stacklevel=2,
            )
            return [self.select((gids == g).nonzero(as_tuple=False).flatten())
                    for g in range(n_groups)]

        # 没有 group_ids：按 rollout 契约假定是连续块
        return [self.select(slice(g * n, (g + 1) * n)) for g in range(n_groups)]

    # ==================================================================
    # 切分
    # ==================================================================
    def split(
        self,
        size: int | Sequence[int],
        *,
        shuffle: bool = False,
        seed: int | None = None,
        drop_last: bool = False,
    ) -> list["Batch"]:
        """切成若干 mini-batch。

        ``size`` 是 int 时表示每块大小（最后一块可能更小，除非 ``drop_last``）；
        是序列时表示各块长度，总和必须等于 batch 维。

        ``shuffle`` 会**打断分组**。这本身无害（组内统计已在 advantage 相位算完），
        但如果 ``num_generations > 1`` 且不能整除，会打 warning —— 因为那时候
        有人在 mini-batch 上做组内统计的话会拿到残缺的组。
        """
        total = len(self)
        if total == 0:
            return []

        if isinstance(size, int):
            if size < 1:
                raise ValueError(f"mini-batch size 必须 >= 1，收到 {size}")
            if total % size != 0:
                n_gen = int(self.meta.get(F.NUM_GENERATIONS, 1))
                if n_gen > 1:
                    warnings.warn(
                        f"batch 维 {total} 不能被 mini_batch_size {size} 整除，"
                        f"分组会被切散（num_generations={n_gen}）。"
                        f"如果组内统计还没算，请先算完再 split。",
                        stacklevel=2,
                    )
            n_full = total // size
            sizes = [size] * n_full
            remainder = total - n_full * size
            if remainder and not drop_last:
                sizes.append(remainder)
        else:
            sizes = list(size)
            if sum(sizes) != total:
                raise ValueError(
                    f"切分长度之和 {sum(sizes)} 与 batch 维 {total} 不一致"
                )

        if shuffle:
            generator = torch.Generator().manual_seed(int(seed) if seed is not None else 0)
            order = torch.randperm(total, generator=generator)
            order = order[: sum(sizes)]
        else:
            order = torch.arange(total)

        out: list[Batch] = []
        cursor = 0
        for chunk_size in sizes:
            out.append(self.select(order[cursor: cursor + chunk_size]))
            cursor += chunk_size
        return out

    def split_by_group(
        self,
        size: int,
        *,
        shuffle: bool = False,
        seed: int | None = None,
        drop_last: bool = False,
        group_size: int | None = None,
    ) -> list["Batch"]:
        """切成若干 mini-batch，**保证同一组的所有行落在同一块里**。

        与 ``split`` 的唯一区别是打乱的粒度：这里打乱的是**组**，组内的行序
        （以及组本身是不是连续块）原样保留。

        为什么成对数据非它不可
        ----------------------
        DPO 的损失是 ``−log σ(β(Δ_w − Δ_l))`` —— 一个对必须在同一次前向里。
        ``split(shuffle=True)`` 按行打乱，会把一个对切到两个 mini-batch 里，
        那半个对算不出损失。而失败的方式有两种，都很糟：

        - 直接报错（``group_view`` 在残缺组上会抛错）—— 还算好的；
        - 更糟：如果实现成「切散了也照算」，损失会基于两个**不相干的**样本，
          梯度方向随机，而 loss 曲线照样好看。

        ``size % group_size != 0`` 直接报错而不是向下取整 —— 后者是「丢掉半个对」，
        静默的那种。
        """
        total = len(self)
        if total == 0:
            return []

        n = int(
            group_size if group_size is not None else self.meta.get(F.NUM_GENERATIONS, 1)
        )
        if n < 1:
            raise ValueError(f"group_size 必须 >= 1，收到 {n}")
        if total % n != 0:
            raise ValueError(
                f"batch 维 {total} 不能被组大小 {n} 整除，无法按组切分。"
                f"行序必须是完整的组（见 Batch.from_rollout 的契约）。"
            )
        if size < 1:
            raise ValueError(f"mini-batch size 必须 >= 1，收到 {size}")
        if size % n != 0:
            raise ValueError(
                f"mini_batch_size={size} 不是组大小 {n} 的整数倍。\n"
                f"按组切分时这会切出半个组 —— 对成对样本（DPO）而言就是丢掉半个对，"
                f"损失要么算不出来，要么算出一个**看似合理的错值**。\n"
                f"请把它设成 {n} 的整数倍"
                f"（当前批次共 {total} 行 = {total // n} 组）。"
            )

        n_groups = total // n
        group_order = torch.arange(n_groups)
        if shuffle:
            generator = torch.Generator().manual_seed(int(seed) if seed is not None else 0)
            group_order = group_order[torch.randperm(n_groups, generator=generator)]

        sizes = [size] * (total // size)
        remainder = total - (total // size) * size
        if remainder and not drop_last:
            # remainder 一定是 n 的整数倍：total 与 size 都是
            sizes.append(remainder)

        # 先按组重排，再展开成**行**序号。
        #
        # 注意 group_order 里存的是**组号**不是行号：组 g 的行是
        # ``g*n .. g*n+n-1``。直接 ``group_order.repeat_interleave(n)`` 会把组号
        # 当成行号用 —— 每组只取到它的第一行，而结果看上去仍然「每组 n 行、
        # 每组一个连续块」，所以不会报错，只会静默地训错数据。
        row_order = (
            group_order[:, None] * n + torch.arange(n)[None, :]
        ).reshape(-1)[: sum(sizes)]

        out: list[Batch] = []
        cursor = 0
        for chunk_size in sizes:
            out.append(self.select(row_order[cursor: cursor + chunk_size]))
            cursor += chunk_size
        return out

    def chunk(self, n: int) -> list["Batch"]:
        """均分成 ``n`` 块。不能整除则抛错。"""
        if len(self) % n != 0:
            raise ValueError(f"batch 维 {len(self)} 不能被 {n} 均分")
        return self.split(len(self) // n)

    # ==================================================================
    # 设备 / 类型 / 梯度（就地修改）
    # ==================================================================
    def to(self, device: torch.device | str, non_blocking: bool = False) -> "Batch":
        self.tensors = {
            k: v.to(device, non_blocking=non_blocking) for k, v in self.tensors.items()
        }
        return self

    def detach(self) -> "Batch":
        """就地 detach 全部张量，切断与已有计算图的联系。"""
        self.tensors = {k: v.detach() for k, v in self.tensors.items()}
        return self

    def cast(self, dtype: torch.dtype, keys: Iterable[str] | None = None) -> "Batch":
        targets = set(keys) if keys is not None else set(self.tensors)
        for k in list(self.tensors):
            if k in targets and self.tensors[k].is_floating_point():
                self.tensors[k] = self.tensors[k].to(dtype)
        return self

    def pin_memory(self) -> "Batch":
        self.tensors = {k: v.pin_memory() for k, v in self.tensors.items()}
        return self

    def clone(self) -> "Batch":
        """深拷贝张量。训练相位每进入一个 mini-batch 都要 clone 一次，
        避免上一轮的 autograd 图残留在 batch 上。"""
        return Batch(
            {k: v.clone() for k, v in self.tensors.items()},
            {k: list(v) for k, v in self.non_tensors.items()},
            dict(self.meta),
            set(self.frozen),
        )

    # ==================================================================
    # 写保护与释放（就地修改）
    # ==================================================================
    def freeze(self, keys: Iterable[str]) -> "Batch":
        """把这些字段设为只读。之后任何写入都抛 KeyError。

        在进入训练循环前对 ``FROZEN_BEFORE_TRAIN`` 调用，用于保证
        rollout 时刻的量不会被后续相位覆盖。
        """
        self.frozen.update(keys)
        return self

    def unfreeze(self, keys: Iterable[str]) -> "Batch":
        self.frozen.difference_update(keys)
        return self

    def drop(self, *keys: str) -> "Batch":
        """删除字段，释放显存。

        显存不是「以后再优化」的问题，是第一次跑就会遇到的问题：
        B=64 × n=8 × L=1024 时，光是中间张量就有十几个 GB。
        典型用法是在 advantage 算完后丢掉用后即弃的 ``rollout_values``。
        """
        for key in keys:
            self.tensors.pop(key, None)
            self.non_tensors.pop(key, None)
            self.meta.pop(key, None)
            self.frozen.discard(key)
        return self

    # ==================================================================
    # 构造
    # ==================================================================
    @classmethod
    def from_prompts(cls, prompts: Sequence[str], **meta: Any) -> "Batch":
        """从裸 prompt 字符串构造一个还没有张量的 batch。"""
        return cls(
            tensors={},
            non_tensors={F.PROMPT_TEXTS: [str(p) for p in prompts]},
            meta=dict(meta),
        )

    @classmethod
    def from_rollout(
        cls,
        prompt_ids: Sequence[torch.Tensor],
        response_ids: Sequence[torch.Tensor],
        *,
        prompt_texts: Sequence[str] | None = None,
        response_texts: Sequence[str] | None = None,
        num_generations: int = 1,
        pad_token_id: int = 0,
    ) -> "Batch":
        """把「变长的 prompt + 变长的生成结果」拼成对齐的批次。

        **这是全仓库唯一决定 response_mask 对齐方式的地方。**
        对齐规则（左 padding）：

        - prompt 左侧补 pad，使所有 prompt 右对齐 -> 生成从同一列继续
        - response 右侧补 pad，长度取本批最大值
        - ``attention_mask`` 在真实 token 上为 1，pad 上为 0
        - ``response_mask`` 只在生成的 token 上为 1，pad 上为 0

        因为 prompt 是左 padding 的，所有 response token 都在 ``i >= Lp`` 的位置，
        而 ``gather_token_logprobs`` 在位置 0 填的 0 永远不会落在 response 里 ——
        两个约定正好互相接上。

        调用方必须保证行序等价于 ``repeat_interleave(prompts, num_generations)``，
        组是连续块。``group_ids`` 会按这个契约写入。
        """
        n = len(prompt_ids)
        if len(response_ids) != n:
            raise ValueError(
                f"prompt 数 {n} 与 response 数 {len(response_ids)} 不一致"
            )
        if n == 0:
            raise ValueError("from_rollout 收到空批次")
        if num_generations < 1:
            raise ValueError(f"num_generations 必须 >= 1，收到 {num_generations}")
        if n % num_generations != 0:
            raise ValueError(
                f"行数 {n} 不能被 num_generations {num_generations} 整除；"
                f"rollout 必须输出完整的组。"
            )

        prompt_lens = [int(p.numel()) for p in prompt_ids]
        response_lens = [int(r.numel()) for r in response_ids]
        max_prompt = max(prompt_lens)
        max_response = max(response_lens)
        length = max_prompt + max_response

        device = prompt_ids[0].device
        input_ids = torch.full((n, length), pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((n, length), dtype=torch.long, device=device)
        response_mask = torch.zeros((n, length), dtype=torch.long, device=device)

        for i in range(n):
            lp, lr = prompt_lens[i], response_lens[i]
            prompt_start = max_prompt - lp
            response_start = max_prompt
            input_ids[i, prompt_start:response_start] = prompt_ids[i]
            if lr:
                input_ids[i, response_start:response_start + lr] = response_ids[i]
            attention_mask[i, prompt_start:response_start + lr] = 1
            response_mask[i, response_start:response_start + lr] = 1

        non_tensors: dict[str, list[Any]] = {
            F.PROMPT_LENGTHS: prompt_lens,
        }
        if prompt_texts is not None:
            non_tensors[F.PROMPT_TEXTS] = [str(t) for t in prompt_texts]
        if response_texts is not None:
            non_tensors[F.RESPONSE_TEXTS] = [str(t) for t in response_texts]

        group_ids = torch.arange(n // num_generations).repeat_interleave(num_generations)

        return cls(
            tensors={
                F.INPUT_IDS: input_ids,
                F.ATTENTION_MASK: attention_mask,
                F.RESPONSE_MASK: response_mask,
                F.GROUP_IDS: group_ids,
            },
            non_tensors=non_tensors,
            meta={F.NUM_GENERATIONS: num_generations},
        )

    # ==================================================================
    @staticmethod
    def token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """薄封装，指向全局唯一的 shift 实现。见 ``core.tensor_ops``。"""
        return gather_token_logprobs(logits, input_ids)

    def __repr__(self) -> str:
        tensor_shapes = {k: tuple(v.shape) for k, v in self.tensors.items()}
        return (
            f"<Batch size={len(self)} tensors={tensor_shapes} "
            f"non_tensors={sorted(self.non_tensors)} frozen={sorted(self.frozen)}>"
        )
