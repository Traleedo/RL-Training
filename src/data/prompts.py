"""RL 的 prompt 来源：本地文件，或从 HuggingFace / ModelScope 下载。

RL 与离线家族不一样：它没有 ``dataset`` 组件，trainer 的 ``train_step`` 只吃
一个 ``list[str]``。所以「数据集」在这里的形态是 **prompt 池 + 每步采样**，
而不是往 Batch 里塞张量。

下载顺序与模型一致（``models/hf_common.py``）：先 HuggingFace，失败再切
ModelScope。``datasets`` 是可选依赖 —— 只用 ``--prompts`` 本地文件时不需要它。

    from data.prompts import PromptSampler, load_prompt_dataset

    pool = load_prompt_dataset("ultrafeedback", limit=5000)
    sampler = PromptSampler(pool, per_step=8, seed=42)
    prompts = sampler.next()        # 每步 8 条，跑完一轮自动洗牌重来
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "DATASET_ALIASES",
    "PromptSampler",
    "load_prompt_dataset",
    "read_prompt_file",
]

logger = logging.getLogger(__name__)

#: 内置别名。自定义数据集直接传 HF repo id 即可。
#: ``repo`` / ``modelscope`` 允许不同（多数镜像站与 HF 同名，但不保证）。
DATASET_ALIASES: dict[str, dict[str, str]] = {
    "ultrafeedback": {
        "repo": "OpenBMB/UltraFeedback",
        "modelscope": "OpenBMB/UltraFeedback",
        "field": "instruction",
        "split": "train",
    },
}


# ----------------------------------------------------------------------
# 本地文件
# ----------------------------------------------------------------------
def read_prompt_file(path: str | Path) -> list[str]:
    """读每行一个 prompt 的文本文件。空行会被丢掉。"""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    prompts = [line.strip() for line in lines if line.strip()]
    if not prompts:
        raise ValueError(f"{path} 里没有非空行 —— prompt 列表是空的。")
    return prompts


# ----------------------------------------------------------------------
# 远程数据集
# ----------------------------------------------------------------------
def load_prompt_dataset(
    dataset: str,
    *,
    split: str | None = None,
    field: str | None = None,
    modelscope_id: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """下载一个 prompt 数据集并抽成一维字符串列表。

    ``dataset`` 可以是内置别名（见 :data:`DATASET_ALIASES`），也可以是任意
    HuggingFace dataset repo id。字段名 / split 不给时，别名用自带值，
    自定义数据集默认 ``instruction`` / ``train``。
    """
    alias = DATASET_ALIASES.get(dataset.lower())
    repo = alias["repo"] if alias else dataset
    split = split or (alias["split"] if alias else "train")
    field = field or (alias["field"] if alias else "instruction")
    ms_id = modelscope_id or (alias["modelscope"] if alias else repo)
    if limit is not None and limit < 1:
        raise ValueError(f"limit 必须 >= 1，收到 {limit}")

    logger.info("下载 prompt 数据集 %s（split=%s，字段=%s）", repo, split, field)

    hf_error: BaseException
    try:
        from datasets import load_dataset

        rows = load_dataset(repo, split=split)
    except Exception as exc:  # noqa: BLE001 — 网络/仓库/没装 datasets 都算「这个源不可用」
        hf_error = exc
        logger.warning("从 HuggingFace 下载数据集 %s 失败：%s", repo, exc)
    else:
        # 字段名写错应当在**这里**立刻报出来，而不是被当成下载失败再切到
        # 另一个源 —— 那样最终的报错会指向「网络问题」，与真正原因无关。
        prompts = _collect(rows, field, limit)
        if not prompts:
            raise RuntimeError(
                f"HuggingFace 数据集 {repo} 里字段 {field!r} 全为空。"
                f"用 --prompt-field 指定正确的字段名。"
            )
        return prompts

    try:
        from modelscope.msdatasets import MsDataset
    except ImportError:
        raise RuntimeError(
            f"从 HuggingFace 下载数据集 {repo} 失败，而 modelscope 未安装，无法 fallback。\n"
            f"  HF 失败原因：{hf_error}\n"
            f"  两条出路任选：pip install datasets（HuggingFace），"
            f"或 pip install modelscope（国内网络）。"
        ) from hf_error

    try:
        rows = MsDataset.load(ms_id, split=split)
    except Exception as ms_exc:  # noqa: BLE001
        raise RuntimeError(
            f"从 HuggingFace 与 modelscope 下载数据集 {repo} 都失败了。\n"
            f"  HuggingFace（{repo}）：{hf_error}\n"
            f"  modelscope（{ms_id}）：{ms_exc}\n"
            f"  字段名不对时用 --prompt-field 指定；split 不对时用 --dataset-split。"
        ) from ms_exc

    prompts = _collect(rows, field, limit)
    if not prompts:
        raise RuntimeError(
            f"modelscope 数据集 {ms_id} 里字段 {field!r} 全为空。"
            f"用 --prompt-field 指定正确的字段名。"
        )
    return prompts


def _collect(source: Any, field: str, limit: int | None) -> list[str]:
    """从 HF Dataset / ModelScope 数据集 / 普通列表里抽出 prompt。

    去重是必要的：UltraFeedback 里存在重复 instruction，重复 prompt 会让
    同一组样本的奖励高度相关，advantage 的方差被低估。
    """
    columns = getattr(source, "column_names", None)
    if columns and field in columns:
        values: Iterable[Any] = source[field]
    else:
        values = (_extract_field(row, field) for row in source)

    prompts: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _as_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        prompts.append(text)
        if limit is not None and len(prompts) >= limit:
            break
    return prompts


def _extract_field(row: Any, field: str) -> Any:
    """取字段；兼容 ModelScope 常见的 ``{"_key": ..., "_value": {...}}`` 包装。"""
    if isinstance(row, dict):
        if field in row:
            return row[field]
        for wrapper in ("_value", "data", "row"):
            inner = row.get(wrapper)
            if isinstance(inner, dict) and field in inner:
                return inner[field]
        keys = sorted(str(k) for k in row)[:12]
        raise KeyError(
            f"数据集里找不到字段 {field!r}；这一行的字段有 {keys}。"
            f"用 --prompt-field 指定正确的字段名。"
        )
    raise TypeError(f"数据集的行应当是 dict，收到 {type(row).__name__}")


def _as_text(value: Any) -> str:
    """把各种形状的字段值归一成字符串。

    ``instruction`` 是字符串；但同一个字段在别的数据集里可能是
    ``{"content": ...}`` 或多轮消息列表，这里一并兜住。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "content", "value", "instruction", "prompt", "question"):
            if key in value:
                return _as_text(value[key])
        return ""
    if isinstance(value, (list, tuple)):
        parts = [_as_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return str(value).strip()


# ----------------------------------------------------------------------
# 每步采样
# ----------------------------------------------------------------------
class PromptSampler:
    """按 ``per_step`` 从 prompt 池里取一批，取完一轮自动洗牌重来。

    为什么需要它：``train_step(prompts)`` 会把**整个列表**一次送进 rollout。
    直接传 6 万条下载来的 prompt 等于一个 step 生成 48 万条序列 —— 必然 OOM。
    所以「数据集」与「每步 batch」必须是两个量。
    """

    def __init__(
        self,
        prompts: list[str],
        *,
        per_step: int | None = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if not prompts:
            raise ValueError("prompt 池是空的")
        self.prompts = list(prompts)
        self.per_step = len(self.prompts) if per_step is None else int(per_step)
        if self.per_step < 1:
            raise ValueError(f"per_step 必须 >= 1，收到 {per_step}")
        self.shuffle = bool(shuffle)
        self._rng = random.Random(seed)
        self._order = list(range(len(self.prompts)))
        if self.shuffle:
            self._rng.shuffle(self._order)
        self._pos = 0

    def __len__(self) -> int:
        return len(self.prompts)

    def next(self) -> list[str]:
        out: list[str] = []
        while len(out) < self.per_step:
            if self._pos >= len(self._order):
                if self.shuffle:
                    self._rng.shuffle(self._order)
                self._pos = 0
            take = min(self.per_step - len(out), len(self._order) - self._pos)
            out.extend(
                self.prompts[index] for index in self._order[self._pos : self._pos + take]
            )
            self._pos += take
        return out

    def describe(self) -> str:
        return f"prompt 池 {len(self.prompts)} 条，每步采样 {self.per_step} 条"
