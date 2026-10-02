"""JSONL 数据集 —— SFT 与偏好对比两个后端。

两种数据的差别只有**编码什么**与**提供什么**：

===================  =====================  ==============================
注册名               每行                              一条样本占几行
===================  =====================  ==============================
``jsonl_sft``        ``{prompt, response}``             1
``jsonl_preference`` ``{prompt, chosen, rejected}``     2（chosen + rejected）
===================  =====================  ==============================

其余（读文件、字段映射、长度过滤、取样顺序）完全共用。

对齐方式
--------
两个后端都调用 ``Batch.from_rollout`` —— 全仓库唯一决定 ``response_mask``
对齐方式的地方。**不写第二条 padding 路径**：离线侧的对齐一旦与 RL 侧差一列，
表现只是 loss 略微不同，查到最后都以为是别的原因。

关于模型对话模板
----------------
prompt 走 ``tokenizer.apply_chat_template(..., add_generation_prompt=True)``，
response 单独编码（``add_special_tokens=False``）。**chosen / rejected 必须与
actor / reference 用同一个模板** —— 模板不一致不会报错，只会让 π_ref 从一开始
就与 π_θ 对不上，于是 DPO 的 Δ 里混进一个常数偏移。
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch

from core import interfaces as F
from core.base_dataset import BaseDataset
from core.batch import Batch
from core.registry import register

__all__ = ["JSONLDataset", "JSONLSFTDataset", "JSONLPreferenceDataset"]

logger = logging.getLogger(__name__)


class JSONLDataset(BaseDataset):
    """读 JSONL、做字段映射与长度过滤。子类决定一条样本编码成什么。"""

    def __init__(
        self,
        path: str,
        *,
        prompt_field: str = "prompt",
        response_field: str = "response",
        chosen_field: str = "chosen",
        rejected_field: str = "rejected",
        system_prompt: str | None = None,
        max_prompt_tokens: int = 1024,
        max_response_tokens: int = 1024,
        min_response_tokens: int = 1,
        batch_size: int = 8,
        shuffle: bool = True,
        seed: int = 42,
        encoding: str = "utf-8",
    ) -> None:
        super().__init__()
        if int(batch_size) < 1:
            raise ValueError(f"batch_size 必须 >= 1，收到 {batch_size}")
        #: 一个 train_step 取几个**样本**（SFT 1 样本 = 1 行，偏好 1 样本 = 1 对）。
        #: 挂在数据集上而不是 trainer 上：多大的 batch 装得下，取决于数据集本身
        #: （10 行的玩具数据与 100 万行的语料，合适的值不可能一样）。
        self.batch_size = int(batch_size)
        self.path = str(path)
        self.prompt_field = prompt_field
        self.response_field = response_field
        self.chosen_field = chosen_field
        self.rejected_field = rejected_field
        self.system_prompt = system_prompt
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_response_tokens = int(max_response_tokens)
        self.min_response_tokens = int(min_response_tokens)
        self.encoding = encoding

        self.records = self._load(Path(self.path))
        self._init_order(shuffle=shuffle, seed=seed)

    # ------------------------------------------------------------------
    # 读取与过滤
    # ------------------------------------------------------------------
    def _load(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError(
                f"数据集文件不存在：{path}\n"
                f"（相对路径按进程当前工作目录解析。configs/ 里的 data.path "
                f"建议写成相对仓库根目录的路径，脚本会从根目录运行。）"
            )
        # 丢了多少、为什么丢 —— 这些数字会以指标的形式出现在日志里。
        # 一条都不丢是理想情况；丢了很多而不说，才是真正的问题。
        self.dropped: Counter[str] = Counter()
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        with path.open("r", encoding=self.encoding) as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{lineno} 不是合法的 JSON：{exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"{path}:{lineno} 应为 JSON 对象，收到 {type(row).__name__}"
                    )
                # 按**原始行文本**去重，不解析语义。重复的样本会以更高的权重
                # 出现在梯度里，而它的表现只是「训练得比预期更偏」；JSONL 里
                # 重复行通常是导出脚本写重了，属于该拦下的那类错误。
                if line in seen:
                    self.dropped["重复行"] += 1
                    continue
                seen.add(line)
                rows.append(row)

        kept = [row for row in rows if self._keep(row)]
        if not kept:
            raise ValueError(
                f"{path} 里没有任何可用样本（共读入 {len(rows)} 行，"
                f"丢弃原因：{dict(self.dropped)}）。"
            )
        if self.dropped:
            logger.warning("数据集 %s 丢弃了样本：%s", path, dict(self.dropped))
        return kept

    def _keep(self, row: dict[str, Any]) -> bool:
        """行级过滤。返回 False 的行被丢掉，原因记进 ``self.dropped``。"""
        raise NotImplementedError

    def _text(self, row: dict[str, Any], field: str) -> str | None:
        value = row.get(field)
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(
                f"字段 {field!r} 应为字符串，收到 {type(value).__name__}"
                f"（字段映射可用 prompt_field / response_field / chosen_field / "
                f"rejected_field 配置）"
            )
        return value

    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------
    # 分词
    # ------------------------------------------------------------------
    def _encode_prompt(self, text: str) -> torch.Tensor:
        tokenizer = self._require_tokenizer()
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": text})
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        return torch.as_tensor(list(ids), dtype=torch.long)

    def _encode_response(self, text: str) -> torch.Tensor:
        tokenizer = self._require_tokenizer()
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return torch.as_tensor(list(ids), dtype=torch.long)

    def _too_long(self, *pieces: torch.Tensor) -> bool:
        """任一段超过 ``max_response_tokens`` 就丢。

        在**编码后**判而不是编码前：字符数不等于 token 数，用字符数剁会把
        中文/代码样本剁得莫名其妙。

        直接截断是更省事的做法，但那会悄悄改变标签 —— 截掉 response 的后半段之后，
        SFT 学到的是「话说到一半」，DPO 学到的是「短的那个更好」。丢掉并计数，
        好过留下一条语义已经变了的样本。
        """
        if any(piece.numel() > self.max_response_tokens for piece in pieces):
            self.dropped["response 过长"] += 1
            return True
        return False

    # ------------------------------------------------------------------
    def _pad_token_id(self) -> int:
        tokenizer = self._require_tokenizer()
        pad = getattr(tokenizer, "pad_token_id", None)
        if pad is None:
            # 没有 pad token 的分词器（例如 GPT-2）经常用 eos 代替。这不是「随便挑
            # 一个凑数」：pad 位置在 attention_mask / response_mask 里都是 0，
            # 永远不会参与任何归约，所以它的取值只影响「读出来的数字好不好看」。
            pad = getattr(tokenizer, "eos_token_id", None)
        return int(pad if pad is not None else 0)

    def metrics(self) -> dict[str, float]:
        if not self.dropped:
            return {}
        return {f"data/dropped/{reason}": float(count) for reason, count in self.dropped.items()}


@register("dataset", "jsonl_sft")
class JSONLSFTDataset(JSONLDataset):
    """``{prompt, response}`` —— 监督微调。"""

    def _keep(self, row: dict[str, Any]) -> bool:
        prompt = self._text(row, self.prompt_field)
        response = self._text(row, self.response_field)
        if not prompt or not response:
            self.dropped["prompt 或 response 为空"] += 1
            return False
        return True

    def build_batch(self, indices: Sequence[int]) -> Batch:
        prompt_ids, response_ids = [], []
        prompt_texts, response_texts = [], []
        for index in indices:
            row = self.records[index]
            prompt = self._text(row, self.prompt_field) or ""
            response = self._text(row, self.response_field) or ""
            p_ids = self._encode_prompt(prompt)
            r_ids = self._encode_response(response)
            if p_ids.numel() == 0 or r_ids.numel() < self.min_response_tokens:
                self.dropped["编码后过短"] += 1
                continue
            if p_ids.numel() > self.max_prompt_tokens:
                self.dropped["prompt 过长"] += 1
                continue
            if self._too_long(r_ids):
                continue
            prompt_ids.append(p_ids)
            response_ids.append(r_ids)
            prompt_texts.append(prompt)
            response_texts.append(response)

        if not prompt_ids:
            raise ValueError(
                f"这一批 {len(indices)} 个样本全部被过滤掉了"
                f"（累计丢弃原因：{dict(self.dropped)}）。"
                f"把 max_*_tokens 调大，或换一批数据。"
            )

        return Batch.from_rollout(
            prompt_ids,
            response_ids,
            prompt_texts=prompt_texts,
            response_texts=response_texts,
            num_generations=1,
            pad_token_id=self._pad_token_id(),
        )


@register("dataset", "jsonl_preference")
class JSONLPreferenceDataset(JSONLDataset):
    """``{prompt, chosen, rejected}`` —— 偏好对比（DPO）。"""

    #: 对齐字段 + ``preference``（``+1`` chosen / ``-1`` rejected）。
    #:
    #: 比 ``BaseDataset`` 多出来的这一个字段就是「成对数据」与「逐行数据」在
    #: 装配层能看见的全部差别：DPO 的损失 require 了 ``preference``，而这个
    #: ``provides`` 是它唯一的提供方 —— **别把 PREFERENCE 加进
    #: DATASET_OUTPUT_FIELDS**：那样 SFT 数据集也会声称自己提供了它，
    #: 「拿 SFT 数据配 DPO 损失」就会一路通过装配检查，直到训练时报
    #: 「Batch 里没有 preference」。现在它在装配阶段就被拦下。
    provides = frozenset(F.DATASET_OUTPUT_FIELDS | {F.PREFERENCE})

    @property
    def group_size(self) -> int:
        return 2

    def _keep(self, row: dict[str, Any]) -> bool:
        prompt = self._text(row, self.prompt_field)
        chosen = self._text(row, self.chosen_field)
        rejected = self._text(row, self.rejected_field)
        if not prompt or not chosen or not rejected:
            self.dropped["prompt / chosen / rejected 有空值"] += 1
            return False
        if chosen == rejected:
            # Δ = 0 的样本：loss 恒为 log 2，梯度是纯噪声。它不会让训练变差很多，
            # 但它会让「loss 为什么降不下去」变成一个查不到的原因。静默留着有害。
            self.dropped["chosen == rejected"] += 1
            return False
        return True

    def build_batch(self, indices: Sequence[int]) -> Batch:
        prompt_ids, response_ids = [], []
        prompt_texts, response_texts = [], []
        preferences: list[float] = []

        for index in indices:
            row = self.records[index]
            prompt = self._text(row, self.prompt_field) or ""
            chosen = self._text(row, self.chosen_field) or ""
            rejected = self._text(row, self.rejected_field) or ""
            p_ids = self._encode_prompt(prompt)
            c_ids = self._encode_response(chosen)
            r_ids = self._encode_response(rejected)
            if p_ids.numel() == 0 or min(c_ids.numel(), r_ids.numel()) < self.min_response_tokens:
                self.dropped["编码后过短"] += 1
                continue
            if p_ids.numel() > self.max_prompt_tokens:
                self.dropped["prompt 过长"] += 1
                continue
            if self._too_long(c_ids, r_ids):
                continue
            # 固定顺序：chosen 先、rejected 后。**但损失不依赖这个顺序** ——
            # F.PREFERENCE 才是它对内身份的权威来源（见 interfaces 里的说明）。
            prompt_ids.extend([p_ids, p_ids])
            response_ids.extend([c_ids, r_ids])
            prompt_texts.extend([prompt, prompt])
            response_texts.extend([chosen, rejected])
            preferences.extend([1.0, -1.0])

        if not prompt_ids:
            raise ValueError(
                f"这一批 {len(indices)} 个偏好对全部被过滤掉了"
                f"（累计丢弃原因：{dict(self.dropped)}）。"
            )

        batch = Batch.from_rollout(
            prompt_ids,
            response_ids,
            prompt_texts=prompt_texts,
            response_texts=response_texts,
            num_generations=2,
            pad_token_id=self._pad_token_id(),
        )
        batch[F.PREFERENCE] = torch.tensor(preferences, dtype=torch.float32)
        return batch
