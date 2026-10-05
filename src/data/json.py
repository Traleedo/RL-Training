"""整份 JSON 数组的数据后端 —— ``json_sft`` / ``json_preference``。

与 ``jsonl_sft`` / ``jsonl_preference`` 只差**文件怎么切成一堆对象**：
这里是 ``[{"prompt": ...}, ...]`` 一份数组，那边是一行一个对象。
去重、字段过滤、编码、成 Batch 全部复用 ``JSONLDataset`` 那一套。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from core.registry import register

from data.jsonl import JSONLPreferenceDataset, JSONLSFTDataset

__all__ = ["JSONSFTDataset", "JSONPreferenceDataset"]


class _JSONArrayRows:
    def _iter_rows(self, path: Path) -> Iterator[tuple[str, str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding=self.encoding))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 不是合法的 JSON：{exc}") from exc

        if not isinstance(payload, list):
            raise ValueError(
                f"{path} 的顶层应为数组（``[{{...}}, ...]``），"
                f"收到 {type(payload).__name__}。\n"
                f"如果文件是一行一个 JSON 对象，请改用 jsonl_sft / jsonl_preference。"
            )

        for index, row in enumerate(payload):
            yield (
                f"{path}[{index}]",
                json.dumps(row, sort_keys=True, ensure_ascii=False),
                row,
            )


@register("dataset", "json_sft")
class JSONSFTDataset(_JSONArrayRows, JSONLSFTDataset):
    """整份 JSON 数组，元素形如 ``{prompt, response}``。"""


@register("dataset", "json_preference")
class JSONPreferenceDataset(_JSONArrayRows, JSONLPreferenceDataset):
    """整份 JSON 数组，元素形如 ``{prompt, chosen, rejected}``。"""
