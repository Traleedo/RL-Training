"""把 Anthropic HH-RLHF 转成本仓库 ``jsonl_preference`` 的格式（单轮）。

HH 每条样本只有 ``chosen`` / ``rejected`` 两列，都是**完整多轮对话**字符串：

    \\n\\nHuman: 第一轮\\n\\nAssistant: 回复\\n\\nHuman: 第二轮\\n\\nAssistant: 回复

偏好比较的是**最后一轮 assistant 回复**。本仓库的 ``jsonl_preference`` 要的是
``{prompt, chosen, rejected}``：prompt 会被当成**一条 user message** 送进
``apply_chat_template``（见 ``data/jsonl.py`` 的 ``_encode_prompt``），response
单独编码并构成 ``response_mask``。所以这里做**单轮化**：

    prompt   = 最后一轮 Human 的内容（更早的历史丢掉）
    chosen   = chosen   里最后一轮 Assistant 的回复
    rejected = rejected 里最后一轮 Assistant 的回复

丢掉多轮历史是有意的取舍：把历史压成一条超长 user message 会让 chat template
变形，而保留多轮又要求 prompt 是消息列表（当前契约是字符串）。单轮最省事、
语义最干净 —— 但产出的数据与 HH 原文的多轮上下文有出入，这是已知代价。

用法::

    # 1) 拉数据（HF 直连不通的机器，在能联网处下载后拷过来即可）
    #    hf download Anthropic/hh-rlhf --repo-type dataset --local-dir data/hh-rlhf
    # 2) 转换（可传多个文件或目录，会合并并去重）
    python -X utf8 scripts/prepare_hh.py \
        data/hh-rlhf/data/helpful-base/train.jsonl.gz --out data/hh_preference.jsonl

    # 3) 接上 DPO —— 不改任何配置文件，--data 覆盖 data.path
    python -X utf8 scripts/train.py -a dpo --data data/hh_preference.jsonl \
        --set data.max_prompt_tokens=512

脚本**不加载分词器**，只统计字符数，所以它不依赖 transformers / datasets，
在本机环境也能跑。末尾打印的字符长度分位用来决定 ``max_prompt_tokens``
（英文大致 4 字符 ≈ 1 token，仅作起点）。
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

MARKER = "\n\nAssistant:"
HUMAN = "\n\nHuman:"


def _split_last_turn(text: str) -> tuple[str, str] | None:
    """把 ``...\\n\\nHuman: Q\\n\\nAssistant: A`` 切成 ``(prompt, response)``。

    prompt 只保留**最后一轮** Human 的内容。找不到标记时返回 ``None``。
    """
    end = text.rfind(MARKER)
    if end < 0:
        return None
    response = text[end + len(MARKER):].strip()
    history = text[:end]
    start = history.rfind(HUMAN)
    if start < 0:
        return None
    return history[start + len(HUMAN):].strip(), response


def _iter_rows(path: Path) -> Iterator[tuple[int, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc


def _collect_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            files += sorted(
                p for p in path.rglob("*") if p.suffix in (".jsonl", ".gz")
            )
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"找不到输入：{path}")
    if not files:
        raise SystemExit("没有匹配到任何 .jsonl / .jsonl.gz 文件。")
    return files


def _percentile(values: list[int], percent: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round(percent / 100 * (len(ordered) - 1))
    return ordered[index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", help="HH 文件（.jsonl / .jsonl.gz）或目录")
    parser.add_argument("--out", default="data/hh_preference.jsonl", help="输出文件")
    parser.add_argument("--chosen-field", default="chosen")
    parser.add_argument("--rejected-field", default="rejected")
    args = parser.parse_args(argv)

    files = _collect_files(args.inputs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    read = kept = 0
    dropped: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()
    prompt_chars: list[int] = []
    response_chars: list[int] = []

    with out_path.open("w", encoding="utf-8") as out:
        for path in files:
            for _lineno, row in _iter_rows(path):
                read += 1
                chosen_raw = row.get(args.chosen_field)
                rejected_raw = row.get(args.rejected_field)
                if not isinstance(chosen_raw, str) or not isinstance(rejected_raw, str):
                    dropped["缺 chosen / rejected 字段"] += 1
                    continue

                pair_c = _split_last_turn(chosen_raw)
                pair_r = _split_last_turn(rejected_raw)
                if pair_c is None or pair_r is None:
                    dropped["切不出 Human / Assistant 轮次"] += 1
                    continue
                prompt, chosen = pair_c
                other_prompt, rejected = pair_r
                if prompt != other_prompt:
                    dropped["chosen / rejected 前缀不一致"] += 1
                    continue
                if not prompt or not chosen or not rejected:
                    dropped["prompt 或回复为空"] += 1
                    continue
                if chosen == rejected:
                    # Δ = 0 的样本：DPO 的 loss 恒为 log 2、梯度是纯噪声。
                    # 框架读入时也会过滤（JSONLPreferenceDataset._keep），
                    # 这里先丢是为了让产出的文件本身干净。
                    dropped["chosen == rejected"] += 1
                    continue

                key = (prompt, chosen, rejected)
                if key in seen:
                    dropped["重复样本"] += 1
                    continue
                seen.add(key)

                out.write(
                    json.dumps(
                        {"prompt": prompt, "chosen": chosen, "rejected": rejected},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept += 1
                prompt_chars.append(len(prompt))
                response_chars += [len(chosen), len(rejected)]

    print(f"读入 {read} 行（{len(files)} 个文件）-> 写出 {kept} 对：{out_path}")
    if dropped:
        print("丢弃：", dict(dropped))
    if prompt_chars:
        print(
            f"prompt   字符数：中位 {_percentile(prompt_chars, 50)}  "
            f"p90 {_percentile(prompt_chars, 90)}  "
            f"p99 {_percentile(prompt_chars, 99)}  "
            f"最大 {max(prompt_chars)}"
        )
        print(
            f"response 字符数：中位 {_percentile(response_chars, 50)}  "
            f"p90 {_percentile(response_chars, 90)}  "
            f"p99 {_percentile(response_chars, 99)}  "
            f"最大 {max(response_chars)}"
        )
    if kept == 0:
        print("一对都没写出来 —— 检查 --chosen-field / --rejected-field 或输入格式。",
              file=sys.stderr)
        return 1

    print(
        "\n下一步：\n"
        f"  python -X utf8 scripts/train.py -a dpo --data {out_path} "
        "--set data.max_prompt_tokens=512\n"
        "（p90 明显超过 512 字符就调大；超长的行会被数据集在编码后静默丢弃，"
        "丢弃原因会出现在训练日志里。）"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
