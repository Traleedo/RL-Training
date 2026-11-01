#!/usr/bin/env python
"""从 ModelScope 预下载本项目用到的模型。

为什么要单独一个脚本，而不是等训练时自动下：
    ``models/hf_common.py::_download_to_local`` 确实会优先走 ModelScope，
    但那条路是**隐式**的 —— 训练启动后卡在下载上时，日志里只有一行
    "正在从 ModelScope 下载…"，看不出还要多久、也看不出下到哪了。
    先在终端里把权重拉好，后面每次启动都只走本地缓存（<1 秒）。

用法::

    python scripts/download_models_ms.py                  # 策略模型（跑 bench 够用）
    python scripts/download_models_ms.py --only policy,reward
    python scripts/download_models_ms.py --only all --cache /root/autodl-tmp/models
    python scripts/download_models_ms.py --list           # 只列清单

下载完成后会打印每个模型的绝对路径，可直接填进配置的
``model.name_or_path`` / ``reward_model.name_or_path``（填本地目录即跳过下载）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

#: key -> (ModelScope model id, 用途说明)
MODELS: dict[str, tuple[str, str]] = {
    "policy": ("Qwen/Qwen3-0.6B", "策略模型：actor 与 reference 共用（bench 只需要它）"),
    "reward": (
        "Skywork/Skywork-Reward-V2-Qwen3-8B",
        "结果奖励模型：configs/model/qwen3-0.6b.yaml 的默认值，bf16 约 16 GB",
    ),
    "reward-small": (
        "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
        "奖励模型的小号替身：24 GB 卡上跑完整训练时用它，省约 15 GB",
    ),
}

DEFAULT_CACHE = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/autodl-tmp/models"))


def _resolve_only(value: str) -> list[str]:
    if value.strip().lower() in {"all", "*"}:
        return list(MODELS)
    keys = [k.strip() for k in value.split(",") if k.strip()]
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        raise SystemExit(f"未知的模型名 {unknown}；可选：{sorted(MODELS)} 或 all")
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description="从 ModelScope 下载本项目用到的模型")
    ap.add_argument(
        "--only",
        default="policy",
        help="要下的模型，逗号分隔：policy / reward / reward-small / all（默认 policy）",
    )
    ap.add_argument("--cache", default=DEFAULT_CACHE, help=f"缓存目录（默认 {DEFAULT_CACHE}）")
    ap.add_argument("--list", action="store_true", help="只打印清单，不下载")
    args = ap.parse_args()

    if args.list:
        print(f"缓存目录：{args.cache}\n")
        for key, (mid, desc) in MODELS.items():
            print(f"  {key:<12} {mid:<42} {desc}")
        return 0

    keys = _resolve_only(args.only)

    try:
        from modelscope import snapshot_download
    except ImportError:
        print(
            "modelscope 没装。装它：\n"
            "    uv pip install modelscope\n"
            "或走 HF（不推荐，国内慢）：\n"
            "    HF_ENDPOINT=https://hf-mirror.com huggingface-cli download <model_id>",
            file=sys.stderr,
        )
        return 1

    os.makedirs(args.cache, exist_ok=True)
    print(f"缓存目录：{args.cache}")
    print(f"待下载 {len(keys)} 个：{', '.join(keys)}\n")

    resolved: dict[str, str] = {}
    failed: list[str] = []

    for i, key in enumerate(keys, 1):
        mid, desc = MODELS[key]
        print(f"[{i}/{len(keys)}] {mid}")
        print(f"        {desc}")
        started = time.perf_counter()
        try:
            path = snapshot_download(model_id=mid, cache_dir=args.cache)
        except Exception as exc:  # noqa: BLE001 — 网络/磁盘/ID 不存在都归到这里
            print(f"        失败：{type(exc).__name__}: {exc}\n")
            failed.append(key)
            continue
        elapsed = time.perf_counter() - started
        resolved[key] = path
        print(f"        完成，用时 {elapsed:.1f}s")
        print(f"        路径：{path}\n")

    if resolved:
        print("=" * 68)
        print("可直接写进配置的本地路径（写本地目录 = 启动时不再联网）：\n")
        for key, path in resolved.items():
            if key == "policy":
                print(f"  model.name_or_path:        {path}")
            elif key == "reward-small":
                print(f"  reward_model.name_or_path: {path}   # 小号，24G 卡用这个")
            else:
                print(f"  reward_model.name_or_path: {path}")
        print("=" * 68)

    if failed:
        print(f"\n以下模型没下成：{failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
