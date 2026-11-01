"""探测器：能不能在本机把 ``sglang.Engine`` 起起来。

为什么单独一个脚本
------------------
1. **必须在子进程里跑**。sglang 的 ``shutdown()`` 不保证把静态显存还给驱动，
   在训练进程内试错失败后，后面的 bench 会莫名其妙 OOM。子进程退出即回收。
2. **要能逐级回退**。不同卡/不同 sglang 版本接受的参数不一样：
   Ampere+ 可以全默认，Turing（T4, sm75）必须 ``attention_backend=triton``
   且 ``sampling_backend=pytorch``（FlashInfer 不支持 sm75）。
   三级从「最完整」到「最保守」试，第一个成功的就是本机可用的配置。
3. **结论要机器可读**。最后一行输出 ``RESULT {json}``，notebook 直接解析，
   不用去正则匹配 traceback。

用法::

    python scripts/probe_sglang_engine.py /kaggle/input/.../0.6b/1 --dtype float16

退出码 0 = 引擎可用；1 = 全部层级都失败（不代表代码有问题，通常是硬件/依赖）。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _path in (str(ROOT / "src"), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

#: 从「最完整」到「最保守」。label 会写进 RESULT，方便记录本机到底吃哪一档。
LEVELS: list[tuple[str, dict]] = [
    (
        "triton+pytorch-sampling",
        {"attention_backend": "triton", "sampling_backend": "pytorch"},
    ),
    ("triton", {"attention_backend": "triton"}),
    ("default", {}),
]

#: 一小段无害的 token（BOS 之后随便几个），只验证「能算出一个 token」。
PROBE_INPUT = [9707, 11, 1879]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="探测 sglang.Engine 能否启动")
    parser.add_argument("model_path", help="HF 目录或 repo id")
    parser.add_argument("--dtype", default="float16", help="T4 必须 float16")
    parser.add_argument("--mem-fraction", type=float, default=0.22)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from models.sglang_rollout import SGLangEngineBackend, SamplingParams

    for label, extra in LEVELS:
        print(f"\n== try [{label}] {extra}", flush=True)
        try:
            backend = SGLangEngineBackend(
                args.model_path,
                dtype=args.dtype,
                tp_size=args.tp_size,
                mem_fraction_static=args.mem_fraction,
                engine_kwargs=extra,
            )
            out = backend.generate(
                [PROBE_INPUT],
                SamplingParams(max_new_tokens=args.max_new_tokens, n=1),
            )
            ids = out[0][0].token_ids
            print("OK ids:", ids, "| text:", repr(out[0][0].text[:60]), flush=True)
            print(
                "RESULT "
                + json.dumps(
                    {"ok": True, "level": label, "engine_kwargs": extra, "ids": ids},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            backend.close()
            return 0
        except Exception as exc:  # noqa: BLE001 - 探测就是要吃掉一切
            print(f"FAIL: {type(exc).__name__}: {str(exc)[:800]}", flush=True)
            traceback.print_exc()

    print("RESULT " + json.dumps({"ok": False}, ensure_ascii=False), flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
