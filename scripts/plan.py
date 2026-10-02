"""打印一个配置的**装配计划** —— 不训练，只回答「会构建什么、为什么」。

用法::

    python -X utf8 scripts/plan.py ppo            # 也接受 rl/ppo 或完整路径
    python -X utf8 scripts/plan.py dapo
    python -X utf8 scripts/plan.py --all          # 十个配置并排看

为什么值得单独有个脚本
--------------------
装配计划是这个框架的核心产物，而它回答的问题出奇地反直觉：

    「配置里写了 ``critic`` 那一段」**不等于**「Critic 会被加载」。

到底加载哪些模型，是从存活组件的 ``needed`` 并集**推**出来的。所以
``configs/rl/dapo.yaml`` 里明明还挂着 ``kl_k3`` 那一行，Reference 却不会被构建
（``coef: 0.0`` 释放了 ``ref_logprobs`` 依赖）。这件事只有把计划打出来才看得见。

十个配置共用同一份 ``base.yaml``，也就是同一份模型配置段，而它们构建的模型
各不相同 —— 把计划并排打印出来，是这个设计最直接的一次演示。

其中 ``sft.yaml`` 与 ``dpo.yaml`` 是同一个 ``OfflineTrainer``、同一份
``base.yaml``、同一套装配函数，**Reference 却一个不建一个建**。差别只有一行：

    dpo 的 requires 里有 ``ref_logprobs``，cross_entropy 的没有。

把 ``--all`` 的输出往下翻到这两段对比着看，是「条件依赖」最干净的证据。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

CONFIGS = ROOT / "configs"


def describe_config(path: Path) -> None:
    from engine.build import build_trainer
    from core.config import load_config

    # 用 build_trainer 而不是 RLTrainer：分派是配置的事（trainer.kind），
    # 这个脚本只负责把结果打出来，不该自己知道有几个家族。
    trainer = build_trainer(load_config(path))

    print(f"\n{'=' * 62}")
    print(f"{path.name}")
    print("=" * 62)
    print(f"  训练器：{type(trainer).__name__}（trainer.kind="
          f"{trainer.cfg.trainer.get('kind', 'rl')}）")

    dataset = getattr(trainer, "dataset", None)
    if dataset is not None:
        # 数据集是离线家族的相位 A，也是它的 provides 里唯一可能多出
        # preference 的地方 —— 而 preference 正是 DPO 被装配出来的依据。
        print(f"\n  {dataset.describe()}")

    print(trainer.plan.describe())

    # 存活的损失项：权重非零的才会进依赖并集，所以这里直接就是结论。
    print("\n  损失项（名称 × 权重）：")
    for term, weight in trainer.loss_terms:
        marker = " " if weight else "×"  # 权重 0 的项在装配阶段就被剔除
        print(f"    {marker} {term.name():<18} weight={weight}  {term.describe()}")

    if trainer.controllers:
        print("\n  控制器：")
        for controller in trainer.controllers:
            print(f"    - {controller.describe()}")
    else:
        print(
            "\n  控制器：无。"
            "（自适应 KL 是在 algorithm.controllers 下加一项，见 "
            "configs/ppo_adaptive_kl.yaml）"
        )

    print("\n  实际构建的模型：")
    print(f"    {'actor':<10} 总是构建")
    print(f"    {'reference':<10} {'构建' if trainer.reference is not None else '不构建'}")
    print(f"    {'critic':<10} {'构建' if trainer.critic is not None else '不构建'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="打印装配计划（不训练）")
    parser.add_argument("config", nargs="?", help="配置名（ppo / rl/ppo）或路径")
    parser.add_argument(
        "--all", action="store_true", help="打印全部算法配置（rl/ 下 + 根目录）"
    )
    args = parser.parse_args()

    # 参数校验放在 import 之前：import torch 要两秒多，而「名字写错了」这个
    # 错误不需要 torch 就能判断。把校验提前，错误路径就是瞬时的 ——
    # tests/test_scripts.py 里那条「配置不存在」的用例因此几乎不花时间。
    from core.config import find_config, iter_algorithm_configs

    if args.all:
        targets = iter_algorithm_configs(CONFIGS)
    elif args.config:
        try:
            targets = [find_config(args.config, CONFIGS)]
        except FileNotFoundError as exc:
            print(f"找不到配置：{args.config}\n{exc}", file=sys.stderr)
            return 2
    else:
        parser.error("要么给一个配置名，要么用 --all")
        return 2

    import components  # noqa: F401  触发真实组件注册
    import data  # noqa: F401  触发数据集注册（离线家族要用）
    import tests.fixtures  # noqa: F401  触发 fixture 组件注册

    for path in targets:
        describe_config(path)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
