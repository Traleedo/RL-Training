"""打印一个配置的**装配计划** —— 不训练、不下载模型，只回答「会构建什么、为什么」。

用法::

    python -X utf8 scripts/plan.py ppo            # 也接受 rl/ppo 或完整路径
    python -X utf8 scripts/plan.py dapo
    python -X utf8 scripts/plan.py --all          # 全部配置并排看

核心问题：``配置里写了 critic 那一段`` **不等于** ``Critic 会被加载``。到底加载
哪些模型，是从存活组件的 ``needed`` 并集推出来的。例如 ``dapo.yaml`` 里还挂着
``kl_k3`` 那一行，Reference 却不会被构建 —— 因为 ``coef: 0.0`` 释放了
``ref_logprobs`` 依赖。这件事只有把计划打出来才看得见。
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
    from core.config import load_config
    from engine.logic import build_logic_components

    # 用 build_logic_components 而不是 build_trainer：后者会**真的把模型建出来**。
    # 计划只依赖各组件的 requires / provides，而那些组件全是纯逻辑 —— 换到真实
    # 的 Qwen2.5 之后，「看一眼计划」不该变成「先下载几 GB + 一个 8B 奖励模型」。
    cfg = load_config(path)
    logic = build_logic_components(cfg)
    plan = logic.plan

    print(f"\n{'=' * 62}")
    print(f"{path.name}")
    print("=" * 62)
    print(f"  训练器：{'RLTrainer' if logic.dataset is None else 'OfflineTrainer'}"
          f"（trainer.kind={cfg.trainer.get('kind', 'rl')}）")

    if logic.dataset is not None:
        # 数据集是离线家族的相位 A，也是它的 provides 里唯一可能多出
        # preference 的地方 —— 而 preference 正是 DPO 被装配出来的依据。
        print(f"\n  {logic.dataset.describe()}")

    print(plan.describe())

    # 存活的损失项：权重非零的才会进依赖并集，所以这里直接就是结论。
    print("\n  损失项（名称 × 权重）：")
    for term, weight in logic.loss_terms:
        marker = " " if weight else "×"  # 权重 0 的项在装配阶段就被剔除
        print(f"    {marker} {term.name():<18} weight={weight}  {term.describe()}")

    if logic.controllers:
        print("\n  控制器：")
        for controller in logic.controllers:
            print(f"    - {controller.describe()}")
    else:
        print(
            "\n  控制器：无。"
            "（自适应 KL 是在 algorithm.controllers 下加一项，见 "
            "configs/ppo_adaptive_kl.yaml）"
        )

    # 模型是否构建，以**计划**为准 —— 那才是 `_assemble()` 照着做的东西。
    print("\n  将会构建的模型（预测，本脚本不真的加载）：")
    print(f"    {'actor':<10} 总是构建")
    print(f"    {'reference':<10} {'构建' if plan.need_reference else '不构建'}")
    print(f"    {'critic':<10} {'构建' if plan.need_critic else '不构建'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="打印装配计划（不训练）")
    parser.add_argument("config", nargs="?", help="配置名（ppo / rl/ppo）或路径")
    parser.add_argument(
        "--all", action="store_true", help="打印全部算法配置（rl/ 下 + 根目录）"
    )
    args = parser.parse_args()

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

    for path in targets:
        describe_config(path)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
