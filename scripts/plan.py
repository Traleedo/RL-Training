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
    parser.add_argument("config", nargs="?", help="配置名（ppo / rl/ppo）或路径",default="--all")
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

    import components  
    import data  
    import tests.fixtures  

    for path in targets:
        describe_config(path)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
