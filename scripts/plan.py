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
    cfg = load_config(path)
    logic = build_logic_components(cfg)
    plan = logic.plan

    print(f"\n{'=' * 62}")
    print(f"{path.name}")
    print("=" * 62)
    print(f"  训练器：{'RLTrainer' if logic.dataset is None else 'OfflineTrainer'}"
          f"（trainer.kind={cfg.trainer.get('kind', 'rl')}）")

    if logic.dataset is not None:
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
