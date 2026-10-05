

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="跑一个离线配置（SFT / DPO）")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("-m", "--model", default=None, help="基座配置名，例如qwen2.5-7b")
    parser.add_argument("--steps", type=int, default=10, help="train_step 的次数")
    parser.add_argument(
        "--save-dir",
        default=None,
        help="checkpoint 目录。给了就在跑完之后存一个 step_<N>",
    )
    parser.add_argument(
        "--resume", default=None, help="从某个 checkpoint 文件续跑（例如 .../step_2）"
    )
    parser.add_argument(
        "--every", type=int, default=0, help="每 N 步打印一次指标（0 = 每步都打）"
    )
    return parser.parse_args(argv)


def resolve_config(name: str) -> Path:
    """允许 ``--config sft`` 这种简写 —— 与 plan.py 走同一份布局知识。

    路径解析集中在 ``core.config.find_config`` 里，这样 configs/ 的目录结构
    只有一份定义；否则每加一个脚本就多一份会漂移的副本。
    """
    from core.config import find_config

    try:
        return find_config(name, ROOT / "configs")
    except FileNotFoundError as exc:
        print(f"找不到配置：{name}\n{exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.steps < 1:
        print("--steps 必须 >= 1", file=sys.stderr)
        return 2

    config_path = resolve_config(args.config)

    import components  # noqa: F401  触发真实组件注册
    import data  # noqa: F401  触发数据集注册

    from engine.build import build_trainer
    from core.config import assemble_with_model, load_config, resolve_model_config

    if args.model is None:
        cfg = load_config(config_path)
    else:
        cfg = assemble_with_model(
            config_path, resolve_model_config(args.model, ROOT / "configs"), ROOT / "configs"
        )
    if args.save_dir is not None:
        cfg.checkpointer.save_dir = args.save_dir

    trainer = build_trainer(cfg)
    kind = str(cfg.trainer.get("kind", "rl")).lower()
    if kind != "offline":
        print(
            f"{config_path.name} 的 trainer.kind 是 {kind!r}，不是 'offline'。\n"
            f"这个脚本只跑离线配置（sft / dpo）；RL 配置请用 scripts/verify_e2e.py。",
            file=sys.stderr,
        )
        return 2

    if args.resume:
        trainer.resume(args.resume)
        print(f"从 {args.resume} 续跑（global_step={trainer.global_step}，"
              f"数据游标={trainer._cursor}）")

    print(f"{config_path.name}  |  {type(trainer).__name__}")
    print(f"  {trainer.dataset.describe()}")
    print(f"  每步取 {trainer.dataset.batch_size} 个样本"
          f"（= {trainer.dataset.batch_size * trainer.dataset.group_size} 行）")
    print(f"  损失项：{[(t.name(), w) for t, w in trainer.loss_terms]}")
    print("-" * 62)

    every = args.every or 1
    history: list[float] = []
    pbar = tqdm(
        range(1, args.steps + 1),
        desc=config_path.name,
        unit="step",
        dynamic_ncols=True,
    )
    for step in pbar:
        metrics = trainer.train_step()
        history.append(metrics["train/loss"])
        # 数据游标是离线家族特有的、跨进程存活的状态 —— 放进后缀，一眼看得见
        # 它是否在推进（不推进就等于一直在训同几行）。
        pbar.set_postfix(
            step=trainer.global_step,
            loss=f"{metrics['train/loss']:.4f}",
            grad=f"{metrics.get('train/grad_norm', float('nan')):.4f}",
            cursor=trainer._cursor,
        )
        if step % every == 0 or step == args.steps:
            pbar.write(
                f"step {trainer.global_step:>4}  loss={metrics['train/loss']:>9.4f}  "
                f"grad_norm={metrics.get('train/grad_norm', float('nan')):>8.4f}  "
                f"游标={trainer._cursor}"
            )

    # 数据集自报的丢弃计数：一条都不丢是理想情况，丢了很多而不说才是问题。
    dropped = trainer.dataset.metrics()
    if dropped:
        print("-" * 62)
        print("数据集丢弃：")
        for key, value in sorted(dropped.items()):
            print(f"  {key} = {int(value)}")

    print("-" * 62)
    print(
        f"跑完 {args.steps} 步：loss {history[0]:.4f} -> {history[-1]:.4f}"
        f"（首末两点，含采样噪声，看趋势请看完整输出）"
    )

    if args.save_dir is not None:
        path = trainer.save_checkpoint(
            str(Path(args.save_dir) / f"step_{trainer.global_step}")
        )
        print(f"checkpoint 已存到 {path}")

    trainer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
