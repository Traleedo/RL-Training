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

CONFIGS = ROOT / "configs"

#: 由基座配置提供的块。
MODEL_BLOCKS = ("model", "actor", "critic", "reference", "rollout", "reward_model")

#: 没给 --prompts 时用的内置 prompt。
DEFAULT_PROMPTS = ["用一个比喻解释梯度下降", "为什么天空是蓝色的", "写一句关于春天的诗"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按配置名训练（RL / 离线）")
    parser.add_argument("-a", "--algorithm", required=True, help="算法配置名：ppo / grpo / sft / dpo ...")
    parser.add_argument("-m", "--model", default=None, help="基座配置名，例如qwen3-0.6b")
    parser.add_argument(
        "--reward-model",
        default=None,
        help="覆盖 reward_model.name_or_path，并把 scorer 切成 hf_reward_model",
    )
    parser.add_argument("--prompts", default=None, help="RL 的 prompt 文件，每行一个")
    parser.add_argument("--data", default=None, help="离线配置的数据集路径（覆盖 data.path）")
    parser.add_argument("--steps", type=int, default=10, help="train_step 的次数")
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="auto（默认）= 有 CUDA 用 CUDA、否则 CPU",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="覆盖 optim.lr")
    parser.add_argument("--save-dir", default=None, help="checkpoint 目录")
    parser.add_argument("--resume", default=None, help="从某个 checkpoint 续跑")
    parser.add_argument("--every", type=int, default=1, help="每 N 步打印一次指标")
    parser.add_argument("--set", action="append", default=[], metavar="K=V", help="任意点号覆盖，可重复")
    parser.add_argument("--plan", action="store_true", help="只打装配计划，不训练")
    return parser.parse_args(argv)


def _own_of(path: Path):
    from omegaconf import OmegaConf

    raw = OmegaConf.load(path)
    defaults = raw.get("defaults", None)
    if defaults is None or len(defaults) != 2 or "model" not in str(defaults[1]):
        raise ValueError(
            f"{path.name} 的 defaults 不是「base + 一个基座」的形状"
            f"（实际是 {defaults}），--model 无法安全地替换基座。\n"
            f"请手动编辑该文件的 defaults，或去掉 -m。"
        )
    own = OmegaConf.to_container(raw, resolve=False)
    own.pop("defaults", None)
    return OmegaConf.create(own)


def load_config_with_model(algorithm: str, model: str | None):
    """按算法 + 基座组装配置。``model`` 为 None 时用配置自己的默认链。"""
    from omegaconf import OmegaConf

    from core.config import find_config, load_config

    algorithm_path = find_config(algorithm, CONFIGS)
    if model is None:
        return load_config(algorithm_path), algorithm_path

    model_path = find_config(f"model/{model}", CONFIGS)

    cfg = OmegaConf.merge(
        load_config(CONFIGS / "base.yaml"),
        load_config(model_path),
        _own_of(algorithm_path),
    )
    return cfg, algorithm_path


def read_prompts(path: str | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    prompts = [line.strip() for line in lines if line.strip()]
    if not prompts:
        raise SystemExit(f"{path} 里没有非空行 —— prompt 列表是空的。")
    return prompts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.steps < 1:
        print("--steps 必须 >= 1", file=sys.stderr)
        return 2
    try:
        cfg, algorithm_path = load_config_with_model(args.algorithm, args.model)
    except FileNotFoundError as exc:
        print(f"找不到配置：{exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"配置无法合并：{exc}", file=sys.stderr)
        return 2

    from omegaconf import OmegaConf

    if args.reward_model:
        if "reward_model" not in cfg:
            cfg.reward_model = OmegaConf.create(
                {"dtype": "bfloat16", "batch_size": 16, "scale": 1.0}
            )
        cfg.reward_model.name_or_path = args.reward_model
        cfg.reward.scorer.type = "hf_reward_model"
    if args.data is not None:
        if "data" not in cfg:
            print(
                f"--data 只对离线配置有意义，而 {algorithm_path.name} 没有 data 块。",
                file=sys.stderr,
            )
            return 2
        cfg.data.path = args.data
    if args.device is not None:
        cfg.trainer.device = args.device
    if args.seed is not None:
        cfg.trainer.seed = args.seed
    if args.lr is not None:
        cfg.optim.lr = args.lr
    if args.save_dir is not None:
        cfg.checkpointer.save_dir = args.save_dir
    if args.set:
        try:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.set))
        except Exception as exc: 
            print(f"--set 解析失败：{exc}", file=sys.stderr)
            return 2
    # 调度器的 horizon 要与真实步数一致，否则 cosine/linear 会算错终点。
    cfg.trainer.total_steps = args.steps

    import components  
    import data  
    kind = str(cfg.trainer.get("kind", "rl")).lower()

    if args.plan:
        # 只看计划：装配计划完全由 processor / advantage / loss / controller
        # 的 requires 决定，**一个权重都不用加载**。真走 build_trainer 的话，
        # 会先把 Qwen2.5 和一个 8B 奖励模型下载下来。
        from engine.logic import build_logic_components

        logic = build_logic_components(cfg)
        print(f"{algorithm_path.name}  |  "
              f"{'RLTrainer' if logic.dataset is None else 'OfflineTrainer'}（kind={kind}）")
        print(f"  基座：{cfg.model.name_or_path}（actor={cfg.actor.type}，"
              f"critic={cfg.critic.type}）")
        print(logic.plan.describe())
        if logic.dataset is not None:
            print(f"  {logic.dataset.describe()}")
        print("\n（--plan 只算计划，没有加载任何模型）")
        return 0

    from engine.build import build_trainer

    trainer = build_trainer(cfg)

    print(f"{algorithm_path.name}  |  {type(trainer).__name__}（kind={kind}）")
    print(f"  基座：{cfg.model.name_or_path}（actor={cfg.actor.type}，critic={cfg.critic.type}）")
    print(trainer.plan.describe())
    if hasattr(trainer, "dataset"):
        print(f"  {trainer.dataset.describe()}")

    if args.resume:
        trainer.resume(args.resume)
        print(f"从 {args.resume} 续跑（global_step={trainer.global_step}）")

    prompts = read_prompts(args.prompts) if kind == "rl" else None
    if prompts is not None:
        print(f"  prompt：{len(prompts)} 条")
    print("-" * 62)

    pbar = tqdm(
        range(1, args.steps + 1),
        desc=algorithm_path.name,
        unit="step",
        dynamic_ncols=True,
    )
    for step in pbar:
        # RL 的 prompt 由调用方喂进来；离线是数据集在训练器里，train_step 无参数。
        metrics = trainer.train_step() if kind == "offline" else trainer.train_step(prompts)
        if metrics.get("train/skipped_step") == 1.0:
            # 被过滤空的步不推进 global_step，进度条上标出来，别当成正常步数
            pbar.set_postfix_str("skipped")
            continue
        # 进度条后缀实时刷新最新 loss / grad；global_step 与循环计数可以不同
        # （跳过不推进步数），所以这里显式带上 global_step。
        pbar.set_postfix(
            step=trainer.global_step,
            loss=f"{metrics['train/loss']:.4f}",
            grad=f"{metrics.get('train/grad_norm', float('nan')):.4f}",
        )
        # 周期性完整行照旧打印（tqdm.write 而不是 print —— 后者会把进度条
        # 冲成好几行）。进度条负责「现在到哪了」，这些行负责「可回看的记录」。
        if step % args.every == 0 or step == args.steps:
            pbar.write(
                f"step {trainer.global_step:>4}  loss={metrics['train/loss']:>9.4f}  "
                f"grad_norm={metrics.get('train/grad_norm', float('nan')):>8.4f}"
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
