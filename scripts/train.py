from __future__ import annotations

import argparse
import sys
from pathlib import Path
from omegaconf import OmegaConf
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
        help="导入预训练好的奖励模型",
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help="RL 的本地 prompt 文件（每行一个）。与 --dataset 二选一",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="RL prompt 数据集：内置别名（ultrafeedback）或 HuggingFace repo id。"
        "先试 HuggingFace，失败再切 ModelScope",
    )
    parser.add_argument("--dataset-split", default=None, help="数据集 split（默认 train）")
    parser.add_argument(
        "--prompt-field", default=None, help="prompt 字段名（默认 instruction）"
    )
    parser.add_argument(
        "--dataset-limit", type=int, default=None, help="最多取多少条 prompt（默认全取）"
    )
    parser.add_argument(
        "--prompts-per-step",
        type=int,
        default=None,
        help="每步采样多少条 prompt；--dataset 默认 8，--prompts 默认全部",
    )
    parser.add_argument("--data", default=None, help="离线配置的数据集路径（覆盖 data.path）")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="train_step 的次数（不传则用配置里的 trainer.total_steps）",
    )
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


def load_config_with_model(algorithm: str, model: str | None):
    """按算法 + 基座组装配置。``model`` 为 None 时用配置自己的默认链。

    合并规则只有一份定义（``core.config.assemble_with_model``），
    这里只负责把「算法名 / 基座名」解析成路径。
    """

    from core.config import (
        assemble_with_model,
        find_config,
        load_config,
        resolve_model_config,
    )

    algorithm_path = find_config(algorithm, CONFIGS)
    if model is None:
        return load_config(algorithm_path), algorithm_path

    model_path = resolve_model_config(model, CONFIGS)
    return assemble_with_model(algorithm_path, model_path, CONFIGS), algorithm_path


def build_prompt_sampler(args: argparse.Namespace, cfg):
    """把 ``--prompts`` / ``--dataset`` 解析成一个每步采样的 prompt 池。

    为什么要采样而不是把整份列表交给 trainer：``train_step(prompts)`` 会把
    **整个列表**一次送进 rollout。下载来的数据集动辄几万条，一次生成会直接
    OOM —— 「数据集大小」与「每步 batch」必须是两个量。
    """
    from data.prompts import PromptSampler, load_prompt_dataset, read_prompt_file

    if args.prompts and args.dataset:
        raise ValueError("--prompts 与 --dataset 只能二选一")

    if args.dataset:
        prompts = load_prompt_dataset(
            args.dataset,
            split=args.dataset_split,
            field=args.prompt_field,
            limit=args.dataset_limit,
        )
        # 下载来的池子通常很大；不指定时每步只取 8 条。
        per_step = args.prompts_per_step or 8
        source = f"数据集 {args.dataset}"
    else:
        prompts = (
            read_prompt_file(args.prompts) if args.prompts else list(DEFAULT_PROMPTS)
        )
        # 本地文件保持旧行为：不传 --prompts-per-step 就整份送进 rollout。
        per_step = args.prompts_per_step
        source = args.prompts or "内置示例 prompt"

    sampler = PromptSampler(
        prompts, per_step=per_step, shuffle=True, seed=int(cfg.trainer.seed)
    )
    return source, sampler


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.steps is not None and args.steps < 1:
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
    # 不传 --steps 时以配置为准 —— 配置里写 1000 就该跑 1000，而不是退回到
    # 命令行里某个「冒烟默认值」。
    if args.steps is not None:
        cfg.trainer.total_steps = args.steps
    steps = int(cfg.trainer.total_steps)
    # run_name 决定 checkpoint 目录（base.yaml 里是 checkpoints/${trainer.run_name}）。
    # 没显式写名字就用算法名，顺带也给 wandb 一个可读的 run 名。
    if not cfg.trainer.get("run_name"):
        cfg.trainer.run_name = algorithm_path.stem
    kind = str(cfg.trainer.get("kind", "rl")).lower()

    if args.plan:
        import components  
        import data  

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

    prompt_sampler = None
    if kind == "rl":
        try:
            source, prompt_sampler = build_prompt_sampler(args, cfg)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"prompt 来源不可用：{exc}", file=sys.stderr)
            return 2
        print(f"  prompt：{source}，{prompt_sampler.describe()}")
    print("-" * 62)

    pbar = tqdm(
        range(1, steps + 1),
        desc=algorithm_path.name,
        unit="step",
        dynamic_ncols=True,
    )
    for step in pbar:
        if kind == "offline":
            metrics = trainer.train_step()
        else:
            metrics = trainer.train_step(prompt_sampler.next())
        if metrics.get("train/skipped_step") == 1.0:
            pbar.set_postfix_str("skipped")
            continue
        pbar.set_postfix(
            step=trainer.global_step,
            loss=f"{metrics['train/loss']:.4f}",
            grad=f"{metrics.get('train/grad_norm', float('nan')):.4f}",
        )
        if step % args.every == 0 or step == steps:
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
