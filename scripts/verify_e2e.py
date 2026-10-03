"""端到端验证脚本。

两个阶段，故意的
----------------
**阶段 1（纯 torch，永远运行）**
    把 ``configs/`` 下的**全部十个配置**各真的构造一次 trainer、各真的跑三步、
    再真的存读一次 checkpoint。用的是 ``tests/fixtures/`` 里那个约两万参数的
    微型 LM —— 它有真参数、走真 autograd，所以能验证梯度语义，而不只是
    「接口闭合」。CPU 上几秒钟跑完。

    十个配置一起跑是刻意的：这个脚本要回答的问题正是「这些变体真的能靠组件
    组装出来吗」。答案不该是一句文档里的话，而应该是十行绿色的输出。

    前八个是 RL 家族（``RLTrainer``，八相位），后两个是离线家族
    （``OfflineTrainer``，两相位）。它们共用同一份 ``base.yaml``、
    同一套装配函数，而模型清单各不相同 —— 离线那两行是其中最干净的一对：
    ``sft`` 不建 Reference、``dpo`` 建，差别只有损失项 ``requires`` 里的一行。

**阶段 2（真实 HF 模型，需要 transformers）**
    验证 ``src/models/`` 那一层。用 ``GPT2Config`` 在内存里构造一个 1 层 1 头的
    GPT-2（**不联网**），配一个手搓的离线分词器，跑 rollout -> 重算 logprob ->
    reference -> critic -> backward。

    为什么阶段 1 不能覆盖它：``src/models/`` 是唯一的 HF 实现层，而
    「rollout 的 logprob 与训练前向算出的不一致」这类问题只在真实
    ``generate`` 上才会出现。阶段 2 用 ``--require-hf`` 时，跳过会变成失败。

用法::

    python scripts/verify_e2e.py                  # 两阶段都跑，阶段 2 可跳过
    python scripts/verify_e2e.py --stage 1        # 只跑阶段 1
    python scripts/verify_e2e.py --require-hf     # 阶段 2 不许跳过
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

PROMPTS = ["甲", "乙", "丙", "丁"]

# 结果分类
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class Report:
    """收集检查结果，最后统一打印。"""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def record(self, stage: str, name: str, status: str, detail: str = "") -> None:
        self.rows.append((stage, name, f"{status}{' — ' + detail if detail else ''}"))
        marker = {PASS: "\033[32m✓\033[0m", FAIL: "\033[31m✗\033[0m", SKIP: "\033[33m-\033[0m"}[status]
        print(f"  {marker} {name}" + (f"  [{detail}]" if detail else ""))

    def check(self, stage: str, name: str, fn: Callable[[], None]) -> bool:
        try:
            fn()
        except Exception as exc:                      # noqa: BLE001 — 验证脚本要报告一切
            detail = f"{type(exc).__name__}: {exc}".splitlines()[0][:200]
            self.record(stage, name, FAIL, detail)
            traceback.print_exc(limit=3)
            return False
        self.record(stage, name, PASS)
        return True

    @property
    def failed(self) -> int:
        return sum(1 for _, _, text in self.rows if text.startswith(FAIL))

    @property
    def skipped(self) -> int:
        return sum(1 for _, _, text in self.rows if text.startswith(SKIP))


# =====================================================================
# 阶段 1：纯 torch
# =====================================================================
#: 配置名 -> (需要 Critic, 需要 Reference, 控制器个数)。
#:
#: 这是**装配计划**的期望值，而不是配置里写了什么 —— 两者是不同的问题。
#: 八个配置的 base.yaml 完全相同，critic / reference 两段配置在每个文件里都存在；
#: 到底哪几个模型会被真的加载，是 plan_assembly 从各组件的 requires 并集推出来的。
#: 这张表把这个推导结果写死，于是「改了某个组件的 requires 导致模型少加载了一个」
#: 会在这里变成一行红色的输出。
#:
#: dapo 的 Reference 是 False，最容易看错：它的 kl_k3 那一行**还在配置里**，
#: 只是 coef: 0.0 释放了 ref_logprobs 依赖，于是参考模型整个不被构建。
EXPECTED_ASSEMBLY = {
    "ppo": (True, True, 0),
    "ppo_kl_k1": (True, True, 0),
    "ppo_kl_k2": (True, True, 0),
    "ppo_adaptive_kl": (True, True, 1),
    "grpo": (False, True, 0),
    "dr_grpo": (False, True, 0),
    "dapo": (False, False, 0),
    "gspo": (False, True, 0),
}


def stage1(report: Report) -> None:
    print("\n=== 阶段 1：纯 torch 全链路（微型 LM，真自动微分）===")

    import torch

    import components  # noqa: F401 — 触发**真实**组件注册
    import data  # noqa: F401 — 触发数据集注册

    try:
        import tests.fixtures  # noqa: F401 — 触发 fixture 组件注册
    except ImportError:
        # tests/ 是开发用的、不随仓库分发。这一整个阶段（连同下面整个离线家族）
        # 都建立在那套微型组件上，缺了它没法跑 —— 干净跳过一条，
        # 而不是让十几项检查一起报 ModuleNotFoundError。
        report.record("阶段1", "纯 torch 全链路", SKIP, "tests/fixtures 不存在")
        return

    from core.config import find_config, load_config
    from engine.logic import plan_from_config
    from engine.rl_trainer import RLTrainer
    from tests.fixtures.configs import overlay_fixture

    names = list(EXPECTED_ASSEMBLY)
    # 两份配置：configs/ 里那份**原样**的（装配计划只由它决定，且不加载模型），
    # 以及把四个模型换成微型 fixture 的那份（真跑训练用）。
    #
    # 为什么不干脆都用 overlay 版：装配计划是从各组件的 requires / provides 推的，
    # 而模型组件根本不参与那个推导 —— 拿真实配置算计划，验的才是**要发布的
    # 那份配置**。overlay 只负责让下面真建模型的步骤不下载十几个 GB。
    configs = {name: load_config(find_config(name, ROOT / "configs")) for name in names}
    tiny = {name: overlay_fixture(cfg) for name, cfg in configs.items()}
    workdir = Path(tempfile.mkdtemp(prefix="rlt-verify-"))
    for name, cfg in tiny.items():
        # 别把 checkpoint 写进仓库目录。目录名用配置名而不是 advantage 类型 ——
        # 四个无 critic 的配置共用 broadcast，用类型名会互相覆盖。
        cfg.checkpointer.save_dir = str(workdir / name)

    for name in names:
        def check_assembly(name=name) -> None:
            want_critic, want_reference, want_controllers = EXPECTED_ASSEMBLY[name]

            # 先只看计划：一个字都不加载模型。
            plan = plan_from_config(configs[name])
            assert plan.need_critic is want_critic, (
                f"{name}: plan.need_critic = {plan.need_critic}，期望 {want_critic}"
            )
            assert plan.need_reference is want_reference, (
                f"{name}: plan.need_reference = {plan.need_reference}，期望 {want_reference}。"
                f"{'DAPO 的 kl_k3 是 coef=0，应当释放 ref_logprobs 依赖' if not want_reference else ''}"
            )
            assert not plan.conflicts, f"{name}: 计划里有冲突 {plan.conflicts}"

            # 再真的建一遍：这一步证明计划确实被 _assemble() 照做了。
            trainer = RLTrainer(tiny[name])
            assert (trainer.critic is not None) is want_critic, (
                f"{name}: Critic 的实际构建情况与装配计划不符"
            )
            assert (trainer.reference is not None) is want_reference, (
                f"{name}: Reference 的构建情况与期望不符（期望 {want_reference}）"
            )
            assert len(trainer.controllers) == want_controllers, (
                f"{name}: 控制器个数是 {len(trainer.controllers)}，期望 {want_controllers}"
            )

        report.check("阶段1", f"{name}.yaml：装配计划（critic/reference/controllers）", check_assembly)

    def check_shared_backbone_critic() -> None:
        """共享骨架的 critic：底座是 actor 底座的**复制**，不是别名。

        PPO 的标准做法 —— critic 从策略的初始权重出发，之后独立演化。
        这也是本仓库里最容易写错的一处：写成别名**不会报任何错**，
        只会让 value loss 的梯度悄悄改掉策略的网络权重。所以逐条断言
        「复制」的四个可观察后果。
        """
        import torch as _torch

        # overlay_fixture 已经把 hf_shared_value 映射成 fixture_tiny_shared_value ——
        # 这条检查要的正是那个映射，所以不在这里手工再指定一遍类型。
        cfg = tiny["ppo"]
        cfg.checkpointer.save_dir = str(workdir / "shared-critic")
        assert cfg.critic.type == "fixture_tiny_shared_value", (
            f"ppo 的 critic 应当被映射成共享骨架的 fixture，实际是 {cfg.critic.type}"
        )
        trainer = RLTrainer(cfg)

        actor_module, critic_module = trainer.actor.module, trainer.critic.module

        assert actor_module is not critic_module, (
            "critic 的底座与 actor 是同一个对象 —— deepcopy 没生效"
        )
        left, right = actor_module.state_dict(), critic_module.state_dict()
        assert left.keys() == right.keys(), (
            "critic 的底座结构与 actor 不同？它应当是同一份结构的复制"
        )
        assert all(_torch.equal(left[k], right[k]) for k in left), (
            "critic 的底座初始值应当与 actor 逐元素相等（从策略的权重出发）"
        )
        assert not (
            {id(p) for p in trainer.actor.parameters()}
            & {id(p) for p in trainer.critic.parameters()}
        ), "actor 与 critic 共享了 nn.Parameter 对象 —— 梯度会互相污染"

        overlap = set(trainer.actor.state_dict()) & set(trainer.critic.state_dict())
        assert not overlap, (
            f"actor 与 critic 的 checkpoint 键重叠：{sorted(overlap)}。"
            f"两者进的是同一份 state_dict，键撞上会静默覆盖。"
        )

    report.check(
        "阶段1", "共享骨架 critic（hf_shared_value 的 fixture 镜像）", check_shared_backbone_critic
    )

    trained: dict[str, tuple[RLTrainer, dict, dict]] = {}

    for name, cfg in tiny.items():
        def run(cfg=cfg, name=name) -> None:
            import torch as _torch

            trainer = RLTrainer(cfg)
            before = {n: p.detach().clone() for n, p in trainer.actor.named_parameters()}

            steps = [trainer.train_step(PROMPTS) for _ in range(3)]
            ran = [m for m in steps if m.get("train/skipped_step") != 1.0]

            # 0. 跳过的步必须是干净的：不推进步数、不报 loss。
            #    DAPO 的动态采样会丢组，丢空了就会走到这条路径 —— 那是**正确**
            #    行为（调用方该换一批 prompt 重采样），不是失败。但如果三步全被
            #    丢掉，这个配置实际上什么都没训，必须报出来。
            assert ran, (
                f"{name}: 三步全部被跳过（奖励处理链把样本全过滤了），"
                f"这个配置实际上什么都没训"
            )
            assert trainer.global_step == len(ran), (
                f"{name}: global_step = {trainer.global_step}，但真的跑过的步数是 {len(ran)}"
                f" —— 跳过的那几步不该记进步数"
            )
            for skipped in (m for m in steps if m.get("train/skipped_step") == 1.0):
                assert not [k for k in skipped if k.startswith("train/loss")], (
                    f"{name}: 跳过的步报了 loss 指标，但它根本没算过 loss"
                )

            metrics = ran[-1]

            # 1. 指标有限
            bad = {k: v for k, v in metrics.items() if not math.isfinite(v)}
            assert not bad, f"指标里有非有限值：{bad}"

            # 2. ratio 在第一个 mini-batch 精确为 1（logprob 重新前向的直接后果）
            ratio = metrics["train/ratio_first"]
            assert abs(ratio - 1.0) < 1e-5, f"train/ratio_first = {ratio}，应当精确为 1"

            # 3. 参数真的变了
            changed = [
                n for n, p in trainer.actor.named_parameters()
                if not _torch.equal(p.detach(), before[n])
            ]
            assert changed, "跑了 3 步之后 actor 参数一个都没变"

            trained[name] = (trainer, metrics, before)

        detail = "跑 3 步（装配 / loss 有限 / ratio=1 / 参数变化）"
        report.check("阶段1", f"{name}.yaml：{detail}", run)

    def check_checkpoint() -> None:
        import torch as _torch

        trainer, _, _ = trained["grpo"]
        expected = {n: p.detach().clone() for n, p in trainer.actor.named_parameters()}
        expected_step = trainer.global_step
        path = trainer.save_checkpoint()

        fresh = RLTrainer(tiny["grpo"])
        fresh.train_step(PROMPTS)                 # 先跑偏，才能证明恢复有效
        fresh.resume(path)

        assert fresh.global_step == expected_step, (
            f"global_step 没恢复：{fresh.global_step} != {expected_step}"
        )
        for name_, value in expected.items():
            assert _torch.equal(dict(fresh.actor.named_parameters())[name_].detach(), value), (
                f"参数 {name_} 没恢复"
            )

    report.check("阶段1", "checkpoint 存 → 读 → 续跑一致", check_checkpoint)

    def check_logprob_metrics() -> None:
        _, metrics, _ = trained["ppo"]
        # 3 步之后策略已经偏离，ratio_mean 不该还停在 1
        assert "train/ratio_mean" in metrics
        assert metrics["train/ratio_mean"] > 0.0

    report.check("阶段1", "PPO：ratio / logprob 漂移指标可用", check_logprob_metrics)

    def check_gspo_reports_sequence_level_metrics() -> None:
        """GSPO 报的必须是 ``seq_clip_frac`` 而不是 ``clip_frac``。

        这两个键名的分工是刻意的：token 占比与序列占比是两个量，同名会让人
        在曲线上一眼看错。所以这条检查键名本身，而不是值。
        """
        _, metrics, _ = trained["gspo"]
        assert "loss/gspo/seq_clip_frac" in metrics, "GSPO 应当上报按序列计的裁剪比例"
        assert "loss/gspo/max_seq_ratio" in metrics
        assert "loss/gspo/max_token_ratio" in metrics
        assert "loss/gspo/clip_frac" not in metrics, (
            "GSPO 不该上报按 token 计的 clip_frac —— 那是 policy_gradient 的量，"
            "同名会让人误以为两者可比"
        )

    report.check("阶段1", "GSPO：上报序列级裁剪指标（键名与 PPO 不同）",
                 check_gspo_reports_sequence_level_metrics)

    def check_gspo_rejects_token_level_advantages() -> None:
        """GSPO 配上 GAE 必须在 compute() 里报错，而不是悄悄压成均值。

        这是「配方对了但料不对」那一类错误的守卫：把 advantage 换成 gae 之后，
        损失照样算得出来、曲线照样好看，但优化的目标已经不是 GSPO 了。
        """
        import copy

        cfg = copy.deepcopy(tiny["gspo"])
        cfg.algorithm.advantage.type = "gae"
        cfg.algorithm.advantage.gamma = 0.99
        cfg.algorithm.advantage.lam = 0.95
        trainer = RLTrainer(cfg)          # 这一步会构建 Critic，正常

        try:
            trainer.train_step(PROMPTS)
        except ValueError as exc:
            assert "序列级" in str(exc), f"报错信息没说到点子上：{exc}"
        else:
            raise AssertionError(
                "GSPO 配上逐 token 的 GAE 优势却**没有报错** —— "
                "它会悄悄把优势压成一个平均值，算法已经不是 GSPO 了"
            )

    report.check("阶段1", "GSPO：配上 GAE 会报错（不静默退化）",
                 check_gspo_rejects_token_level_advantages)

    def check_adaptive_kl_moves_beta() -> None:
        """控制器真的把 β 写回了损失项 —— 自适应 KL 的端到端证据。

        只检查「日志里有个 beta」是不够的：那只证明控制器**报告**了一个数。
        要证明它**生效**，必须去看损失项自己的 ``coef`` —— 那个才是下一步
        真正被乘进 KL 项的值。
        """
        trainer, metrics, _ = trained["ppo_adaptive_kl"]
        key = "controller/kl_k3/beta"
        assert key in metrics, f"本步指标里没有 {key}，控制器可能没被调用"

        term = next(t for t, _ in trainer.loss_terms if t.name() == "kl_k3")
        assert term.coef == metrics[key], (
            f"控制器报告的 β = {metrics[key]} 与损失项实际的 coef = {term.coef} 不一致 —— "
            f"要么报告的是上一步的旧值，要么根本没写回去"
        )
        # 初始 coef 是 0.01；真调过才会变。
        assert term.coef != 0.01, (
            "β 一步都没动过。检查 target_kl 是否恰好等于观测 KL，"
            "或者控制器钩子是否被调用"
        )

    report.check("阶段1", "Adaptive-KL：β 真的写回了损失项", check_adaptive_kl_moves_beta)

    def check_dapo_works_without_reference() -> None:
        """DAPO 在没有 Reference 的情况下也能跑完 —— coef=0 那条路径的端到端验证。

        这条曾经是红的：coef=0 释放了 ref_logprobs 依赖，compute() 却还在
        无条件 require 它，于是配置合法、装配计划自洽、跑到运行期才炸。

        注意这条**不覆盖**空批次跳过路径：微型 fixture 模型生成的组内奖励
        互不相同，dynamic_filter 一个组都没丢。那条路径（``train/skipped_step``
        不推进 global_step）由 tests/test_train_step.py 的四个专项用例覆盖，
        那里可以直接换成 ``fixture_drop_all`` 来构造空批次。
        """
        trainer, metrics, _ = trained["dapo"]
        assert trainer.reference is None
        assert "loss/kl_k3" in metrics, "KL 项仍在组合里（coef=0 不等于被剔除）"
        assert metrics["loss/kl_k3"] == 0.0, "coef=0 的项对总损失的贡献必须是 0"

    report.check("阶段1", "DAPO：无 Reference 也能跑（coef=0 路径）",
                 check_dapo_works_without_reference)

    stage1_offline(report, workdir)


# =====================================================================
# 阶段 1（续）：离线家族
# =====================================================================
#: 配置名 -> (需要 Reference, 损失项是否要求整组)。
#:
#: 与上面那张表同一个道理：这是**装配计划**的期望值，不是配置里写了什么。
#: 两个配置用的是同一个训练器、同一份 base.yaml，reference 那一段在两边都挂着。
#: 建不建，取决于损失项的 requires 里有没有 ref_logprobs ——
#: cross_entropy 没有，dpo 有。
EXPECTED_OFFLINE = {
    "sft": (False, False),
    "dpo": (True, True),
}


def stage1_offline(report: Report, workdir: Path) -> None:
    print("\n  --- 离线家族（两相位：数据集 -> 训练）---")

    import torch

    import data  # noqa: F401 — 触发数据集注册
    from core import interfaces as F
    from engine.build import build_trainer
    from core.config import find_config, load_config
    from tests.fixtures.configs import overlay_fixture

    def config_for(name: str):
        # 与 RL 家族同样的闸门：actor / reference 换成微型 fixture，
        # 数据集与损失项保持真实（它们本来就是被验的对象）。
        cfg = overlay_fixture(load_config(find_config(name, ROOT / "configs")))
        cfg.checkpointer.save_dir = str(workdir / f"offline-{name}")
        return cfg

    names = list(EXPECTED_OFFLINE)
    configs = {name: config_for(name) for name in names}

    # ---- 装配 ----
    for name in names:
        def check_assembly(name=name) -> None:
            from engine.offline_trainer import OfflineTrainer

            want_reference, want_intact = EXPECTED_OFFLINE[name]
            trainer = build_trainer(configs[name])
            label = f"{name}.yaml"

            assert isinstance(trainer, OfflineTrainer), (
                f"{label}: trainer.kind 是 offline，却没被分派给 OfflineTrainer"
            )
            # 离线家族没有优势相位，于是也不该有 Critic —— 配置里那一段还在
            assert trainer.critic is None, f"{label}: 离线家族不该构建 Critic"
            assert (trainer.reference is not None) is want_reference, (
                f"{label}: Reference 的构建情况与期望不符（期望 {want_reference}）"
            )
            assert trainer.plan.need_critic is False
            # 打分与优势两个相位整个不存在 —— 这才是「隐式信号不需要奖励模型」
            assert F.REWARDS not in trainer.plan.providers, (
                f"{label}: 离线家族不该有 rewards 的提供方"
            )
            assert F.ADVANTAGES not in trainer.plan.providers
            assert trainer._needs_intact_groups() is want_intact, (
                f"{label}: 切分策略与损失的声明不符"
            )

        report.check("阶段1", f"{name}.yaml：装配计划（无 critic / 无 rewards / 切分策略）",
                     check_assembly)

    # ---- 训练 ----
    trained: dict[str, tuple] = {}

    for name in names:
        def run(name=name) -> None:
            trainer = build_trainer(configs[name])
            before = {n: p.detach().clone() for n, p in trainer.actor.named_parameters()}

            steps = [trainer.train_step() for _ in range(3)]
            metrics = steps[-1]

            bad = {k: v for k, v in metrics.items() if not math.isfinite(v)}
            assert not bad, f"指标里有非有限值：{bad}"
            assert trainer.global_step == 3, (
                f"{name}: global_step = {trainer.global_step}，三个 train_step 应当恰好三步"
            )

            # 数据游标：三步共消费 3 × batch_size 个**样本**（不是行）。
            # 取模的话「刚好走完一整圈」与「从没训过」就分不开了。
            expected_cursor = 3 * trainer.dataset.batch_size
            assert trainer._cursor == expected_cursor, (
                f"{name}: 游标是 {trainer._cursor}，期望 {expected_cursor}"
            )

            changed = [
                n for n, p in trainer.actor.named_parameters()
                if not torch.equal(p.detach(), before[n])
            ]
            assert changed, f"{name}: 跑了 3 步之后 actor 参数一个都没变"

            trained[name] = (trainer, metrics)

        report.check("阶段1",
                     f"{name}.yaml：跑 3 步（装配 / loss 有限 / 游标推进 / 参数变化）",
                     run)

    # ---- checkpoint 带上数据游标 ----
    def check_offline_checkpoint_carries_the_cursor() -> None:
        trainer, _ = trained["dpo"]
        expected_step = trainer.global_step
        expected_cursor = trainer._cursor
        expected = {n: p.detach().clone() for n, p in trainer.actor.named_parameters()}
        path = trainer.save_checkpoint()

        fresh = build_trainer(configs["dpo"])
        fresh.train_step()                        # 先跑偏，才能证明恢复有效
        fresh.resume(path)

        assert fresh.global_step == expected_step, (
            f"global_step 没恢复：{fresh.global_step} != {expected_step}"
        )
        assert fresh._cursor == expected_cursor, (
            f"数据游标没恢复：{fresh._cursor} != {expected_cursor}。"
            f"不存游标的话续跑会从数据集开头重训 —— 那是从 loss 曲线上看不出来的偏差。"
        )

        # 续跑取到的**下一批数据**必须与不中断时完全一致，而不只是游标数字相等
        resumed_next, _ = fresh.dataset.next_batch_indices(
            fresh._cursor, fresh.dataset.batch_size
        )
        expected_next, _ = trainer.dataset.next_batch_indices(
            trainer._cursor, trainer.dataset.batch_size
        )
        assert resumed_next == expected_next, (
            f"续跑取到的样本与不中断时不同：{resumed_next} != {expected_next}"
        )

        for key, value in expected.items():
            assert torch.equal(dict(fresh.actor.named_parameters())[key].detach(), value), (
                f"参数 {key} 没恢复"
            )

    report.check("阶段1", "离线：checkpoint 带上数据游标，续跑取到同一批数据",
                 check_offline_checkpoint_carries_the_cursor)

    # ---- DPO：它唯一能看出「在学」的量 ----
    def check_dpo_learns() -> None:
        """隐式奖励差 ``β·margin`` 随训练上升。

        只看 loss 下降是不够的 —— 一个 Δ 恒为 0 的退化实现也有一条平坦的
        曲线，而它其实什么都没学。这条盯的是**损失之外的证据**。
        """
        trainer = build_trainer(config_for("dpo"))
        gaps = [trainer.train_step()["loss/dpo/implicit_reward_gap"] for _ in range(12)]
        head, tail = sum(gaps[:4]) / 4, sum(gaps[-4:]) / 4
        assert tail > head, f"隐式奖励差没有上升：{head:.4f} -> {tail:.4f}"

    report.check("阶段1", "DPO：隐式奖励差随训练上升（loss 之外的证据）", check_dpo_learns)

    def check_dpo_reference_never_moves() -> None:
        """π_ref 的参数一步都不能动 —— 否则它就不再是「参考」。"""
        trainer = build_trainer(config_for("dpo"))
        before = [p.detach().clone() for p in trainer.reference.parameters()]
        for _ in range(3):
            trainer.train_step()
        after = list(trainer.reference.parameters())
        assert all(torch.equal(a, b) for a, b in zip(before, after)), (
            "Reference 的参数在训练中被改动了"
        )

    report.check("阶段1", "DPO：Reference 参数全程冻结", check_dpo_reference_never_moves)

    def check_wrong_dataset_is_caught_at_assembly() -> None:
        """拿 SFT 数据配 DPO 损失 —— 装配期就该报错。

        这正是 ``preference`` 没有被塞进 ``DATASET_OUTPUT_FIELDS`` 换来的：
        SFT 数据集不声称提供它，于是「料配错了」在**构建模型之前**就暴露。
        """
        cfg = config_for("dpo")
        cfg.data.type = "fixture_jsonl_sft"
        try:
            build_trainer(cfg)
        except ValueError as exc:
            assert "preference" in str(exc), f"报错信息没说到点子上：{exc}"
        else:
            raise AssertionError(
                "拿 SFT 数据配 DPO 损失却**没有报错** —— 它会一路跑到训练时才炸在"
                "「Batch 里没有 preference」上"
            )

    report.check("阶段1", "离线：数据与损失不匹配在装配期就被拦下",
                 check_wrong_dataset_is_caught_at_assembly)


# =====================================================================
# 阶段 2：真实 HF 模型（离线构造，不联网）
# =====================================================================
def build_offline_tokenizer(words: list[str]):
    """手搓一个不需要网络的词表。

    用 ``WordLevel`` + 空格切分，避开了 GPT-2 byte-level BPE 的字节映射 ——
    那需要把每个字节映射成 unicode 字符，手写极易出错。
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1}
    for word in words:
        vocab.setdefault(word, len(vocab))

    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>"
    )
    # decoder-only 生成必须左 padding，否则生成会从 pad 后面继续
    tokenizer.padding_side = "left"
    return tokenizer


def _exception_chain(exc: BaseException):
    """沿着 __cause__ / __context__ 走一遍，把整条异常链吐出来。

    transformers 的懒加载会把真实原因包在 ``ModuleNotFoundError: Could not import
    module 'GPT2LMHeadModel'`` 后面，只看最外层会得到完全错误的结论。
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def hf_modeling_unavailable_reason() -> str | None:
    """能不能真的构造一个 HF 模型？不能的话返回人话版原因。

    这里**只做探测，不做任何修复** —— 环境问题归环境，脚本不越界。
    """
    try:
        from transformers import GPT2Config, GPT2LMHeadModel  # noqa: F401
    except Exception as exc:                          # noqa: BLE001
        chain = list(_exception_chain(exc))
        if any("torchvision" in str(item) for item in chain):
            return (
                "本机的 torchvision 与 torch ABI 不匹配（torchvision::nms 不存在）。\n"
                "     transformers 的懒加载在导入任何建模类时都会连带 import "
                "image_utils -> torchvision，于是被这个环境问题挡住。\n"
                "     这是环境问题，不是框架问题：脚本不会去动你的依赖。\n"
                "     修好 torchvision（重装匹配版本）之后重跑即可。\n"
                f"     原始错误：{chain[-1]}"
            )
        return f"{type(exc).__name__}: {exc}"
    return None


def tiny_gpt2(vocab_size: int, seed: int, pad_id: int, eos_id: int):
    """在内存里构造一个 1 层 1 头的 GPT-2。不联网。"""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=64,
        n_ctx=64,
        n_embd=32,
        n_layer=1,
        n_head=1,
        # dropout 必须为 0：测试要断言 ratio == 1，
        # 开着 dropout 两次前向的输出不同，ratio 就不等于 1（那是 dropout 而非 bug）
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=pad_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )
    torch.manual_seed(seed)
    return GPT2LMHeadModel(config)


def stage2(report: Report, require_hf: bool) -> None:
    print("\n=== 阶段 2：真实 HF 模型（内存构造，不联网）===")

    try:
        import transformers  # noqa: F401
    except Exception as exc:                          # noqa: BLE001
        detail = f"transformers 不可用：{type(exc).__name__}: {exc}"
        if require_hf:
            report.record("阶段2", "导入 transformers", FAIL, detail)
        else:
            report.record("阶段2", "导入 transformers", SKIP, detail)
        return

    print(f"  transformers {transformers.__version__}")

    reason = hf_modeling_unavailable_reason()
    if reason is not None:
        status = FAIL if require_hf else SKIP
        report.record("阶段2", "构造 HF 建模类", status, reason)
        return

    report.check(
        "阶段2",
        "构造离线分词器",
        lambda: build_offline_tokenizer(["alpha", "beta", "gamma", "delta"]),
    )
    tokenizer = build_offline_tokenizer(["alpha", "beta", "gamma", "delta"])
    prompts = ["alpha beta", "gamma delta", "alpha gamma", "beta delta"]
    vocab_size = len(tokenizer)
    state: dict = {}

    def build_components() -> None:
        from models.hf_actor import HFPolicyActor
        from models.hf_critic import HFValueCritic
        from models.hf_reference import HFFrozenReference
        from models.hf_rollout import HFRolloutEngine

        policy = tiny_gpt2(vocab_size, seed=0, pad_id=tokenizer.pad_token_id,
                           eos_id=tokenizer.eos_token_id)
        reference_model = tiny_gpt2(vocab_size, seed=1234, pad_id=tokenizer.pad_token_id,
                                    eos_id=tokenizer.eos_token_id)
        critic_model = tiny_gpt2(vocab_size, seed=7, pad_id=tokenizer.pad_token_id,
                                 eos_id=tokenizer.eos_token_id)

        model_config = {"name_or_path": "tiny-gpt2", "dtype": "float32"}

        state["actor"] = HFPolicyActor(model_config=model_config, tokenizer=tokenizer,
                                       model=policy)
        state["rollout"] = HFRolloutEngine(model=policy, tokenizer=tokenizer,
                                           num_generations=2, max_new_tokens=4,
                                           temperature=1.0, top_p=1.0)
        state["reference"] = HFFrozenReference(model_config=model_config,
                                               tokenizer=tokenizer, model=reference_model)
        state["critic"] = HFValueCritic(model_config=model_config, tokenizer=tokenizer,
                                        model=critic_model)

    if not report.check("阶段2", "构造 actor / rollout / reference / critic", build_components):
        return

    def rollout_and_align() -> None:
        batch = state["rollout"].generate(prompts)
        # 契约：行序等价于 repeat_interleave(prompts, num_generations)
        assert len(batch) == len(prompts) * 2, f"行数应为 {len(prompts) * 2}，实得 {len(batch)}"
        assert batch.non_tensors["prompt_texts"] == [
            p for p in prompts for _ in range(2)
        ], "行序不符合 repeat_interleave 契约"
        assert batch["group_ids"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
        # response_mask 只落在生成 token 上。
        #
        # 空行是**合法**的：policy 完全可能第一个 token 就吐 eos，那一行确实
        # 没有 response token（from_rollout 的 `if lr:` 与 masked_mean 的 eps
        # 都为此留了路）。所以只要求整批非空 —— 随机初始化的微型模型上，一行
        # 立刻吐 eos 是很常见的事，要求每行非空等于让这条断言随机地红。
        #
        # 真正抓对齐错误的是下面两条结构断言。
        import torch

        mask = batch["response_mask"]
        assert mask.sum() > 0, "整批一个 response token 都没有"

        # 左 padding 的契约：prompt 右对齐到同一列，所以**所有** response 都
        # 从 max_prompt 起、连续写满自己的长度。prompt 列上不该有 response，
        # 行与行之间也不该错开。
        #
        # 逐列求和（`(mask.sum(dim=0) > 0).all()`）在这里是**恒真**的 ——
        # prompt 列本来就该是 0，那条断言无论对齐对不对都会挂。改成按各行长度
        # 重建整张 mask 再比对：错开的行、中间断开的行都会在这一步露出来。
        max_prompt = max(batch.non_tensors["prompt_lengths"])
        assert mask[:, :max_prompt].sum() == 0, (
            f"有 response token 落在 prompt 列上（前 {max_prompt} 列）—— 对齐错了"
        )
        expected = torch.zeros_like(mask)
        for row, length in enumerate(mask.sum(dim=1).tolist()):
            expected[row, max_prompt : max_prompt + length] = 1
        assert torch.equal(mask, expected), (
            "response_mask 不是「从 max_prompt 起、连续」的块 —— "
            "有行从别的列开始，或者中间断了"
        )
        state["batch"] = batch

    report.check("阶段2", "rollout：行序契约 / response_mask 对齐", rollout_and_align)

    def recomputed_logprob_matches() -> None:
        """阶段 2 存在的理由：验证 HF 上「重新前向算 logprob」确实给出 ratio == 1。"""
        import torch

        from core.tensor_ops import masked_mean

        batch = state["batch"].clone()
        fresh = state["rollout"].generate(prompts)
        # 同一次生成的 logprob 必须与随后的 actor 前向完全一致
        batch = state["actor"].logprobs(batch)
        delta = batch["curr_logprobs"] - batch["rollout_logprobs"]
        ratio = float(masked_mean(torch.exp(delta), batch["response_mask"]).detach())
        assert abs(ratio - 1.0) < 1e-4, (
            f"ratio = {ratio}，应当精确为 1。"
            f"偏离说明 rollout 的 logprob 不是重新前向算的。"
        )
        assert fresh["rollout_logprobs"].shape == batch["rollout_logprobs"].shape

    report.check("阶段2", "rollout_logprobs 与 actor 前向一致（ratio == 1）",
                 recomputed_logprob_matches)

    def reference_is_independent() -> None:
        import torch

        batch = state["reference"].logprobs(state["batch"].clone())
        ref = batch["ref_logprobs"]
        assert not ref.requires_grad, "参考模型的 logprob 必须是 detached 的"
        assert ref.shape == batch["rollout_logprobs"].shape
        assert not torch.allclose(ref, batch["rollout_logprobs"], atol=1e-4), (
            "ref_logprobs 与 rollout_logprobs 完全相同 —— 参考模型可能错误地共享了 "
            "策略权重（那样 KL 惩罚会变成自己减自己）"
        )

    report.check("阶段2", "reference：独立副本、detached、与策略不同", reference_is_independent)

    def critic_gradient_semantics() -> None:
        import torch

        critic = state["critic"]

        batch = state["batch"].clone()
        batch = critic.forward_values(batch, detach=True)
        assert not batch["rollout_values"].requires_grad, "行为策略的价值必须是 detached"

        for param in critic.parameters():
            param.grad = None

        batch = critic.forward_values(batch, detach=False)
        assert batch["values"].requires_grad, "当前 critic 的价值必须可微"

        loss = (batch["values"] ** 2).mean()
        loss.backward()

        # 要比的是 ``.grad``，不是参数值：backward 只填梯度，不碰权重，所以
        # 「反向前后参数值不同」这条断言恒假 —— 它执行不了也通过不了。真正要
        # 证明的是「value loss 的梯度确实落到了 critic 自己的参数上」，
        # 与上面 actor 那条同一套写法。
        grads = [p.grad for p in critic.parameters() if p.grad is not None]
        assert grads, "critic 一个参数都没拿到梯度"
        assert any(float(g.abs().sum()) > 0 for g in grads), "critic 的梯度全为 0"
        assert all(torch.isfinite(g).all() for g in grads), "critic 的梯度里有 NaN/Inf"

    report.check("阶段2", "critic：rollout_values 与 values 的梯度语义分开", critic_gradient_semantics)

    def actor_backward_moves_params() -> None:
        import torch

        from core.tensor_ops import masked_mean

        actor = state["actor"]
        for param in actor.parameters():
            param.grad = None

        batch = state["batch"].clone()
        batch = actor.logprobs(batch)
        loss = -masked_mean(batch["curr_logprobs"] * batch["rollout_logprobs"].detach(),
                            batch["response_mask"])
        loss.backward()

        grads = [p.grad for p in actor.parameters() if p.grad is not None]
        assert grads, "actor 一个参数都没拿到梯度"
        assert any(float(g.abs().sum()) > 0 for g in grads), "actor 的梯度全为 0"
        assert all(torch.isfinite(g).all() for g in grads), "actor 的梯度里有 NaN/Inf"

    report.check("阶段2", "actor：梯度非零、有限，能穿过 HF 模型", actor_backward_moves_params)

    def rollout_no_recompute_is_refused() -> None:
        from models.hf_rollout import HFRolloutEngine

        engine = HFRolloutEngine(model=state["rollout"]._model, tokenizer=tokenizer,
                                 num_generations=1, max_new_tokens=2,
                                 recompute_logprobs=False)
        try:
            engine.generate(["alpha beta"])
        except NotImplementedError as exc:
            assert "ratio" in str(exc) or "logprob" in str(exc)
            return
        raise AssertionError("recompute_logprobs=False 应当明确拒绝，而不是悄悄用 generate 的 scores")

    report.check("阶段2", "recompute_logprobs=False 被明确拒绝", rollout_no_recompute_is_refused)


# =====================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="端到端验证")
    parser.add_argument("--stage", choices=("1", "2", "all"), default="all")
    parser.add_argument(
        "--require-hf",
        action="store_true",
        help="阶段 2 不许跳过（CI 上用；本地缺 transformers 时不要加）",
    )
    args = parser.parse_args()

    report = Report()
    if args.stage in ("1", "all"):
        stage1(report)
    if args.stage in ("2", "all"):
        try:
            stage2(report, require_hf=args.require_hf)
        except Exception as exc:                       # noqa: BLE001
            report.record("阶段2", "执行阶段 2", FAIL, f"{type(exc).__name__}: {exc}")
            traceback.print_exc(limit=5)

    print("\n" + "=" * 60)
    if report.failed:
        print(f"验证未通过：{report.failed} 项失败，{report.skipped} 项跳过")
        return 1
    print(f"验证通过（{report.skipped} 项跳过）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
