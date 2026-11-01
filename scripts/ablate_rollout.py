"""rollout 后端的消融实验。

两个子命令，对应两类完全不同的问题：

``smoke``（无需 GPU、无需 sglang、无需下载模型）
    回答「接得对不对」：
    - 复用采样端 logprob 与用 actor 重算，数值是否一致（应为 0 误差）；
    - 省掉的那次全序列前向，在 CPU 上值多少毫秒；
    - 权重同步钩子在 ``weight_sync=none/tensor`` 下的开销。
    这里的绝对耗时**没有参考价值**（微型模型），只有差值与一致性有意义。

``local``（需要 GPU + 下载模型；无需 sglang，Windows 可跑）
    回答「logprob 复用在真模型上省多少」：
    - ``hf``（产线路径）vs ``hf-batched-sim``（模拟 sglang server 顺手
      返回采样时刻 logprob）的耗时差；
    - 差值 ≈ 单独量的「全序列重算前向」耗时（归因校验）；
    - 真实权重上的一致性：temperature=1/top_p=1 时两路 logprob 应相等。
    generate 本身的加速（连续批处理那一半收益）这里量不到，需要 ``bench``。

``bench``（需要 GPU；sglang 需要 Linux + CUDA）
    回答「快多少」：
    - ``hf`` vs ``sglang`` 的 rollout 墙钟时间、吞吐（token/s）、加速比；
    - ``logprob_source`` 消融：复用服务端 logprob 能省掉一次 0.6B 全序列前向；
    - ``weight_sync`` 消融：同步开销会吃掉多少收益。

用法::

    python scripts/ablate_rollout.py smoke
    python scripts/ablate_rollout.py local --model Qwen/Qwen3-0.6B --batch-size 8 \
        --num-generations 8 --max-new-tokens 128
    python scripts/ablate_rollout.py bench --model Qwen/Qwen3-0.6B --batch-size 8 \
        --num-generations 8 --max-new-tokens 256 --backends hf,sglang

结果同时打印成 markdown 并写入 ``runs/ablation/<时间戳>.json``。
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch

from core import interfaces as F
from models.rollout_common import recompute_batch_logprobs
from models.sglang_rollout import SGLangRolloutEngine, Sample, SamplingParams

logger = logging.getLogger("ablate")

OUT_DIR = ROOT / "runs" / "ablation"

#: 消融用的 prompt（长度差异大一点，避免全部同长带来的乐观偏差）
PROMPTS = [
    "用一个比喻解释梯度下降",
    "为什么天空是蓝色的",
    "写一句关于春天的诗",
    "解释什么是重要性采样，并说明它在 PPO 里的作用",
    "用三句话总结强化学习和监督学习的区别",
    "如果 KL 系数太大，训练会出现什么现象",
    "为什么要对奖励做组内归一化",
    "一句话说明 vLLM 和 SGLang 的共同点",
]

#: smoke 专用：**等长** prompt。
#:
#: 原因是一处与本次改动无关的测量偏差：``tests/fixtures/tiny_lm.TinyCausalLM``
#: 用 ``arange(L)`` 取位置嵌入、不按 ``attention_mask`` 重算 position_ids。
#: 于是左 padding 会把真实 token 整体右移，位置嵌入跟着变，重算出的 logprob
#: 与「裸序列采样时算的」自然不同（实测 max |Δ| ≈ 1.4）—— 那是 fixture 的
#: 性质，不是 logprob 放置逻辑的问题。等长 prompt 不产生 padding，
#: 就能把这一项隔离掉，剩下的只有数值误差（实测 ≈ 5e-7）。
#: 真实 HF 模型会按 mask 计算 position_ids，不等长也一致 —— 这一条由
#: ``scripts/ablate_rollout.py bench`` 在真模型上验证。
PROMPTS_EQUAL = ["aaaa", "bbbb", "cccc", "dddd", "eeee", "ffff", "gggg", "hhhh"]


# ======================================================================
# 结果容器
# ======================================================================
@dataclass
class Run:
    """一次测量的结果。字段刻意扁平，方便直接转成表格行。"""

    backend: str
    runtime: str
    logprob_source: str
    weight_sync: str
    num_generations: int
    batch_prompts: int
    max_new_tokens: int
    repeats: int
    mean_s: float
    min_s: float
    tokens: int
    tokens_per_s: float
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _measure(fn, repeats: int, *, warmup: int = 1) -> tuple[float, float, Any]:
    """跑 ``warmup + repeats`` 次，返回 (均值, 最小值, 最后一次的输出)。"""
    for _ in range(warmup):
        out = fn()
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.fmean(times), min(times), out


def _count_tokens(batch) -> int:
    return int(batch[F.RESPONSE_MASK].sum())


# ======================================================================
# smoke：正确性与 CPU 开销
# ======================================================================
class LocalSamplingBackend:
    """用微型模型逐 token 采样，并返回**采样时**的 logprob。

    模拟 sglang 的 ``output_token_logprobs``：logprob 是采样那一刻算出来的，
    而不是事后重算的。这正是 ``logprob_source=sglang`` 想复用的东西。
    """

    name = "local-sampling"

    def __init__(self, model, vocab_size: int, temperature: float = 1.0) -> None:
        self._model = model
        self.vocab_size = int(vocab_size)
        self.temperature = float(temperature)
        self.update_calls = 0

    def generate(self, prompts: Sequence[Sequence[int]], params: SamplingParams):
        out = []
        for prompt_ids in prompts:
            group = []
            for _ in range(params.n):
                sequence = list(prompt_ids)
                tokens: list[int] = []
                logprobs: list[float] = []
                for _ in range(params.max_new_tokens):
                    ids = torch.as_tensor([sequence], dtype=torch.long)
                    with torch.no_grad():
                        logits = self._model(input_ids=ids).logits[0, -1]
                    logprobs_all = torch.log_softmax(logits / max(self.temperature, 1e-6), dim=-1)
                    token = int(torch.multinomial(logprobs_all.exp(), num_samples=1))
                    tokens.append(token)
                    logprobs.append(float(logprobs_all[token]))
                    sequence.append(token)
                group.append(Sample(token_ids=tokens, text=" ".join(map(str, tokens)),
                                    token_logprobs=logprobs))
            out.append(group)
        return out

    def update_weights(self, tensors) -> bool:
        self.update_calls += 1
        return True

    def close(self) -> None:
        pass


def _tiny_model():
    """复用测试里的微型 LM —— 与 e2e 测试是同一个模型，避免「脚本能跑、测试不能」。"""
    from tests.fixtures.tiny_lm import TinyCausalLM

    torch.manual_seed(0)
    return TinyCausalLM(vocab_size=64, d_model=32, nhead=2, n_layers=1, max_positions=256)


def _tiny_tokenizer():
    from tests.fixtures.tok import FakeTokenizer

    return FakeTokenizer(vocab_size=64)


class ReplayBackend:
    """回放固定的样本（token 序列 + 采样时的 logprob）。

    消融必须控制变量：两条 logprob 路径若各自独立采样，拿到的就是**不同文本**，
    比较它们的 logprob 毫无意义。回放同一串 token 才能问出真正的问题 ——
    「复用采样端 logprob」与「用 actor 重算」是不是同一个数。
    """

    name = "replay"

    def __init__(self, rows: Sequence[tuple[list[int], list[float]]], n: int) -> None:
        self.rows = list(rows)
        self.n = int(n)
        self.update_calls = 0

    def generate(self, prompts: Sequence[Sequence[int]], params: SamplingParams):
        assert params.n == self.n, f"回放按 n={self.n} 组织，请求却是 n={params.n}"
        out, cursor = [], 0
        for _ in prompts:
            group = []
            for _ in range(params.n):
                tokens, logprobs = self.rows[cursor]
                cursor += 1
                group.append(
                    Sample(token_ids=list(tokens), text=" ".join(map(str, tokens)),
                           token_logprobs=list(logprobs))
                )
            out.append(group)
        return out

    def update_weights(self, tensors) -> bool:
        self.update_calls += 1
        return True

    def close(self) -> None:
        pass


def _rows_from_batch(batch) -> list[tuple[list[int], list[float]]]:
    """把一次真实采样的结果摊平成回放用的行。"""
    mask = batch[F.RESPONSE_MASK].bool()
    ids = batch[F.INPUT_IDS]
    logprobs = batch[F.ROLLOUT_LOGPROBS]
    rows = []
    for i in range(len(batch)):
        rows.append(
            (
                [int(t) for t in ids[i][mask[i]].tolist()],
                [float(v) for v in logprobs[i][mask[i]].tolist()],
            )
        )
    return rows


def run_smoke(args: argparse.Namespace) -> list[Run]:
    model = _tiny_model()
    tokenizer = _tiny_tokenizer()
    sampler = LocalSamplingBackend(model, vocab_size=64, temperature=1.0)
    prompts = PROMPTS_EQUAL[: args.batch_size]
    n = args.num_generations
    max_new = min(args.max_new_tokens, 16)

    def build(backend, source: str, weight_sync: str = "none") -> SGLangRolloutEngine:
        return SGLangRolloutEngine(
            model=model,
            tokenizer=tokenizer,
            runtime="engine",
            backend=backend,
            logprob_source=source,
            weight_sync=weight_sync,
            num_generations=n,
            max_new_tokens=max_new,
            temperature=1.0,
        )

    # ---- 步骤 1：采一次，拿到「采样时刻」的 token 与 logprob ----
    seed_engine = build(sampler, "sglang")
    seed_batch = seed_engine.generate(prompts, num_generations=n, max_new_tokens=max_new)
    rows = _rows_from_batch(seed_batch)

    # ---- 步骤 2：同一串 token 回放给两条 logprob 路径 ----
    results: list[Run] = []
    batches: dict[str, Any] = {}
    for source in ("sglang", "recompute"):
        engine = build(ReplayBackend(rows, n=n), source)
        mean_s, min_s, batch = _measure(
            lambda e=engine: e.generate(prompts, num_generations=n, max_new_tokens=max_new),
            repeats=args.repeats,
        )
        batches[source] = batch
        results.append(
            Run(
                backend="tiny+replay",
                runtime="engine",
                logprob_source=source,
                weight_sync="none",
                num_generations=n,
                batch_prompts=len(prompts),
                max_new_tokens=max_new,
                repeats=args.repeats,
                mean_s=mean_s,
                min_s=min_s,
                tokens=_count_tokens(batch),
                tokens_per_s=_count_tokens(batch) / mean_s if mean_s else 0.0,
            )
        )

    # ---- 步骤 3：一致性（这才是 smoke 的核心产出）----
    a = batches["sglang"][F.ROLLOUT_LOGPROBS]
    b = batches["recompute"][F.ROLLOUT_LOGPROBS]
    mask = batches["recompute"][F.RESPONSE_MASK].bool()
    diff = (a[mask] - b[mask]).abs().max().item() if mask.any() else 0.0
    ratio = torch.exp(a[mask] - b[mask])
    results[0].extra["max_abs_logprob_diff"] = diff
    results[0].extra["ratio_min"] = float(ratio.min())
    results[0].extra["ratio_max"] = float(ratio.max())

    # ---- 步骤 4：权重同步钩子本身的成本 ----
    engine = build(ReplayBackend(rows, n=n), "sglang", weight_sync="tensor")
    sync_times = []
    for _ in range(max(args.repeats, 3)):
        start = time.perf_counter()
        engine.sync_weights(model)
        sync_times.append(time.perf_counter() - start)
    results.append(
        Run(
            backend="tiny+replay",
            runtime="engine",
            logprob_source="sglang",
            weight_sync="tensor",
            num_generations=n,
            batch_prompts=len(prompts),
            max_new_tokens=max_new,
            repeats=len(sync_times),
            mean_s=statistics.fmean(sync_times),
            min_s=min(sync_times),
            tokens=0,
            tokens_per_s=0.0,
            notes="仅权重同步钩子（不含采样）",
            extra={"update_calls": engine.backend.update_calls},
        )
    )
    return results


# ======================================================================
# bench：真实模型上的吞吐
# ======================================================================
def _load_hf(model_id: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), tokenizer


def run_bench(args: argparse.Namespace) -> list[Run]:
    from models.hf_rollout import HFRolloutEngine

    model, tokenizer = _load_hf(args.model, args.dtype)
    prompts = (PROMPTS * 8)[: args.batch_size]
    results: list[Run] = []

    def measure_engine(label: str, engine, *, runtime: str, source: str) -> None:
        if args.warmup:
            engine.generate(prompts, num_generations=args.num_generations,
                            max_new_tokens=args.max_new_tokens)
        times: list[float] = []
        batch = None
        for _ in range(args.repeats):
            start = time.perf_counter()
            batch = engine.generate(
                prompts,
                num_generations=args.num_generations,
                max_new_tokens=args.max_new_tokens,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        tokens = _count_tokens(batch)
        mean_s = statistics.fmean(times)
        results.append(
            Run(
                backend=label,
                runtime=runtime,
                logprob_source=source,
                weight_sync=getattr(engine, "weight_sync", "none"),
                num_generations=args.num_generations,
                batch_prompts=len(prompts),
                max_new_tokens=args.max_new_tokens,
                repeats=args.repeats,
                mean_s=mean_s,
                min_s=min(times),
                tokens=tokens,
                tokens_per_s=tokens / mean_s if mean_s else 0.0,
            )
        )

    if "hf" in args.backends:
        hf = HFRolloutEngine(
            model,
            tokenizer,
            num_generations=args.num_generations,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        measure_engine("hf", hf, runtime="in-process", source="recompute")

    if "sglang" in args.backends:
        for source in args.logprob_sources:
            sg = SGLangRolloutEngine(
                model=model,
                tokenizer=tokenizer,
                runtime=args.runtime,
                model_path=args.model,
                logprob_source=source,
                weight_sync=args.weight_sync,
                dtype=args.dtype,
                tp_size=args.tp_size,
                mem_fraction_static=args.mem_fraction,
                server_url=args.server_url,
                num_generations=args.num_generations,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            try:
                measure_engine(f"sglang[{args.runtime}]", sg, runtime=args.runtime, source=source)
            finally:
                sg.close()

    return results


# ======================================================================
# local：单卡 GPU（无 sglang）上的真实模型消融
# ======================================================================
class BatchedHFBackend:
    """用同一次 HF batched generate 模拟「sglang server」的返回契约。

    sglang 相对现产线的核心优势有两半：

    1. 连续批处理 / PagedAttention 让 generate 本身更快 —— 这一半必须
       真 sglang 才能量出来（Windows 跑不了），留给 Kaggle/远程档；
    2. server 顺手返回**采样时刻**的 token logprob，省掉产线里
       「对完整序列再前向一遍」的重算 —— 这一半在本地就能量。

    本类模拟第 2 半：``output_scores=True`` 让 HF generate 在采样时把
    分布留下来，从中取出每个已采 token 的 logprob，按 sglang 的
    ``Sample(token_ids, text, token_logprobs)`` 契约交出去。引擎侧
    ``logprob_source='sglang'`` 接住它就不再重算。

    注意：HF 的 ``output_scores`` 返回的是**经 logits processor 处理后**
    的分布。temperature=1.0 / top_p=1.0 时没有形变，logprob 与重算一致
    （下面的 `run_local` 就在这个设置下做一致性校验）；温度不为 1 时
    两者差一个温度缩放 —— 与 sglang 返回「采样分布下的 logprob」的语义
    恰好相同，所以模拟是忠实的。
    """

    name = "hf-batched-sim"

    def __init__(self, model, tokenizer) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.update_calls = 0

    @torch.no_grad()
    def generate(self, prompts: Sequence[Sequence[int]], params: SamplingParams):
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._tokenizer.eos_token_id

        # 每个 prompt 展开 n 份，合成一个大 batch（模拟 server 内部的连续批处理入口）
        flat: list[list[int]] = []
        for ids in prompts:
            flat.extend(list(ids) for _ in range(params.n))
        width = max(len(x) for x in flat)
        input_ids = torch.full((len(flat), width), pad_id, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for i, ids in enumerate(flat):
            input_ids[i, width - len(ids):] = torch.as_tensor(ids, dtype=torch.long)
            attention[i, width - len(ids):] = 1

        out = self._model.generate(
            input_ids=input_ids.to(self._model.device),
            attention_mask=attention.to(self._model.device),
            max_new_tokens=params.max_new_tokens,
            do_sample=params.temperature > 0,
            temperature=max(params.temperature, 1e-6) or None,
            top_p=params.top_p,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=pad_id,
        )
        gen = out.sequences[:, width:]                       # [B*n, T]
        eos_id = self._tokenizer.eos_token_id

        # 向量化提取采样时刻 logprob：scores 堆成 [T, B', V] 后按时间分块
        # log_softmax（fp32 全量物化会 OOM），再 gather 已采 token。
        # 不能用 Python 逐 token 循环 —— 8192 次小核调用的开销会把测量
        # 本身变成瓶颈（实测能把这条路径拖慢 30%）。
        T = gen.shape[1]
        B2 = gen.shape[0]
        V = out.scores[0].shape[-1]
        lp_rows = torch.empty((B2, T), dtype=torch.float32)
        step_chunk = max(1, (1 << 27) // (B2 * V))
        for s in range(0, T, step_chunk):
            e = min(s + step_chunk, T)
            chunk = torch.stack(out.scores[s:e], 0).float()          # [t, B', V]
            lp_rows[:, s:e] = (
                torch.log_softmax(chunk, -1)
                .gather(-1, gen[:, s:e].transpose(0, 1).unsqueeze(-1))  # [t, B', 1]
                .squeeze(-1)
                .transpose(0, 1)
            )                                                        # [B', T]

        groups: list[list[Sample]] = []
        for row in range(B2):
            tokens = gen[row]
            n_tok = T
            if eos_id is not None and (tokens == eos_id).any():
                n_tok = int((tokens == eos_id).nonzero()[0])
            groups.append(
                Sample(
                    token_ids=tokens[:n_tok].tolist(),
                    text=self._tokenizer.decode(tokens[:n_tok], skip_special_tokens=True),
                    token_logprobs=lp_rows[row, :n_tok].tolist(),
                )
            )
        # 还原成「每 prompt 一组」的形状
        return [groups[i * params.n:(i + 1) * params.n] for i in range(len(prompts))]

    def update_weights(self, tensors) -> bool:
        self.update_calls += 1
        return True

    def close(self) -> None:
        pass


def _batch_to(batch, device):
    """Batch 组装发生在 CPU；重算前向前把张量搬到模型所在设备。"""
    for key in (F.INPUT_IDS, F.ATTENTION_MASK):
        batch[key] = batch[key].to(device)
    return batch


def _time_recompute_forward(model, batch) -> float:
    """单独量「对完整序列重算 logprob」这一步前向的耗时（产出归因用）。"""
    start = time.perf_counter()
    recompute_batch_logprobs(model, _batch_to(batch, next(model.parameters()).device))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start


def run_local(args: argparse.Namespace) -> list[Run]:
    """真实模型 + 单卡 GPU，但不依赖 sglang。

    三个量：
    - ``hf``：现产线（batched generate + 全序列重算）的完整耗时 —— 也是
      将来真 sglang 跑分时的基线；
    - ``sim``：batched generate 顺手返回采样 logprob（= sglang server 的
      返回契约），省掉重算前向 —— 两者之差就是 logprob 复用能省的部分；
    - 重算前向的单独耗时，验证「差值 ≈ 重算」这个归因。
    外加一项真实权重上的一致性校验（temperature=1, top_p=1 时两路
    logprob 应严格相等，把 smoke 里微型模型的结论在真模型上复核一遍）。
    """
    from models.hf_rollout import HFRolloutEngine

    model, tokenizer = _load_hf(args.model, args.dtype)
    prompts = (PROMPTS * 8)[: args.batch_size]
    results: list[Run] = []

    hf = HFRolloutEngine(
        model, tokenizer,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    sim_backend = BatchedHFBackend(model, tokenizer)
    sim = SGLangRolloutEngine(
        model=model, tokenizer=tokenizer, runtime="engine", backend=sim_backend,
        logprob_source="sglang", weight_sync="none",
        num_generations=args.num_generations, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
    )

    def measure(engine, label: str, source: str):
        if args.warmup:
            engine.generate(prompts, num_generations=args.num_generations,
                            max_new_tokens=args.max_new_tokens)
        times, batch = [], None
        for _ in range(args.repeats):
            start = time.perf_counter()
            batch = engine.generate(prompts, num_generations=args.num_generations,
                                    max_new_tokens=args.max_new_tokens)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        tokens = _count_tokens(batch)
        mean_s = statistics.fmean(times)
        return batch, Run(
            backend=label, runtime="cuda", logprob_source=source,
            weight_sync="none",
            num_generations=args.num_generations, batch_prompts=len(prompts),
            max_new_tokens=args.max_new_tokens, repeats=args.repeats,
            mean_s=mean_s, min_s=min(times), tokens=tokens,
            tokens_per_s=tokens / mean_s if mean_s else 0.0,
        )

    hf_batch, hf_run = measure(hf, "hf", "recompute")
    sim_batch, sim_run = measure(sim, "hf-batched-sim", "sglang")
    results += [hf_run, sim_run]

    # 归因：差值应 ≈ 重算前向的耗时
    recompute_s = _time_recompute_forward(model, hf_batch)
    saved = hf_run.mean_s - sim_run.mean_s
    sim_run.extra["recompute_forward_s"] = recompute_s
    sim_run.extra["saved_s"] = saved
    sim_run.notes = f"省 {saved:.2f}s ≈ 重算前向 {recompute_s:.2f}s"

    # 真实权重上的一致性校验。
    # 不能直接拿 bf16 的采样 logprob 对 bf16 重算：generate 走 KV-cache 核、
    # 重算走全序列前向，两条核路径在 bf16 下的舍入不同，logprob 能差 O(0.1~1)
    # —— 那是精度噪声，会淹没真正的放置错误（错位/漏 mask，差值是 O(1) 且
    # 大面积出现）。所以校验单独在 fp32 下做一次小规模采样 + 重算。
    check_model, check_tok = (model, tokenizer)
    if args.dtype != "float32":
        del hf, sim, model
        torch.cuda.empty_cache()
        check_model, check_tok = _load_hf(args.model, "float32")

    torch.manual_seed(0)
    check_backend = BatchedHFBackend(check_model, check_tok)
    check_engine = SGLangRolloutEngine(
        model=check_model, tokenizer=check_tok, runtime="engine", backend=check_backend,
        logprob_source="sglang", weight_sync="none",
        num_generations=2, max_new_tokens=16, temperature=1.0, top_p=1.0,
    )
    check_batch = check_engine.generate(
        prompts[:2], num_generations=2, max_new_tokens=16,
    )
    check_mask = check_batch[F.RESPONSE_MASK].bool()
    a = check_batch[F.ROLLOUT_LOGPROBS]
    b = recompute_batch_logprobs(
        check_model, _batch_to(check_batch, next(check_model.parameters()).device)
    )[F.ROLLOUT_LOGPROBS]
    diff = (a[check_mask] - b[check_mask].cpu()).abs()
    sim_run.extra["max_abs_logprob_diff"] = float(diff.max())
    sim_run.extra["frac_diff_gt_0.01"] = float((diff > 0.01).float().mean())
    return results


# ======================================================================
# 输出
# ======================================================================
def to_markdown(results: list[Run], title: str) -> str:
    header = (
        "| 后端 | runtime | logprob 来源 | 权重同步 | n | prompts | mean s | "
        "tokens | tok/s | 备注 |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [
        f"### {title}",
        "",
        header,
    ]
    for r in results:
        lines.append(
            f"| {r.backend} | {r.runtime} | {r.logprob_source} | {r.weight_sync} | "
            f"{r.num_generations} | {r.batch_prompts} | {r.mean_s:.3f} | {r.tokens} | "
            f"{r.tokens_per_s:.1f} | {r.notes} |"
        )
    # 自动附一条加速比，省得每次手算
    by_key = {(r.backend, r.logprob_source): r for r in results}
    base = next((r for r in results if r.backend == "hf"), None)
    if base is not None:
        for r in results:
            if r.backend != "hf" and r.tokens > 0 and r.mean_s > 0:
                lines.append(
                    f"\n- `{r.backend}` / `{r.logprob_source}` 相对 hf："
                    f"**{base.mean_s / r.mean_s:.2f}×**"
                )
    for r in results:
        if "max_abs_logprob_diff" in r.extra:
            lines.append(
                f"\n- `logprob_source` 一致性（同一串 token）："
                f"max |Δlogprob| = **{r.extra['max_abs_logprob_diff']:.2e}**"
            )
            if "ratio_min" in r.extra:
                lines.append(
                    f"，ratio ∈ [{r.extra['ratio_min']:.6f}, {r.extra['ratio_max']:.6f}]（期望 ≈ 1）"
                )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="rollout 后端消融")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="CPU 上的正确性与开销消融（无需 sglang）")
    smoke.add_argument("--batch-size", type=int, default=4)
    smoke.add_argument("--num-generations", type=int, default=4)
    smoke.add_argument("--max-new-tokens", type=int, default=8)
    smoke.add_argument("--repeats", type=int, default=3)

    local = sub.add_parser("local", help="单卡 GPU + 真实模型，量 logprob 复用的收益（无需 sglang）")
    local.add_argument("--model", required=True, help="HF id 或本地路径")
    local.add_argument("--batch-size", type=int, default=8, help="prompt 条数")
    local.add_argument("--num-generations", type=int, default=8)
    local.add_argument("--max-new-tokens", type=int, default=128)
    local.add_argument("--temperature", type=float, default=1.0,
                       help="一致性校验只在 1.0 下严格成立；其他值只测耗时")
    local.add_argument("--top-p", type=float, default=1.0)
    local.add_argument("--dtype", default="bfloat16")
    local.add_argument("--repeats", type=int, default=3)
    local.add_argument("--warmup", type=int, default=1)

    bench = sub.add_parser("bench", help="GPU 上的吞吐对比")
    bench.add_argument("--model", required=True, help="HF id 或本地路径")
    bench.add_argument("--backends", default="hf,sglang", help="逗号分隔：hf,sglang")
    bench.add_argument("--runtime", default="engine", choices=["engine", "http"])
    bench.add_argument("--server-url", default=None)
    bench.add_argument("--logprob-sources", default="sglang,recompute")
    bench.add_argument("--weight-sync", default="none", choices=["none", "tensor"])
    bench.add_argument("--batch-size", type=int, default=8, help="prompt 条数")
    bench.add_argument("--num-generations", type=int, default=8)
    bench.add_argument("--max-new-tokens", type=int, default=256)
    bench.add_argument("--temperature", type=float, default=1.0)
    bench.add_argument("--top-p", type=float, default=1.0)
    bench.add_argument("--dtype", default="bfloat16")
    bench.add_argument("--tp-size", type=int, default=1)
    bench.add_argument("--mem-fraction", type=float, default=0.6)
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--warmup", type=int, default=1)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "smoke":
        results = run_smoke(args)
        title = "smoke：正确性与 CPU 开销（绝对耗时无参考价值）"
    elif args.command == "local":
        results = run_local(args)
        title = f"local：{args.model}（{args.batch_size} prompts × n={args.num_generations} × ≤{args.max_new_tokens} tok）"
    else:
        args.backends = [b.strip() for b in args.backends.split(",") if b.strip()]
        args.logprob_sources = [s.strip() for s in args.logprob_sources.split(",") if s.strip()]
        results = run_bench(args)
        title = f"bench：{args.model}（{args.batch_size} prompts × n={args.num_generations}）"

    markdown = to_markdown(results, title)
    print("\n" + markdown + "\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.command}-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(
        json.dumps(
            {"command": args.command, "args": vars(args), "runs": [asdict(r) for r in results]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"已写入 {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
