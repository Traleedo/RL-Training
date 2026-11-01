"""用 SGLang 作为 rollout 后端。

为什么需要它
------------
``HFRolloutEngine`` 走 ``model.generate``：一次前向只推进**一个** token，且
每一步都要在 Python 里循环 —— 对 0.6B 模型、8 条采样、512 token 的组采样来说，
rollout 常常占据整个 train_step 的绝大部分时间（``time/rollout_s`` 会告诉你
具体比例）。SGLang 用 paged KV cache + CUDA graph + 连续批处理把这段压下去。

设计边界（与全仓库一致的解耦方式）
----------------------------------
1. 只实现 ``RolloutEngine`` 这一个抽象：装配逻辑、``Batch`` 契约、
   ``num_generations`` 的分组顺序全部沿用基类与 ``models.rollout_common``。
   trainer 不需要知道后端是谁。
2. **后端可插拔**：``runtime`` 选 ``engine``（同进程 ``sglang.Engine``）/
   ``http``（``sglang.launch_server`` 的 REST 接口）/ ``mirror``（本地 HF 生成，
   仅用于 CI 与链路验证）。三者实现同一个 ``RolloutBackend`` 协议，
   引擎层代码**一行都不用改**。
3. 依赖是**惰性**的：``import models`` 不会 import sglang。没装 sglang 的机器
   照样能跑全部 CPU 测试，只有真的选 ``runtime=engine`` 时才会报错。
4. 权重同步是显式契约：``uses_external_weights`` + ``sync_weights`` 两个钩子由
   基类声明、``RLTrainer`` 在每个优化步之后调用。不实现同步不会报错，
   但会**静默地训练一个过期策略** —— 所以 ``weight_sync=none`` 会在启动时
   打一条 WARNING，别把它当成默认的生产配置。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import torch

from core.base_rollout import RolloutEngine
from core.batch import Batch
from core.registry import register
from models.rollout_common import (
    assemble_batch,
    attach_token_logprobs,
    chat_prompt_ids,
    recompute_batch_logprobs,
    resolve_pad_token_id,
    trim_at_eos,
)

__all__ = [
    "Sample",
    "SamplingParams",
    "RolloutBackend",
    "SGLangEngineBackend",
    "SGLangHTTPBackend",
    "HFMirrorBackend",
    "SGLangRolloutEngine",
]

logger = logging.getLogger(__name__)

#: 支持的运行时
RUNTIMES = ("engine", "http", "mirror")


# ======================================================================
# 值对象
# ======================================================================
@dataclass(frozen=True)
class SamplingParams:
    """一次采样的参数。

    单独一个值对象而不是**kwargs，是为了让「哪些参数真的传到了服务端」
    可枚举 —— sglang 的参数名与 HF 的 ``generate`` 并不一一对应，
    混在一起最容易出现「配置写了但没生效」。
    """

    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    return_logprob: bool = True
    stop: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "max_new_tokens": int(self.max_new_tokens),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "n": int(self.n),
            "return_logprob": bool(self.return_logprob),
        }
        if self.stop:
            payload["stop"] = list(self.stop)
        return payload


@dataclass
class Sample:
    """一条采样结果。``token_logprobs`` 为 None 表示后端没返回。"""

    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    token_logprobs: list[float] | None = None
    finish_reason: str | None = None


class RolloutBackend(Protocol):
    """后端协议：只负责「给定 token ids，返回样本」。

    刻意做成 Protocol 而不是 ABC —— 测试与消融需要一个假后端，
    而假后端不该被迫继承生产代码的基类（那会让「有没有实现全」变成
    继承检查，而不是行为检查）。
    """

    name: str

    def generate(
        self, prompts: Sequence[Sequence[int]], params: SamplingParams
    ) -> list[list[Sample]]:
        """每条 prompt 返回 ``params.n`` 个样本，外层顺序与输入一致。"""

    def update_weights(self, tensors: list[tuple[str, torch.Tensor]]) -> bool:
        """把新权重推给推理端。返回 False 表示该后端不支持。"""

    def close(self) -> None:
        """释放资源。"""


# ======================================================================
# 后端 1：同进程 sglang.Engine
# ======================================================================
class SGLangEngineBackend:
    """``sglang.Engine``（同进程、共享 GPU）。

    只在 Linux + CUDA 上可用；Windows / 无卡机器上构造时给出明确报错，
    而不是让 ``import sglang`` 的 ImportError 冒到用户面前。
    """

    name = "sglang-engine"

    def __init__(
        self,
        model_path: str,
        *,
        tokenizer_path: str | None = None,
        dtype: str = "bfloat16",
        tp_size: int = 1,
        mem_fraction_static: float = 0.6,
        trust_remote_code: bool = True,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        try:
            import sglang as sgl
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise ImportError(
                "runtime='engine' 需要 sglang，但当前环境没有安装。\n"
                "  pip install 'sglang[all]'      # 需要 Linux + CUDA\n"
                "如果只是想验证链路或做本地基准，把 rollout.runtime 改成 "
                "'mirror'（用 HF 生成模拟服务端）。"
            ) from exc

        kwargs: dict[str, Any] = {
            "model_path": model_path,
            "dtype": dtype,
            "tp_size": int(tp_size),
            "mem_fraction_static": float(mem_fraction_static),
            "trust_remote_code": bool(trust_remote_code),
        }
        if tokenizer_path:
            kwargs["tokenizer_path"] = tokenizer_path
        kwargs.update(engine_kwargs or {})

        logger.info("启动 sglang.Engine：%s", {k: v for k, v in kwargs.items() if k != "model_path"})
        self._engine = sgl.Engine(**kwargs)
        self._weights_supported = hasattr(self._engine, "update_weights_from_tensor")

    # ------------------------------------------------------------------
    def generate(
        self, prompts: Sequence[Sequence[int]], params: SamplingParams
    ) -> list[list[Sample]]:
        payload = params.to_payload()
        out: list[list[Sample]] = []
        for prompt_ids in prompts:
            result = self._engine.generate(
                input_ids=list(prompt_ids), sampling_params=payload
            )
            out.append(_samples_from_payload(result, n=params.n))
        return out

    def update_weights(self, tensors: list[tuple[str, torch.Tensor]]) -> bool:
        if not self._weights_supported:
            return False
        self._engine.update_weights_from_tensor(tensors)
        return True

    def close(self) -> None:
        shutdown = getattr(self._engine, "shutdown", None)
        if callable(shutdown):
            shutdown()


# ======================================================================
# 后端 2：HTTP（sglang.launch_server）
# ======================================================================
class SGLangHTTPBackend:
    """走 ``/generate`` 的 REST 后端。适合「服务端已经起好、训练脚本另起一进程」。
    """

    name = "sglang-http"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:30000",
        *,
        timeout: float = 600.0,
        model_name: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.model_name = model_name

    # ------------------------------------------------------------------
    def generate(
        self, prompts: Sequence[Sequence[int]], params: SamplingParams
    ) -> list[list[Sample]]:
        import requests

        out: list[list[Sample]] = []
        for prompt_ids in prompts:
            body: dict[str, Any] = {
                "input_ids": list(prompt_ids),
                "sampling_params": params.to_payload(),
            }
            if self.model_name:
                body["model"] = self.model_name
            resp = requests.post(
                f"{self.base_url}/generate", json=body, timeout=self.timeout
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"sglang 服务端返回 {resp.status_code}：{resp.text[:500]}\n"
                    f"检查 {self.base_url} 是否在跑、模型是否已加载。"
                )
            out.append(_samples_from_payload(resp.json(), n=params.n))
        return out

    def update_weights(self, tensors: list[tuple[str, torch.Tensor]]) -> bool:
        # REST 端没有通用的「按张量更新」接口：要么用 disk 路径，要么由
        # 训练端与服务端共享内存。这里明确不支持，让上层打 WARNING。
        return False

    def close(self) -> None:
        pass


# ======================================================================
# 后端 3：本地 HF 镜像（CI / 链路验证）
# ======================================================================
class HFMirrorBackend:
    """用本地 HF 模型模拟服务端。

    存在的理由不是「再多一条生成路径」，而是让**引擎层代码**（模板、分组、
    logprob 放置、Batch 契约）能在没有 GPU、没有 sglang 的地方被真实跑到。
    CI 里跑的就是它 —— 性能数字对 sglang 无意义，报告里必须标明。
    """

    name = "hf-mirror"

    def __init__(self, model: Any, tokenizer: Any, *, do_sample: bool = True) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._do_sample = bool(do_sample)

    # ------------------------------------------------------------------
    def generate(
        self, prompts: Sequence[Sequence[int]], params: SamplingParams
    ) -> list[list[Sample]]:
        out: list[list[Sample]] = []
        pad = resolve_pad_token_id(self._tokenizer)
        device = next(self._model.parameters()).device

        for prompt_ids in prompts:
            ids = torch.as_tensor([list(prompt_ids)], dtype=torch.long, device=device)
            ids = ids.repeat_interleave(int(params.n), dim=0)
            with torch.no_grad():
                seqs = self._model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=int(params.max_new_tokens),
                    do_sample=self._do_sample,
                    temperature=float(params.temperature),
                    top_p=float(params.top_p),
                    pad_token_id=pad,
                )
            generated = seqs[:, ids.shape[1]:]
            group: list[Sample] = []
            for row in generated:
                row = trim_at_eos(row, self._tokenizer.eos_token_id)
                group.append(
                    Sample(
                        token_ids=[int(t) for t in row],
                        text=self._tokenizer.decode(row, skip_special_tokens=True),
                    )
                )
            out.append(group)
        return out

    def update_weights(self, tensors: list[tuple[str, torch.Tensor]]) -> bool:
        # 镜像后端与 actor 本来就是同一个对象，无所谓同步
        return False

    def close(self) -> None:
        pass


# ======================================================================
# payload 解析
# ======================================================================
def _samples_from_payload(payload: Any, *, n: int) -> list[Sample]:
    """把 sglang 的返回（dict 或 dict 列表）解析成 ``Sample``。

    单独拆出来是为了可以对着**真实返回格式**写单测 —— 这类解析代码最常见的
    失效方式是「字段名对上了但层级不对」，而它只会在半夜的 GPU 机器上炸。
    """
    items = payload if isinstance(payload, list) else [payload]
    out: list[Sample] = []
    for item in items:
        if not isinstance(item, dict):
            raise TypeError(f"sglang 返回了非 dict 元素：{type(item).__name__}")
        meta = item.get("meta_info") or {}
        token_ids = _extract_token_ids(item, meta)
        out.append(
            Sample(
                token_ids=token_ids,
                text=str(item.get("text", "")),
                token_logprobs=_extract_logprobs(meta),
                finish_reason=_extract_finish_reason(meta),
            )
        )
    if len(out) != n:
        raise ValueError(
            f"请求 n={n} 条采样，服务端返回 {len(out)} 条。"
            f"检查 sampling_params.n 是否被服务端忽略。"
        )
    return out


def _extract_token_ids(item: dict, meta: dict) -> list[int]:
    for key in ("output_ids", "token_ids"):
        value = item.get(key)
        if isinstance(value, list) and value and isinstance(value[0], int):
            return [int(t) for t in value]
    # sglang 的 meta_info 里 output_token_logprobs 的每项是
    # (logprob, token_id) 或 (logprob, token_id, text)
    logprobs = meta.get("output_token_logprobs")
    if isinstance(logprobs, list):
        ids = [int(entry[1]) for entry in logprobs if isinstance(entry, (list, tuple)) and len(entry) >= 2]
        if ids:
            return ids
    raise KeyError(
        "无法从 sglang 返回里取到生成的 token ids。"
        f"顶层键：{sorted(item)}；meta_info 键：{sorted(meta)}。\n"
        "请在服务端开启 return_logprob，或改用 logprob_source='recompute' 并确认"
        "返回里至少带 output_ids。"
    )


def _extract_logprobs(meta: dict) -> list[float] | None:
    entries = meta.get("output_token_logprobs")
    if not isinstance(entries, list) or not entries:
        return None
    out: list[float] = []
    for entry in entries:
        if isinstance(entry, (list, tuple)) and entry:
            out.append(float(entry[0]))
        elif isinstance(entry, dict) and "logprob" in entry:
            out.append(float(entry["logprob"]))
        else:
            return None
    return out


def _extract_finish_reason(meta: dict) -> str | None:
    reason = meta.get("finish_reason")
    if isinstance(reason, dict):
        return reason.get("type") or reason.get("reason")
    return reason if isinstance(reason, str) else None


# ======================================================================
# 引擎
# ======================================================================
@register("rollout", "sglang")
class SGLangRolloutEngine(RolloutEngine):
    """把采样委托给 SGLang，其余全部沿用框架契约。"""

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        *,
        runtime: str = "engine",
        backend: RolloutBackend | None = None,
        logprob_source: str | None = None,
        weight_sync: str = "none",
        model_path: str | None = None,
        dtype: str = "bfloat16",
        tp_size: int = 1,
        mem_fraction_static: float = 0.6,
        server_url: str | None = None,
        server_timeout: float = 600.0,
        trust_remote_code: bool = True,
        engine_kwargs: dict[str, Any] | None = None,
        do_sample: bool = True,
        recompute_logprobs: bool | None = None,
        num_generations: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> None:
        super().__init__(
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        self.tokenizer = tokenizer

        if recompute_logprobs is not None:
            # 兼容 hf 引擎的配置键：语义等价于 logprob_source
            logprob_source = "recompute" if recompute_logprobs else "sglang"

        self.runtime = str(runtime).lower()
        if self.runtime not in RUNTIMES and backend is None:
            raise ValueError(
                f"未知的 rollout.runtime {runtime!r}；支持 {list(RUNTIMES)}"
            )

        self._actor = model
        self.backend: RolloutBackend = backend or self._make_backend(
            runtime=self.runtime,
            model=model,
            tokenizer=tokenizer,
            model_path=model_path,
            dtype=dtype,
            tp_size=tp_size,
            mem_fraction_static=mem_fraction_static,
            server_url=server_url,
            server_timeout=server_timeout,
            trust_remote_code=trust_remote_code,
            engine_kwargs=engine_kwargs,
            do_sample=do_sample,
        )

        # 默认策略：后端能给 logprob 就直接用（省掉一次 0.6B 全序列前向），
        # 给不了就只能重算 —— 两条路径都保证 rollout_logprobs 是「采样时」的值。
        if logprob_source is None:
            logprob_source = "recompute" if self.runtime == "mirror" else "sglang"
        if logprob_source not in ("sglang", "recompute"):
            raise ValueError(
                f"未知的 logprob_source {logprob_source!r}；支持 'sglang' / 'recompute'"
            )
        self.logprob_source = logprob_source

        if weight_sync not in ("none", "tensor"):
            raise ValueError(f"未知的 weight_sync {weight_sync!r}；支持 'none' / 'tensor'")
        self.weight_sync = weight_sync

        self._last: dict[str, float] = {}
        self._warned_no_logprob = False
        self._warned_stale = False
        self._sync_broken = False

        logger.info("rollout 引擎：%s", self.describe())
        if self.uses_external_weights:
            logger.warning(
                "rollout 使用外部权重副本，但 weight_sync='none' —— "
                "采样用的策略不会随训练更新。这只适用于吞吐/延迟基准，"
                "用于真实训练会让重要性采样静默失效（rollout_logprobs 与当前策略不一致）。"
            )

    # ------------------------------------------------------------------
    def _make_backend(self, *, runtime: str, **kwargs: Any) -> RolloutBackend:
        if runtime == "engine":
            model_path = kwargs.pop("model_path", None) or _guess_model_path(kwargs["tokenizer"])
            if not model_path:
                raise ValueError(
                    "runtime='engine' 需要 model_path。在 rollout 配置里显式写 "
                    "model_path，或确保 tokenizer.name_or_path 可用。"
                )
            kwargs.pop("do_sample", None)
            kwargs.pop("server_url", None)
            kwargs.pop("server_timeout", None)
            kwargs.pop("model", None)
            kwargs.pop("tokenizer", None)
            return SGLangEngineBackend(model_path, **kwargs)
        if runtime == "http":
            url = kwargs.pop("server_url", None)
            if not url:
                raise ValueError("runtime='http' 需要 server_url（例如 http://127.0.0.1:30000）")
            return SGLangHTTPBackend(url, timeout=kwargs.pop("server_timeout") or 600.0)
        # mirror
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")
        if model is None:
            raise ValueError("runtime='mirror' 需要 model（actor 模块），用于本地生成")
        return HFMirrorBackend(model, tokenizer, do_sample=bool(kwargs.get("do_sample", True)))

    # ------------------------------------------------------------------
    @property
    def uses_external_weights(self) -> bool:
        return self.runtime in ("engine", "http") and self.weight_sync != "none"

    def describe(self) -> str:
        return (
            f"SGLangRolloutEngine(runtime={self.runtime}, backend={self.backend.name}, "
            f"logprob_source={self.logprob_source}, weight_sync={self.weight_sync}, "
            f"num_generations={self.num_generations})"
        )

    # ------------------------------------------------------------------
    def generate(
        self,
        prompts: list[str],
        *,
        num_generations: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **gen_kwargs: Any,
    ) -> Batch:
        params = self.resolve(
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        n = params["num_generations"]
        if not prompts:
            raise ValueError("generate 收到空的 prompt 列表")

        # 组顺序契约：同一 prompt 的 n 条必须连续（Batch.from_rollout 依赖它）。
        # 注意发给后端的是**去重后的 prompt**（n 作为采样参数），不是展开后的
        # n 份 —— 后者会让服务端把同一个 prompt 重复预填 n 次，白白丢掉
        # 连续批处理与 prefix cache 的收益。
        unique_prompt_ids = [chat_prompt_ids(self.tokenizer, p) for p in prompts]
        expanded = [p for p in prompts for _ in range(n)]

        t0 = time.perf_counter()
        groups = self.backend.generate(
            unique_prompt_ids,
            SamplingParams(
                max_new_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                n=n,
                return_logprob=self.logprob_source == "sglang",
                stop=tuple(gen_kwargs.get("stop") or ()),
            ),
        )
        self._last["rollout/backend_s"] = time.perf_counter() - t0

        if len(groups) != len(prompts):
            raise ValueError(
                f"后端返回 {len(groups)} 组，期望 {len(prompts)} 组（每条 prompt 一组）"
            )

        prompt_ids: list[torch.Tensor] = []
        response_ids: list[torch.Tensor] = []
        response_texts: list[str] = []
        token_logprobs: list[list[float]] = []
        all_have_logprobs = True

        for ids, group in zip(unique_prompt_ids, groups):
            if len(group) != n:
                raise ValueError(
                    f"请求 n={n} 条采样，服务端实际返回 {len(group)} 条。"
                    f"检查 sampling_params.n 是否被服务端忽略。"
                )
            for sample in group:
                prompt_ids.append(torch.as_tensor(ids, dtype=torch.long))
                response_ids.append(torch.as_tensor(sample.token_ids, dtype=torch.long))
                response_texts.append(sample.text)
                if sample.token_logprobs is None:
                    all_have_logprobs = False
                else:
                    token_logprobs.append(list(sample.token_logprobs))

        batch = assemble_batch(
            self.tokenizer,
            prompt_ids,
            response_ids,
            prompt_texts=expanded,
            response_texts=response_texts,
            num_generations=n,
        )

        use_backend_logprobs = self.logprob_source == "sglang" and all_have_logprobs
        if self.logprob_source == "sglang" and not all_have_logprobs and not self._warned_no_logprob:
            self._warned_no_logprob = True
            logger.warning(
                "后端未返回 token logprobs，退回 logprob_source='recompute'"
                "（多一次全序列前向）。要省掉它，确认服务端开启了 return_logprob。"
            )
        if use_backend_logprobs:
            batch = attach_token_logprobs(batch, token_logprobs)
        else:
            if self._actor is None:
                raise ValueError(
                    "logprob_source='recompute' 需要 actor 模块，但构造时没有传 model。"
                )
            batch = recompute_batch_logprobs(self._actor, batch)

        tokens = int(batch["response_mask"].sum())
        elapsed = self._last["rollout/backend_s"]
        self._last["rollout/backend_tokens_per_s"] = tokens / elapsed if elapsed > 0 else 0.0
        return batch

    # ------------------------------------------------------------------
    def sync_weights(self, actor_module: Any = None) -> None:
        if not self.uses_external_weights:
            return
        module = actor_module if actor_module is not None else self._actor
        if module is None:
            return
        if hasattr(module, "module"):
            module = module.module

        named = list(module.named_parameters())
        trainable = [(name, p) for name, p in named if p.requires_grad]
        if len(trainable) < len(named) and not self._sync_broken:
            # PEFT / LoRA：可训练参数只是 adapter，推过去也不会改变服务端的底座。
            # 与其「推了一半」，不如明确说清楚 —— 静默的半同步会让
            # rollout_logprobs 与当前策略悄悄分叉。
            self._sync_broken = True
            logger.warning(
                "检测到 LoRA（%d/%d 个参数可训练）：sglang 侧的权重同步尚未实现"
                "（需要先 merge adapter 再推送）。本次运行不会同步权重 —— "
                "请用 weight_sync='none' 明确接受这一点，或改用全参微调。",
                len(trainable),
                len(named),
            )
            return

        tensors = [(name, p.detach().contiguous()) for name, p in trainable]
        try:
            ok = self.backend.update_weights(tensors)
        except Exception as exc:  # pragma: no cover - 依赖服务端实现
            if not self._sync_broken:
                self._sync_broken = True
                logger.warning(
                    "权重同步失败（%r）。后续步将不再尝试 —— 训练会继续，但采样用的"
                    "策略是过期的。请检查 sglang 版本是否支持 update_weights_from_tensor。",
                    exc,
                )
            return
        if not ok and not self._sync_broken:
            self._sync_broken = True
            logger.warning(
                "后端 %s 不支持张量级权重更新，rollout 权重不会随训练变化。"
                "要么改用 runtime='engine'，要么只在基准测试里使用当前配置。",
                self.backend.name,
            )

    def metrics(self) -> dict[str, float]:
        return dict(self._last)

    def close(self) -> None:
        self.backend.close()


def _guess_model_path(tokenizer: Any) -> str | None:
    """从 tokenizer 反推 HF 仓库 id。

    ``RLTrainer`` 只把 actor 模块与 tokenizer 传给 rollout，不传 model 配置；
    而 sglang 需要的是「一个能被 transformers 加载的路径或 id」。
    tokenizer 的 ``name_or_path`` 正好是它 —— 本地路径与 Hub id 都成立。
    """
    for attr in ("name_or_path", "name"):
        value = getattr(tokenizer, attr, None)
        if value:
            return str(value)
    return None
