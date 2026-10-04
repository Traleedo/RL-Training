"""HF 模型加载的公共部分。"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

__all__ = [
    "resolve_dtype",
    "resolve_model_source",
    "load_tokenizer",
    "load_causal_lm",
    "load_value_model",
    "load_reward_model",
    "extract_transformer",
    "find_hidden_size",
]

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_dtype(name: str | None) -> torch.dtype:
    if name is None:
        return torch.float32
    key = str(name).lower()
    if key not in _DTYPE_MAP:
        raise ValueError(
            f"未知的 dtype {name!r}；支持 {sorted(_DTYPE_MAP)}"
        )
    return _DTYPE_MAP[key]


@lru_cache(maxsize=None)
def _download_to_local(name: str, modelscope_id: str | None) -> str:
    """把模型下载到本地，返回路径。先 HF，失败则落到 modelscope。

    返回值是本地目录，之后统一用 ``AutoModel.from_pretrained(local_path)``
    加载 —— 只有**下载源**在 HF 与 modelscope 之间切换，加载逻辑一条不变。
    modelscope 下载的目录同样是标准 HF 格式（config.json / tokenizer /
    pytorch_model 权重），所以下游不用知道它来自哪里。
    """
    hf_error: BaseException | None = None

    # 1. HF
    try:
        from huggingface_hub import snapshot_download

        # 注意：snapshot_download 只下载文件、不执行 remote code，所以没有
        # trust_remote_code 参数（那是 from_pretrained 的参数）。remote code
        # 文件（modeling_*.py）默认也会被一并下载下来。
        return snapshot_download(repo_id=name)
    except Exception as exc:  # noqa: BLE001 — 网络/权限/仓库不存在都算「这个源不可用」
        hf_error = exc
        logger.warning("从 HuggingFace 下载 %s 失败：%s", name, exc)

    # 2. modelscope
    try:
        from modelscope import snapshot_download as ms_snapshot_download
    except ImportError:
        raise RuntimeError(
            f"从 HuggingFace 下载 {name} 失败，而 modelscope 未安装，无法 fallback。\n"
            f"  HF 失败原因：{hf_error}\n"
            f"  两条出路任选：pip install modelscope（国内网络），"
            f"或确认这台机器能访问 huggingface.co"
        ) from hf_error

    ms_name = modelscope_id or name
    try:
        return ms_snapshot_download(model_id=ms_name)
    except Exception as ms_exc:  # noqa: BLE001
        raise RuntimeError(
            f"从 HuggingFace 与 modelscope 下载 {name} 都失败了。\n"
            f"  HuggingFace（{name}）：{hf_error}\n"
            f"  modelscope（{ms_name}）：{ms_exc}"
        ) from ms_exc


def resolve_model_source(model_config: Any) -> str:
    """把 ``name_or_path`` 解析成可供 ``from_pretrained`` 直接加载的路径。

    三种情况：
    1. 已经是本地目录 → 原样返回（不下载）。
    2. ``local_files_only`` → 原样返回 ``name_or_path``，交给 ``from_pretrained``
       从 HF 缓存读，不联网。
    3. 其余 → ``_download_to_local``：先 HF、失败落 modelscope，返回本地路径。

    下游三个调用点（config / tokenizer / model）各自解析一次，但
    ``_download_to_local`` 的 lru_cache 按 ``name_or_path`` 记忆结果，
    所以同一个模型只下载一次，也不会出现「config 从 HF、model 从 modelscope」
    这种混合源。
    """
    name = str(model_config.name_or_path)
    if Path(name).is_dir():
        return name
    if model_config.get("local_files_only", False):
        return name
    return _download_to_local(
        name,
        model_config.get("modelscope_id"),
    )


def _load_config(model_config: Any):
    """加载 HF config。``config_overrides`` 用来在脚本里把模型改小（离线跑测试）。"""
    from transformers import AutoConfig

    overrides = dict(model_config.get("config_overrides", {}) or {})
    kwargs: dict[str, Any] = {"trust_remote_code": bool(model_config.get("trust_remote_code", False))}
    if model_config.get("local_files_only", False):
        kwargs["local_files_only"] = True
    return AutoConfig.from_pretrained(resolve_model_source(model_config), **overrides, **kwargs)


def _tokenizer_kwargs(model_config: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
    }
    if model_config.get("local_files_only", False):
        kwargs["local_files_only"] = True
    return kwargs


def load_tokenizer(model_config: Any, tokenizer=None):
    """加载分词器。decoder-only 生成必须左 padding，否则生成会从 pad 后面继续。"""
    if tokenizer is not None:
        return tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        resolve_model_source(model_config), **_tokenizer_kwargs(model_config)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _from_pretrained(auto_cls, model_config: Any, tokenizer, **extra):
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
    }
    if model_config.get("local_files_only", False):
        kwargs["local_files_only"] = True
    if model_config.get("attn_implementation"):
        kwargs["attn_implementation"] = str(model_config.attn_implementation)

    config = model_config.get("config")
    if config is not None:
        # 允许直接把一个已构造好的 config 塞进来（e2e 脚本离线跑用）
        model = auto_cls(config, **extra)
        logger.info("从内存里的 config 构造 %s（未加载预训练权重）", auto_cls.__name__)
    else:
        model = auto_cls.from_pretrained(resolve_model_source(model_config), **kwargs, **extra)

    dtype = resolve_dtype(model_config.get("dtype"))
    model = model.to(dtype) if dtype != torch.float32 else model
    return model


def load_causal_lm(model_config: Any, tokenizer=None):
    """加载策略模型。返回 ``(model, tokenizer)``。"""
    from transformers import AutoModelForCausalLM

    tok = load_tokenizer(model_config, tokenizer)
    model = _from_pretrained(AutoModelForCausalLM, model_config, tok)
    return model, tok


def load_value_model(model_config: Any, tokenizer=None):
    from transformers import AutoModelForTokenClassification

    tok = load_tokenizer(model_config, tokenizer)
    model = _from_pretrained(
        AutoModelForTokenClassification, model_config, tok, num_labels=1
    )
    return model, tok


def load_reward_model(model_config: Any, tokenizer=None):
    """加载结果奖励模型（序列级标量）。"""
    from transformers import AutoModelForSequenceClassification

    tok = load_tokenizer(model_config, tokenizer)
    model = _from_pretrained(
        AutoModelForSequenceClassification, model_config, tok, num_labels=1
    )
    return model, tok


#: 隐藏维度在不同架构里的叫法。
_HIDDEN_KEYS = ("hidden_size", "n_embd", "d_model", "dim")


def find_hidden_size(config: Any) -> int:
    """从 HF config 里取隐藏维度。

    ``Qwen``/``Llama`` 叫 ``hidden_size``，``GPT-2`` 叫 ``n_embd``。
    挨个试而不是写死一个，是因为写错了不会报错 —— 会静默建出一个维度不对的
    value head，直到前向时才 shape mismatch，而那时的报错信息离真正的原因很远。
    """
    for key in _HIDDEN_KEYS:
        value = getattr(config, key, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError(
        f"无法从 config 里找到隐藏维度（试过 {_HIDDEN_KEYS}）。"
        f"请给 value head 显式指定 hidden_size。"
    )


def extract_transformer(model: Any) -> nn.Module:
    """从 ``...ForCausalLM`` 里取出底座，丢掉 ``lm_head``。

    HF 的因果 LM 把底座放在 ``.model``（Qwen / Llama / GPT-2 都是），
    顶层那个 ``lm_head`` 对价值函数毫无用处 —— 留着它会让一份没用的
    vocab × hidden 权重进入 critic 的 state_dict 和优化器参数组。

    取不到底座就原样返回，交给调用方去判断。自定义模型（比如仓库的
    ``tests/fixtures/``）本来就是扁平的，没有内嵌底座。
    """
    base = getattr(model, "model", None)
    if isinstance(base, nn.Module) and base is not model:
        return base
    return model
