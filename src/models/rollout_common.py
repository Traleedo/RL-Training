"""Rollout 引擎之间共享的「文本 -> Batch」逻辑。

为什么要有这个模块
------------------
``hf`` 与 ``sglang`` 两个引擎的**差异**只有一处：谁来采样。
其余全部相同 —— chat 模板格式化、左 padding、``response_mask`` 对齐、
``group_ids`` 的连续块契约、logprob 的放置方式。

这些规则一旦有两份实现，迟早会有一份是错的，而错法是**静默的**：
mask 少一格不会报错，只会让某个 token 的梯度按错误的权重回传；
logprob 错位一位不会报错，只会让 ``train/ratio_first`` 轻微偏离 1，
看起来像「数值误差」。所以它们只能有一份实现，放在这里。

调用顺序（两个引擎都一样）::

    texts  = [format_chat_text(tok, p) for p in prompts]
    batch  = assemble_batch(...)                 # 对齐 + group_ids
    batch  = attach_token_logprobs(batch, lps)   # 采样端返回的 logprob
    # 或
    batch  = recompute_batch_logprobs(model, batch)   # 用 actor 重新前向
"""

from __future__ import annotations

import torch

from core import interfaces as F
from core.batch import Batch
from core.tensor_ops import gather_token_logprobs

__all__ = [
    "format_chat_text",
    "chat_prompt_ids",
    "trim_at_eos",
    "assemble_batch",
    "attach_token_logprobs",
    "recompute_batch_logprobs",
    "resolve_pad_token_id",
]


# ======================================================================
# 文本 / token
# ======================================================================
def format_chat_text(tokenizer, prompt: str) -> str:
    """把裸 instruction 包进 chat 模板（带生成引导）。

    只影响「喂给模型」的文本；``PROMPT_TEXTS`` 仍存原始 instruction，
    打分器会拿它和 response 重新拼 ``[user, assistant]`` 对话。
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def chat_prompt_ids(tokenizer, prompt: str) -> list[int]:
    """chat 模板 + 分词，一次完成。

    直接 ``tokenize=True`` 而不是「先成文本再 ``tokenizer(...)``」，是为了让
    **记录的 prompt_ids 与实际送进服务端的输入是同一份**。sglang 侧我们传
    ``input_ids`` 而不是文本，正是为了避免两边的分词细节（是否加 BOS、
    是否剥离特殊 token）悄悄分叉 —— 那会让 rollout 的 logprob 与训练前向
    的 logprob 对不上，表现为 ``ratio_first != 1``。
    """
    try:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    except (TypeError, ValueError):
        # 少数 tokenizer 的 apply_chat_template 不接受 tokenize=True
        ids = tokenizer(format_chat_text(tokenizer, prompt), add_special_tokens=False)["input_ids"]

    # transformers 5.x 起，``apply_chat_template(tokenize=True)`` 返回的是
    # BatchEncoding 而不是 ``list[int]`` —— 直接迭代会拿到 "input_ids" 这个 key，
    # 报 ``invalid literal for int() with base 10: 'input_ids'``。
    # 版本间差异不该泄漏到引擎层，在这里一次抹平。
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple, torch.Tensor)):
        first = ids[0]
        ids = first.tolist() if isinstance(first, torch.Tensor) else list(first)
    return [int(t) for t in ids]


def trim_at_eos(row: torch.Tensor, eos_id: int | None) -> torch.Tensor:
    """截到第一个 EOS（不含），让补出来的 pad 不参与 mask。"""
    if eos_id is None:
        return row
    hits = (row == eos_id).nonzero(as_tuple=False)
    if hits.numel():
        return row[: int(hits[0])]
    return row


def resolve_pad_token_id(tokenizer, default: int = 0) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    return int(default if pad is None else pad)


# ======================================================================
# Batch 组装
# ======================================================================
def assemble_batch(
    tokenizer,
    prompt_ids: list[torch.Tensor],
    response_ids: list[torch.Tensor],
    *,
    prompt_texts: list[str],
    response_texts: list[str],
    num_generations: int,
    pad_token_id: int | None = None,
) -> Batch:
    """对齐规则委托给 ``Batch.from_rollout`` —— 全仓库唯一决定对齐的地方。

    这里只做两件事：解析 pad_token_id，以及把「变长 token 列表」转成张量。
    """
    if pad_token_id is None:
        pad_token_id = resolve_pad_token_id(tokenizer)
    return Batch.from_rollout(
        prompt_ids,
        response_ids,
        prompt_texts=prompt_texts,
        response_texts=response_texts,
        num_generations=num_generations,
        pad_token_id=int(pad_token_id),
    )


def attach_token_logprobs(
    batch: Batch, token_logprobs: list[list[float]]
) -> Batch:
    """把**采样端返回的**逐 token logprob 放进 ``[B, L]`` 的 rollout_logprobs。

    放置位置用 ``RESPONSE_MASK`` 定位，而不是「算一个偏移量」。理由：
    偏移量（``max_prompt`` / ``L - max_response``）是 ``from_rollout`` 的内部
    约定，把它复制到这里就又多了一份需要同步的知识；而 mask 是**契约本身**，
    mask 为 1 的位置就是生成的 token，顺序即生成顺序。
    """
    n = len(batch)
    if len(token_logprobs) != n:
        raise ValueError(
            f"logprob 条数 {len(token_logprobs)} 与 batch 行数 {n} 不一致"
        )
    mask = batch[F.RESPONSE_MASK]
    out = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)

    for i in range(n):
        positions = mask[i].nonzero(as_tuple=False).flatten()
        values = token_logprobs[i]
        if len(values) != len(positions):
            raise ValueError(
                f"第 {i} 行：生成 token 数 {len(positions)} 与 logprob 数 "
                f"{len(values)} 不一致。\n"
                f"常见原因：采样端返回的 logprob 含 prompt 段（应只取输出段），"
                f"或 EOS 截断与 logprob 截断不同步。"
            )
        if len(values):
            out[i, positions] = torch.as_tensor(values, dtype=torch.float32, device=mask.device)

    batch[F.ROLLOUT_LOGPROBS] = out
    return batch


def recompute_batch_logprobs(model, batch: Batch) -> Batch:
    """用 actor 对完整序列重新前向，算行为策略 logprob。

    这是 ``logprob_source=recompute`` 与 ``HFRolloutEngine`` 的共用路径。
    它的正确性由 ``train/ratio_first == 1`` 这条断言守着。
    """
    module = model
    if hasattr(module, "module"):          # DataParallel / DDP 包装
        module = module.module

    with torch.no_grad():
        outputs = module(
            input_ids=batch[F.INPUT_IDS],
            attention_mask=batch[F.ATTENTION_MASK],
        )
        logprobs = gather_token_logprobs(outputs.logits, batch[F.INPUT_IDS])

    batch[F.ROLLOUT_LOGPROBS] = logprobs.detach()
    return batch
