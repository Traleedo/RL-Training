"""张量级公共算子。

**这个文件承担一个全局不变量：logits/labels 的 shift 只允许在这里发生。**

因果语言模型的 logits 位于位置 ``i`` 时预测的是位置 ``i+1`` 的 token。
也就是说，一个 token 的 logprob 要用**前一个位置**的 logits 来取。
这个 off-by-one 看起来简单，但它是这类框架里最难查的一类 bug 源头：
如果组件 A 自己 shift 了一次、组件 B 忘了 shift，两者的张量形状仍然对得上，
语义却错位一位 —— 不会报错，只会让训练效果莫名变差。

因此约定：``gather_token_logprobs`` 是全仓库唯一做 shift 的地方，
任何组件（包括自定义的 loss term）都不许自己写 ``logits[..., :-1]``。
``tests/test_no_manual_shift.py`` 会静态扫描 ``src/models`` 与
``src/components`` 来守护这条约定。
"""

from __future__ import annotations

import torch

__all__ = ["gather_token_logprobs", "masked_mean", "masked_sum", "broadcast_sequence_to_tokens"]


def gather_token_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """从 logits 里取出每个位置 token 的 logprob。

    输入
    ----
    logits    : ``[B, L, V]``  因果 LM 的原始输出（未过 softmax）
    input_ids : ``[B, L]``     完整序列

    输出
    ----
    ``[B, L]``，其中 ``out[b, i] = log_softmax(logits[b, i-1])[input_ids[b, i]]``。

    位置 0 的 logprob 在数学上没有定义（没有「前一个位置」），此处填 0。
    这是安全的，因为位置 0 永远是 prompt 的第一个 token，而 ``response_mask``
    在位置 0 上恒为 0，不会参与任何归约。

    注意输出与 ``input_ids`` **位置对齐**：``out[b, i]`` 就是 token ``i`` 自己的
    logprob。所以调用方不需要、也不应该再做任何 shift。
    """
    if logits.dim() != 3:
        raise ValueError(f"logits 应为 [B, L, V]，收到 {tuple(logits.shape)}")
    if input_ids.shape != logits.shape[:2]:
        raise ValueError(
            f"input_ids {tuple(input_ids.shape)} 与 logits 的前两维 "
            f"{tuple(logits.shape[:2])} 不一致"
        )

    # logits[:, :-1] 预测 input_ids[:, 1:]
    shifted_logits = logits[:, :-1, :]                  # [B, L-1, V]
    shifted_labels = input_ids[:, 1:].unsqueeze(-1)     # [B, L-1, 1]

    log_probs = torch.log_softmax(shifted_logits.float(), dim=-1)
    gathered = log_probs.gather(-1, shifted_labels).squeeze(-1)   # [B, L-1]

    # 右侧补一位，保持与 input_ids 位置对齐
    out = torch.zeros(
        input_ids.shape, dtype=gathered.dtype, device=gathered.device
    )
    out[:, 1:] = gathered
    return out.to(logits.dtype)


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
    dim: int | None = None,
) -> torch.Tensor:
    """按 mask 求均值。``tensor`` 与 ``mask`` 形状一致（或可广播）。

    ``dim=None``（默认）对整个张量求一个标量 —— 这是「全局 token 均值」，
    也就是 DAPO / Dr.GRPO 要的固定分母。

    ``dim`` 指定时只在该维归约，其余维保留。传 ``dim=-1`` 就是**按行**求均值，
    即「每条序列各自平均」，GSPO 的序列级 ratio 与 seq_mean 归约都用它。
    这个参数是后加的，默认值保持原行为，所以既有的调用点不受影响。
    """
    mask = mask.to(tensor.dtype)
    if dim is None:
        return (tensor * mask).sum() / (mask.sum() + eps)
    return (tensor * mask).sum(dim=dim) / (mask.sum(dim=dim) + eps)


def masked_sum(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """按 mask 求和。``dim`` 的语义与 ``masked_mean`` 完全一致。

    ``dim=None``（默认）对整个张量求一个标量；传 ``dim=-1`` 是**按行**求和。

    按行求和是隐式信号那一类损失的原料：DPO 的每个序列得分
    ``Σ_t (curr_logprobs − ref_logprobs)`` 就是一次 ``masked_sum(..., dim=-1)``，
    而且分母是 1 不是 token 数 —— 所以这里必须是 sum 而不能拿 mean 凑。
    """
    if dim is None:
        return (tensor * mask.to(tensor.dtype)).sum()
    return (tensor * mask.to(tensor.dtype)).sum(dim=dim)


def broadcast_sequence_to_tokens(
    values: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """把序列级的 ``[B]`` 值广播成逐 token 的 ``[B, L]``。

    这是「序列级 -> 逐 token」的唯一推荐做法，保证所有 loss term 拿到的
    advantages / returns 都是 ``[B, L]``，形状约定不会被破坏。
    """
    if values.dim() != 1:
        raise ValueError(f"values 应为 [B]，收到 {tuple(values.shape)}")
    if values.shape[0] != response_mask.shape[0]:
        raise ValueError(
            f"values 的 batch 维 {values.shape[0]} 与 response_mask 的 "
            f"{response_mask.shape[0]} 不一致"
        )
    return values.unsqueeze(-1) * response_mask.to(values.dtype)
