from __future__ import annotations

import torch

__all__ = ["gather_token_logprobs", "masked_mean", "masked_sum", "broadcast_sequence_to_tokens"]


def gather_token_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    chunk_rows: int | None = None,
) -> torch.Tensor:
    """取每个位置上「真实 token」的 logprob，结果右对齐到 input_ids。

    ``logits[:, :-1]`` 预测 ``input_ids[:, 1:]``。

    分块：``log_softmax`` 前要先 ``.float()``，全 batch 一起转会在显存里
    同时物化两份 ``[B, L-1, V]`` 的 fp32（实测 64 序列 × 150 token × 151k
    词表 ≈ 13GB，6GB 卡直接 OOM）。log_softmax 逐行独立，按 batch 维分块
    数值**逐位等价**、峰值显存只与 chunk 大小相关。``chunk_rows=None``
    时按「fp32 chunk ≈ 2^27 个元素」自动定块长。
    """
    if logits.dim() != 3:
        raise ValueError(f"logits 应为 [B, L, V]，收到 {tuple(logits.shape)}")
    if input_ids.shape != logits.shape[:2]:
        raise ValueError(
            f"input_ids {tuple(input_ids.shape)} 与 logits 的前两维 "
            f"{tuple(logits.shape[:2])} 不一致"
        )

    B, L, V = logits.shape
    if chunk_rows is None:
        chunk_rows = max(1, (1 << 27) // max(1, (L - 1) * V))

    out = torch.zeros(input_ids.shape, dtype=torch.float32, device=logits.device)
    for start in range(0, B, chunk_rows):
        sl = slice(start, min(start + chunk_rows, B))
        # logits[:, :-1] 预测 input_ids[:, 1:]
        chunk_lp = torch.log_softmax(logits[sl, :-1, :].float(), dim=-1)
        gathered = chunk_lp.gather(
            -1, input_ids[sl, 1:].unsqueeze(-1)
        ).squeeze(-1)                                            # [b, L-1]
        out[sl, 1:] = gathered

    return out.to(logits.dtype)


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
    dim: int | None = None,
) -> torch.Tensor:
    mask = mask.to(tensor.dtype)
    if dim is None:
        return (tensor * mask).sum() / (mask.sum() + eps)
    return (tensor * mask).sum(dim=dim) / (mask.sum(dim=dim) + eps)


def masked_sum(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    if dim is None:
        return (tensor * mask.to(tensor.dtype)).sum()
    return (tensor * mask.to(tensor.dtype)).sum(dim=dim)


def broadcast_sequence_to_tokens(
    values: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    if values.dim() != 1:
        raise ValueError(f"values 应为 [B]，收到 {tuple(values.shape)}")
    if values.shape[0] != response_mask.shape[0]:
        raise ValueError(
            f"values 的 batch 维 {values.shape[0]} 与 response_mask 的 "
            f"{response_mask.shape[0]} 不一致"
        )
    return values.unsqueeze(-1) * response_mask.to(values.dtype)
