from __future__ import annotations

import torch

__all__ = ["gather_token_logprobs", "masked_mean", "masked_sum", "broadcast_sequence_to_tokens"]


def gather_token_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
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
