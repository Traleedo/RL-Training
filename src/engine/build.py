from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

__all__ = ["build_trainer", "TRAINER_KINDS"]


def build_trainer(config: DictConfig) -> Any:
    from engine.offline_trainer import OfflineTrainer
    from engine.rl_trainer import RLTrainer

    kind = str(config.get("trainer", {}).get("kind", "rl")).lower()
    if kind == "rl":
        return RLTrainer(config)
    if kind == "offline":
        return OfflineTrainer(config)
    raise ValueError(
        f"未知的 trainer.kind {kind!r}；支持 {sorted(TRAINER_KINDS)}。\n"
        f"（'rl' = 样本来自策略采样，八相位；'offline' = 样本来自数据集，两相位）"
    )


#: 合法的 trainer.kind
TRAINER_KINDS = frozenset({"rl", "offline"})
