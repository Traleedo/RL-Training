from __future__ import annotations

from core.base_controller import Controller
from core.registry import register

__all__ = ["AdaptiveKLController"]


@register("controller", "adaptive_kl")
class AdaptiveKLController(Controller):
    """按「观测 KL 与目标 KL 的差」成比例地调整 β。"""

    def __init__(
        self,
        term: str | None = None,
        target_kl: float = 0.01,
        step_size: float = 0.1,
        beta_min: float = 0.0,
        beta_max: float = 1.0,
    ) -> None:
        super().__init__(term=term)
        self.target_kl = float(target_kl)
        self.step_size = float(step_size)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

        if self.beta_min > self.beta_max:
            raise ValueError(
                f"beta_min={self.beta_min} 大于 beta_max={self.beta_max}，"
                f"夹取区间为空，β 会被钉在 beta_min 上。"
            )
        if self.step_size <= 0.0:
            raise ValueError(
                f"step_size={self.step_size} 必须为正，否则 β 永远不会变化。"
                f"如果想保持 β 恒定，请改用 controller/fixed_kl。"
            )

    # ------------------------------------------------------------------
    def on_train_step_end(self, metrics: dict[str, float]) -> None:
        key = f"loss/{self.target_name}/mean_kl"
        if key not in metrics:
            available = sorted(k for k in metrics if k.startswith("loss/"))
            raise KeyError(
                f"AdaptiveKLController 需要指标 {key!r}，但本步的指标里没有它。\n"
                f"    它绑定的是损失项 {self.target_name!r}，所以那一项必须在 compute() "
                f"返回的指标字典里报 'mean_kl'（kl_k1 / kl_k2 / kl_k3 都报）。\n"
                f"    本步可用的 loss/* 指标：{available}"
            )

        kl = float(metrics[key])
        previous = self.current_coef
        updated = previous + self.step_size * (kl - self.target_kl)
        beta = min(max(updated, self.beta_min), self.beta_max)
        self.current_coef = beta
        prefix = f"controller/{self.target_name}"
        metrics[f"{prefix}/beta"] = beta
        metrics[f"{prefix}/kl"] = kl
        metrics[f"{prefix}/kl_error"] = kl - self.target_kl
        metrics[f"{prefix}/at_bound"] = (
            1.0 if beta >= self.beta_max else (-1.0 if beta <= self.beta_min else 0.0)
        )

    def describe(self) -> str:
        return (
            f"{super().describe()}  target_kl={self.target_kl} "
            f"step_size={self.step_size} bounds=[{self.beta_min}, {self.beta_max}]"
        )
