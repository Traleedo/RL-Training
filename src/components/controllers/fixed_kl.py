
from __future__ import annotations

from core.base_controller import Controller
from core.registry import register

__all__ = ["FixedKLController"]


@register("controller", "fixed_kl")
class FixedKLController(Controller):
    """β 保持 YAML 里写的值。只上报，不修改。"""

    def on_train_step_end(self, metrics: dict[str, float]) -> None:
        prefix = f"controller/{self.target_name}"
        metrics[f"{prefix}/beta"] = self.current_coef
