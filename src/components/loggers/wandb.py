"""Weights & Biases 日志后端。

``wandb`` 是**可选依赖**：没装、或者初始化失败（未登录 / 连不上外网）时，
这个后端降级成 no-op 并打一条 WARNING，训练照跑 —— 日志后端不该有能力
让整个训练挂掉。所以 ``import wandb`` 一律写在方法体里，模块顶层不 import。

用法（``configs/base.yaml`` 里有注释好的示例）::

    loggers:
      - type: console
        every_n_steps: 1
      - type: wandb
        project: rl-training
        mode: offline        # 没外网时；也可 export WANDB_MODE=offline

没装时::

    pip install wandb
"""

from __future__ import annotations

import logging
from typing import Any

from core.base_logger import Logger
from core.registry import register

__all__ = ["WandbLogger"]

logger = logging.getLogger(__name__)


@register("logger", "wandb")
class WandbLogger(Logger):
    """把指标转发给 wandb。"""

    def __init__(
        self,
        project: str = "rl-training",
        entity: str | None = None,
        run_name: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        mode: str | None = None,
        every_n_steps: int = 1,
    ) -> None:
        super().__init__()
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.group = group
        self.tags = list(tags) if tags else None
        #: online / offline / disabled。None 时交给 wandb 自己决定
        #: （它会读 WANDB_MODE 环境变量）。
        self.mode = mode
        self.every_n_steps = int(every_n_steps)
        #: ``wandb.init()`` 的返回值；None 表示没起来（没装 / 失败 / 还没 init）。
        self._run: Any = None
        self._tried = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def _ensure_run(self, run_name: str | None = None) -> None:
        """惰性初始化。没装或是初始化失败时把自己变成 no-op，不抛异常。"""
        if self._run is not None or self._tried:
            return
        self._tried = True
        try:
            import wandb
        except ImportError:
            logger.warning(
                "配置里开了 wandb 日志后端，但没有安装 wandb —— 已降级为 "
                "no-op，训练不受影响。要启用：pip install wandb"
            )
            return
        try:
            self._run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.run_name or run_name,
                group=self.group,
                tags=self.tags,
                mode=self.mode,
            )
        except Exception as exc:  # noqa: BLE001 - 任何 init 失败都该降级而非中断训练
            logger.warning("wandb.init 失败，已降级为 no-op：%r", exc)
            self._run = None

    def log_config(self, cfg: Any) -> None:
        """把配置写进 wandb 的 hparams。同时是延迟初始化的触发点。

        ``trainer`` 建完 logger 就调这个，所以 ``run_name`` 能从这里兜底 ——
        用户不必在 YAML 里和 ``trainer.run_name`` 写两遍。
        """
        run_name = None
        try:
            run_name = str(cfg.trainer.get("run_name", "")) or None
        except Exception:  # noqa: BLE001 - cfg 不是 DictConfig 时忽略
            pass
        self._ensure_run(run_name=run_name)
        if self._run is None:
            return

        try:
            from omegaconf import OmegaConf

            payload = (
                OmegaConf.to_container(cfg, resolve=True)
                if OmegaConf.is_config(cfg)
                else cfg
            )
        except Exception:  # noqa: BLE001
            payload = str(cfg)
        try:
            self._run.config.update(payload, allow_val_change=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("wandb 记录配置失败（不影响训练）：%r", exc)

    def log(self, metrics: dict[str, float], step: int, prefix: str = "train") -> None:
        self._ensure_run()
        if self._run is None:
            return
        if self.every_n_steps > 1 and step % self.every_n_steps != 0:
            return
        self._run.log(metrics, step=step)

    def close(self) -> None:
        if self._run is None:
            return
        try:
            import wandb

            wandb.finish()
        except Exception as exc:  # noqa: BLE001
            logger.warning("wandb.finish 失败：%r", exc)
        finally:
            self._run = None
