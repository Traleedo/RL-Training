"""日志后端实现。

``loggers`` 在 YAML 里是一个**列表**，所以「控制台 + tensorboard + wandb 同时开」
只是多写一行 —— 不需要任何框架改动。

新增一个后端：继承 ``core.base_logger.Logger``，实现 ``log``，加
``@register("logger", "<名字>")``，然后在本文件 import。

值得一并实现的几个可选钩子：

- ``log_config(cfg)``  —— wandb / tensorboard 的 hparams 记录
- ``log_text(key, text, step)``  —— 把抽样生成的文本记下来。
  这在 RL 后训练里其实很重要：reward 曲线好坏远不如「模型到底生成了什么」直观，
  而 reward hacking 往往在文本里一眼可见、在曲线上完全看不出来。
"""

from __future__ import annotations

from components.loggers.console import ConsoleLogger

__all__ = ["ConsoleLogger"]
