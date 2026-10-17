from __future__ import annotations

# 新增的损失项在这里 import —— **import 就是注册**。少一行，YAML 里引用它就会
# 报「未注册的组件」（tests/test_configs.py 会在跑测试时抓到）。
#
# 注意 policy_gradient 必须在 ppo_clip **之前**：ppo_clip 是它的子类，
# 反过来 import 会成环。顺序在这里是有意义的，不是随手排的。
from components.losses.cross_entropy import CrossEntropyLoss
from components.losses.dpo import DPOLoss
from components.losses.gspo import GSPOLoss
from components.losses.kl_penalty import KLK1Loss, KLK2Loss, KLK3Loss, KLEstimator
from components.losses.policy_gradient import PolicyGradientLoss
from components.losses.ppo_clip import PPOClippedLoss
from components.losses.value_loss import ValueLoss

__all__ = [
    "CrossEntropyLoss",
    "DPOLoss",
    "GSPOLoss",
    "KLEstimator",
    "KLK1Loss",
    "KLK2Loss",
    "KLK3Loss",
    "PolicyGradientLoss",
    "PPOClippedLoss",
    "ValueLoss",
]
