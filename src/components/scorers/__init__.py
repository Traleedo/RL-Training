from __future__ import annotations

# 「import 就是注册」。少这一行，YAML 里写
# reward.scorer.type: ruled_based_math_reward 就会报「未注册的组件」，
# 而文件明明存在。
from components.scorers.ruled_based_math_reward import RuledBasedMathReward

__all__ = ["RuledBasedMathReward"]
