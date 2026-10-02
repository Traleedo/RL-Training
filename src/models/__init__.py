"""HuggingFace 模型实现层。

**这一层是可选的**，不是框架核心。``core`` 与 ``engine`` 都不 import transformers，
你可以用完全自定义的组件替换掉这里的一切。

因此本模块在 transformers 不可用时会**跳过注册并给出警告**，而不是让整个框架
import 失败 —— 「核心不依赖 transformers」是这个项目想保住的性质。

但警告不等于静默吞掉：如果真的没有 transformers，给出 INFO 级的提示；
如果是 transformers 装好了、只是这里某个 import 写错了，那必须直接炸 ——
否则一个笔误会被伪装成「环境问题」，非常难查。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from models.hf_actor import HFPolicyActor
    from models.hf_common import load_causal_lm, load_tokenizer, load_value_model
    from models.hf_critic import HFSharedValueCritic, HFValueCritic
    from models.hf_reference import HFFrozenReference
    from models.hf_rollout import HFRolloutEngine
    from models.reward import HFOutcomeRewardScorer
except ImportError as exc:  # pragma: no cover - 取决于环境
    try:
        import transformers  # noqa: F401
    except ImportError:
        # 环境确实没有 transformers —— 合理的降级
        logger.info(
            "transformers 不可用，HF 组件不会被注册（这是允许的：框架核心不依赖 "
            "transformers，你可以用自定义组件替换 models/ 这一层）。原因：%r", exc
        )
    else:
        # transformers 在，说明是 models/ 自己的 import 写错了 —— 必须暴露出来
        raise

__all__ = [
    "HFPolicyActor",
    "HFValueCritic",
    "HFSharedValueCritic",
    "HFFrozenReference",
    "HFRolloutEngine",
    "HFOutcomeRewardScorer",
    "load_causal_lm",
    "load_tokenizer",
    "load_value_model",
]
