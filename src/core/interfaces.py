"""全局保留字段名、相位划分与模型映射 —— 跨模块的唯一真相来源。

任何组件都不应该硬编码字段名字符串，一律从这里 import 常量。
这样「某个字段改名字」只会影响这一个文件。

相位（phase）划分
-----------------
整个训练步骤被切成六个相位，每个相位只允许写特定的字段：

    A. rollout      生成序列，写入 input_ids / response_mask / rollout_logprobs
    B. score        打分，写入 rewards
    C. prepare      准备阶段前向，写入 ref_logprobs / rollout_values（detached）
    D. advantage    算优势，写入 advantages / returns
    E. 冻结          把 A~D 的产物 detach + freeze，切梯度，进入训练循环
    F. train        每个 mini-batch 内重算携带梯度的 curr_logprobs / values

**相位决定梯度，而不是组件自己声明。** 规则很简单：名字以 ``rollout_`` 开头的字段
以及 B~D 相位的产物一律 detached；只有 ``TRAIN_PHASE_FIELDS`` 里的两个字段携带梯度。
这条规则把「某个 loss term 误把 detached 张量当可微输入用」这类静默 bug
从「靠人记得」变成「由相位保证」。

形状约定
--------
``advantages`` / ``returns`` / ``*_logprobs`` / ``values`` 一律逐 token ``[B, L]``，
归约一律靠 ``response_mask``。序列级的量（例如 GRPO 的序列级 reward、
序列级 advantage）由产出方广播到 ``response_mask`` 为 1 的位置。
这一条消除了「逐 token 还是逐序列」这个最常见的形状 bug。
"""

from __future__ import annotations

# =====================================================================
# 相位 A：rollout 阶段写入（全部 detached）
# =====================================================================
INPUT_IDS = "input_ids"                  # [B, L]  prompt + response 全序列
ATTENTION_MASK = "attention_mask"        # [B, L]
RESPONSE_MASK = "response_mask"          # [B, L]  只在生成 token 上为 1，所有 loss 的归约域
ROLLOUT_LOGPROBS = "rollout_logprobs"    # [B, L]  行为策略 logprob（PPO 语境下的 old_logprobs）
ROLLOUT_VALUES = "rollout_values"        # [B, L]  行为策略价值，只给 GAE 用
GROUP_IDS = "group_ids"                  # [B]     = arange(G).repeat_interleave(n)

# non_tensors（逐样本元数据，不进设备）
PROMPT_TEXTS = "prompt_texts"            # list[str]，长度 B（已按 num_generations 展开）
RESPONSE_TEXTS = "response_texts"        # list[str]
PROMPT_LENGTHS = "prompt_lengths"        # list[int]

# =====================================================================
# 相位 B：打分与奖励处理
# =====================================================================
REWARDS = "rewards"                      # [B]  序列级原始分；RewardProcessor 可改

# =====================================================================
# 相位 C：准备阶段前向（detached，只算一次）
# =====================================================================
REF_LOGPROBS = "ref_logprobs"            # [B, L]

# =====================================================================
# 相位 D：优势
# =====================================================================
ADVANTAGES = "advantages"                # [B, L]
RETURNS = "returns"                      # [B, L]  只有需要 value loss 的算法才有

# =====================================================================
# 相位 F：训练（每个 mini-batch 内重算，携带梯度）
# =====================================================================
CURR_LOGPROBS = "curr_logprobs"          # [B, L]
VALUES = "values"                        # [B, L]

# =====================================================================
# 离线家族：样本来自固定数据集，不由策略采样
# =====================================================================
#: ``[B]``，``+1`` = chosen，``-1`` = rejected。
#:
#: 用符号而不是「组内第 0 行 / 第 1 行」来区分，是因为后者依赖行序 —— 而任何一次
#: ``split`` / ``filter`` 都可能打乱它，且打乱之后**不会报错**，只会算出反向的梯度。
#: 有符号之后，DPO 的 ``Δ_w − Δ_l`` 就与对内的行序无关。
PREFERENCE = "preference"

# =====================================================================
# meta 字典的保留键
# =====================================================================
NUM_GENERATIONS = "num_generations"
METRICS = "metrics"

# =====================================================================
# 模型标识（用于 plan_assembly 的模型清单）
# =====================================================================
MODEL_ACTOR = "actor"
MODEL_CRITIC = "critic"
MODEL_REFERENCE = "reference"

# =====================================================================
# 相位分组
# =====================================================================

#: rollout 相位必须写出的字段（RolloutEngine 的契约）
ROLLOUT_OUTPUT_FIELDS = frozenset({
    INPUT_IDS, ATTENTION_MASK, RESPONSE_MASK, ROLLOUT_LOGPROBS, GROUP_IDS,
})

#: dataset 相位必须写出的字段（BaseDataset 的契约）。
#:
#: 比 ROLLOUT_OUTPUT_FIELDS 少了 ``rollout_logprobs`` —— 离线样本没有「采样时的
#: 策略」这个概念，那个字段不该被任何离线损失项依赖。少一个字段不是遗漏，
#: 而是把「离线路径不存在行为策略」这件事写进类型里。
DATASET_OUTPUT_FIELDS = frozenset({
    INPUT_IDS, ATTENTION_MASK, RESPONSE_MASK, GROUP_IDS,
})

#: 进入训练循环前必须冻结（写保护）的字段。
#:
#: 尤其是 ROLLOUT_LOGPROBS —— 它在第 2 个 epoch 必须仍是 rollout 时刻的值，
#: 不能被「顺手刷新」成当前策略的 logprob，否则 PPO 的 ratio 恒等于 1，
#: 策略梯度项恒为 0，loss 曲线看起来完全正常但训练悄悄失效。
FROZEN_BEFORE_TRAIN = frozenset({
    ROLLOUT_LOGPROBS, ROLLOUT_VALUES, REWARDS, REF_LOGPROBS, ADVANTAGES, RETURNS,
})

#: 唯一两个携带梯度的字段，只在相位 F 由 actor / critic 写入
TRAIN_PHASE_FIELDS = frozenset({CURR_LOGPROBS, VALUES})

#: 字段 -> 需要哪个模型存在。
#:
#: plan_assembly() 靠它把「组件声明的 needed 并集」翻译成「要构建哪些模型」。
#: 注意 CURR_LOGPROBS / VALUES 不在这里：它们由 actor / critic 在训练相位产生，
#: 不是「需要某个模型」的意思。plan_assembly 必须先剥掉这两个字段再查表。
MODEL_FOR_FIELD: dict[str, str] = {
    REF_LOGPROBS: MODEL_REFERENCE,
    ROLLOUT_VALUES: MODEL_CRITIC,
    VALUES: MODEL_CRITIC,
}

#: 训练相位字段 —— plan_assembly 查表前必须剥离，否则会得到
#: 「需要名为 curr_logprobs 的模型」这种荒谬结论。
NON_MODEL_FIELDS = TRAIN_PHASE_FIELDS

# =====================================================================
# 相位名与「哪个家族跑哪些相位」
# =====================================================================
#: 相位名。字符串常量而不是枚举，是为了让 trainer 里写
#: ``phases={PHASE_ROLLOUT, ...}`` 一眼能读出来。
PHASE_DATASET = "dataset"
PHASE_ROLLOUT = "rollout"
PHASE_SCORE = "score"
PHASE_ADVANTAGE = "advantage"
PHASE_PREPARE = "prepare"
PHASE_TRAIN = "train"

#: RL 家族跑的相位。样本由策略采样，所以有 rollout / score / advantage。
RL_PHASES = frozenset({
    PHASE_ROLLOUT, PHASE_SCORE, PHASE_ADVANTAGE, PHASE_PREPARE, PHASE_TRAIN,
})

#: 离线家族跑的相位。样本来自数据集，于是**打分与优势两个相位整个不存在** ——
#: 这正是「隐式信号不需要奖励模型」这句话在代码里的样子。
OFFLINE_PHASES = frozenset({PHASE_DATASET, PHASE_PREPARE, PHASE_TRAIN})
