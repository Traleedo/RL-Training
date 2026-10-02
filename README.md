<<<<<<< HEAD
# RL-Training
a moduled project for RL
=======
# 模块化 RL 训练框架（骨架）

一个用于 LLM 后训练的 RL 框架。核心动机只有一句：

> **不同强化学习算法的差异，只是组件的有无与组合方式。**

如果每个组件都能独立插拔，那么「换算法」就退化成编辑 YAML，而「加一项自定义损失」
只需要加一个文件加一个装饰器 —— 训练循环一行都不用改。

本仓库交付的是**骨架与接口**，外加**十个靠组件组装出来的算法配置** ——
八个 RL 家族（样本来自策略采样）与两个离线家族（样本来自固定数据集）。
这一切是同一套组件的不同组合：

| 配置 | 算法 | 与上一个的差量 |
|---|---|---|
| `configs/rl/ppo.yaml` | PPO | — |
| `configs/rl/ppo_kl_k1.yaml` | PPO-KL（k1 估计量） | **一个值** |
| `configs/rl/ppo_kl_k2.yaml` | PPO-KL（k2 估计量） | **一个值** |
| `configs/rl/ppo_adaptive_kl.yaml` | Adaptive-KL PPO | **一个块**（`controllers`） |
| `configs/rl/grpo.yaml` | GRPO | 三个组件（advantage / processor / loss） |
| `configs/rl/dr_grpo.yaml` | Dr.GRPO | 两个值（`divide_std` / `reduction`） |
| `configs/rl/dapo.yaml` | DAPO | 链首插一个 processor + 两个 loss 参数 |
| `configs/rl/gspo.yaml` | GSPO | **一个值**（`policy_gradient` → `gspo`） |
| `configs/sft.yaml` | SFT | 换一个家族：`trainer.kind: offline` + `data` 块 |
| `configs/dpo.yaml` | DPO | 与 sft 差一个损失项 + 成对数据 |

前八个共用 `RLTrainer`（八相位），后两个共用 `OfflineTrainer`（两相位）。
两个训练器共用同一份**骨架基类** `Trainer` —— 种子、优化器、调度器、
epoch × mini-batch 循环、checkpoint、resume、配置指纹、日志全在基类里，
有一个 AST 扫描的测试守着「它们没有被复制成两份」。

配置分成**两条独立的轴**，这是本仓库与「一个算法一个脚本」那种写法的分界：

```
configs/base.yaml          框架公共部分：optim / loggers / checkpointer / trainer
configs/model/<基座>.yaml  模型段：model / actor / critic / reference / rollout / reward_model
configs/rl/<算法>.yaml     算法段：algorithm / reward / rollout
```

一个算法配置同时继承前两者（`defaults: [../base, ../model/fixture]`），
所以**换基座不必碰算法配置，反之亦然**：

```bash
python -X utf8 scripts/train.py -a grpo -m qwen3-0.6b     # 算法不动，换基座
python -X utf8 scripts/train.py -a dapo -m fixture        # 基座不动，换算法
```

`rollout` 是唯一被两条轴都写的块（基座给 `type` / `max_new_tokens`，算法给
`num_generations`），所以合并是按**键** deep-merge 而不是整块替换 ——
整块替换会静默丢掉 `num_generations`，GRPO 悄悄退化成单采样而配置看起来完全正常。

「差量」那一列有几个**测试在守着**（`tests/test_configs.py`）：它不是宣传语，
而是一条会红的不变量。每个算法配置与它**整条默认链**的差量都只落在
`algorithm` / `reward` / `rollout` 三块里 —— 反过来说，任何配置都不许在
`model` / `actor` / `critic` 上出现差异，那意味着它绑死在了某个基座上。
两个离线配置多出 `data` 与 `trainer.kind`，这两块的差异本身也被单独断言 ——
没有一行训练循环为某个算法改过。

支撑它们的是 `src/components/` 下的真实实现：`advantages/`（`gae` / `broadcast`）、
`processors/`（`group_normalize` / `dynamic_filter`）、`losses/`
（`policy_gradient` / `ppo_clip` / `gspo` / `kl_k1` / `kl_k2` / `kl_k3` /
`value_loss` / `cross_entropy` / `dpo`）、`controllers/`（`fixed_kl` / `adaptive_kl`），
加上 `src/data/` 的数据集层。它们同时也是「新增组件 SOP」的活样板 ——
照抄一个文件改改就能用。

**「显式信号 / 隐式信号 / 监督」这个划分没有对应任何基类或分支** ——
它就是九个损失项各自 `requires` 的差别：

| | 样本来自 rollout | 样本来自数据集 |
|---|---|---|
| 显式信号（reward → advantage） | 八个 RL 配置 | 拒绝采样/RFT（未做） |
| 隐式信号（偏好对比） | RTO（未做） | **DPO** |
| 监督（无强化信号） | — | **SFT** |

`src/models/` 那一层的 HF 封装（`hf_causal_lm` / `hf_value_head` /
`hf_shared_value` / `hf_frozen` / `hf` / `hf_reward_model`）已经写好，
但**没在本机跑过**（本机 torchvision 与 torch 的 ABI 不匹配，是本机环境问题，
按约定没有去修）。所以 `configs/model/fixture.yaml` 目前把模型那几段指向
`tests/fixtures/` 里的微型模型，**算法是真的、模型是假的** ——
这样能在几秒内把算法调对，再换真模型时算法配置一个字都不用动
（见[验证与已知限制](#验证与已知限制)）。

critic 有两种实现，差别只在**权重从哪来**：

| `critic.type` | 权重来源 |
|---|---|
| `hf_value_head` | 独立 `from_pretrained` 一份自己的模型，与策略无关 |
| `hf_shared_value` | 从 actor 的底座 **deepcopy** 一份，接一个 `nn.Linear(hidden, 1)` |

后者是 PPO 的标准做法：critic 从策略的初始权重出发（此时它对「哪些 token
重要」的判断已经比随机初始化好得多），之后两者独立演化。**必须是 deepcopy**：
写成别名不会报任何错，只会让 value loss 的梯度悄悄改掉策略的网络权重 ——
loss 照样下降，ratio 照样正常，只是 PPO 早已不是 PPO。
`tests/test_model_axis.py` 里有四条断言守着这件事，包括「只跑价值回归时
actor 一个参数都不许动」。

---

## 目录

- [五分钟跑通](#五分钟跑通)
- [分层](#分层)
- [一、字段命名与相位规则](#一字段命名与相位规则)
- [二、GRPO 的归一化归属](#二grpo-的归一化归属)
- [三、response_mask 的 shift 只有一个出口](#三response_mask-的-shift-只有一个出口)
- [四、新增一个组件的 SOP](#四新增一个组件的-sop)
- [五、category 对应表](#五category-对应表)
- [六、组件 → 模型构建映射表](#六组件--模型构建映射表)
- [七、离线家族：SFT 与 DPO](#七离线家族sft-与-dpo)
- [组件速查](#组件速查)
- [验证与已知限制](#验证与已知限制)

---

## 五分钟跑通

```bash
pip install -r requirements.txt

pytest tests/ -q                         # 582 项，CPU 上一分钟
python scripts/verify_e2e.py --stage 1   # 十个配置各真跑一遍（装配 + 训练 + checkpoint）
```

训练：

```bash
python -X utf8 scripts/train.py -a ppo -m fixture --steps 20          # RL
python -X utf8 scripts/train.py -a dpo -m fixture --steps 20 --save-dir /tmp/dpo
python -X utf8 scripts/train.py -a dpo --resume /tmp/dpo/step_20      # 续跑
python -X utf8 scripts/train.py -a grpo -m qwen3-0.6b --steps 100     # 换真基座
python -X utf8 scripts/train.py -a ppo -m qwen3-0.6b \
    --reward-model Skywork/Skywork-Reward-V2-Qwen3-8B                 # 换奖励模型
```

`-a` 给算法、`-m` 给基座，两者互不干扰：`-m` 走的是「配方」而不是「覆写」
（`base + 选中的基座 + 算法自己的增量`），否则旧基座的 `vocab_size` 之类会留下来，
被 `build()` 原样当成构造参数传给 `hf_causal_lm`。
`--plan` 只打装配计划、不训练；`--set k=v`（可重复）是最后一道逃生口。
上面几条命令都有测试在跑（`tests/test_scripts.py`、`tests/test_model_axis.py`）。

或者只回答「这个变体会装配出什么」，不训练：

```bash
python -X utf8 scripts/plan.py dapo       # 也接受 rl/dapo 或完整路径
python -X utf8 scripts/plan.py --all      # 十个配置并排看
```

`dapo` 那个输出里的两行就是本框架最反直觉的一点：

```
  损失项（名称 × 权重）：
      kl_k3   weight=1.0  ...  needs=['curr_logprobs', 'response_mask']  coef=0.0
  实际构建的模型：
    reference  不构建
```

`kl_k3` 那一行**还在配置里**（`weight=1.0`），但它的 `needs` 里没有
`ref_logprobs`（`coef: 0.0` 释放掉了），所以 Reference 不加载。
「配置里写了」与「会被构建」是两件事，这个脚本把它们分开打印。

> **Windows 提示**：控制台默认是 GBK，中文日志会变成乱码。用
> `python -X utf8 ...` 或设 `PYTHONUTF8=1`。目标平台是 Linux 的话不用管。

看一个完整的训练步骤长什么样：

```python
from core.config import load_config
from engine.rl_trainer import RLTrainer

trainer = RLTrainer(load_config("configs/rl/grpo.yaml"))
print(trainer.plan.describe())      # 会构建哪些模型、为什么
metrics = trainer.train_step(["1+1=?", "2+3=?"])
```

想换成 PPO？**不用改一行代码**：

```python
trainer = RLTrainer(load_config("configs/rl/ppo.yaml"))
```

`configs/` 下**每一个**配置与它整条默认链的差量都只有 `rollout` / `reward` /
`algorithm` 三块，其中 `rollout` 只差 `num_generations` 一个键。有两个测试在守着
这条（`tests/test_configs.py`）：

- `test_every_config_differs_from_its_defaults_only_in_three_blocks` —— 逐个配置对默认链检查，
  所以新增一个配置不会稀释覆盖；
- `test_kl_variants_differ_from_ppo_in_exactly_one_leaf` —— 比到**叶子路径**，
  「看起来只改了一个值、其实顺带动了别处」也会被抓住。

---

## 分层

```
src/core/          抽象与契约。不知道任何具体算法的存在
src/engine/        装配推导 + 训练主循环（Trainer 骨架 + 两个家族的子类）
src/components/    纯逻辑实现（advantage / processor / loss / controller / 基础设施）
src/data/          数据集层（离线家族的相位 A）
src/models/        HF 模型封装（可选层，core/engine 都不 import transformers）
tests/fixtures/    假组件 —— 同时是「新增组件」的参考模板
configs/           base.yaml（框架公共）+ model/（基座）+ rl/（算法）+ sft / dpo
scripts/           train.py / train_offline.py / plan.py / verify_e2e.py
```

`src/engine/` 里三个训练器文件的分工，本身就是「骨架与算法分离」那句话：

```
trainer.py          Trainer —— 骨架。连 RLTrainer 这个词都不出现（AST 测试守着）
rl_trainer.py       RLTrainer(Trainer)   —— 八相位
offline_trainer.py  OfflineTrainer(Trainer) —— 两相位
build.py            trainer.kind 的唯一解释点
```

三条硬规则，都有测试守着：

1. **`components/` 内部不许横向 import**（`losses/` 不得 import `processors/`）。
   一旦横向引用，「单独启用 losses 而关掉 processors」就不成立，YAML 的模块化随之失效。
   共享逻辑请下沉到 `core/`。
2. **`components/` 不许反向 import `engine/`**。engine 依赖 components，反向就成了环。
3. **`core/` 与 `engine/` 不 import `transformers`**。缺了它框架核心照样能跑。

---

## 一、字段命名与相位规则

整个训练步骤切成八个相位（A ~ H），**相位决定梯度**，而不是让每个组件自己声明
「我需要可微的输入」。下表是其中**会写字段**的 A ~ F；`G`（记录指标）与
`H`（存盘）不碰 Batch，两个家族完全共用。

| 相位 | 做什么 | 写入的字段 | 梯度 |
|---|---|---|---|
| **A** rollout | 生成序列 | `input_ids` `attention_mask` `response_mask` `rollout_logprobs` `group_ids` | 无 |
| **B** score | 打分 + 奖励处理链 | `rewards` | 无 |
| **C** prepare | 阶段前向，只算一次 | `ref_logprobs` `rollout_values` | 无 |
| **D** advantage | 算优势 | `advantages` `returns` | 无 |
| **E** 冻结 | detach + `freeze()` + 切梯度 | — | — |
| **F** train | 每个 mini-batch 内重算 | `curr_logprobs` `values` | **有** |

规则一句话：**名字以 `rollout_` 开头的字段，以及 B~D 相位的全部产物，一律
detached；只有 `curr_logprobs` 与 `values` 携带梯度。**

这条规则把「某个 loss term 误把 detached 张量当可微输入用」从「靠人记得」
变成「由相位保证」。想声明自己需要可微输入，用 `grad_fields`：

```python
class PPOClip(LossTerm):
    requires    = frozenset({CURR_LOGPROBS, RESPONSE_MASK, ADVANTAGES})
    grad_fields = frozenset({CURR_LOGPROBS})    # 合法值只有 curr_logprobs / values
```

`plan_assembly` 会在启动时静态校验：`grad_fields` 超出训练相位、或者声明了却
没写进 `requires`，都直接报错。因为这类错误的表现是**策略梯度项恒为 0、
loss 曲线正常下降** —— 不报错，也查不出来。

### 离线家族：八个相位塌成两个

样本不是策略采样来的，所以 **A（rollout）换成数据集、B（打分）与 D（优势）
整个消失**：

| 八个相位 | 离线家族 | 做什么 |
|---|---|---|
| A rollout | **P** 准备 | 从数据集取一个 batch（`BaseDataset.build_batch`） |
| C prepare | **P** 准备 | 需要时跑一遍 Reference 前向（detached，只算一次） |
| E 冻结 | **P** 准备 | `detach().freeze(FROZEN_BEFORE_TRAIN)` |
| F train | **T** 训练 | epoch × mini-batch 的可微循环（**与 RL 共用基类的 `_train_epochs`**） |
| G / H | G / H | 记录与存盘（**与 RL 共用**） |
| **B score** | — | **不存在** |
| **D advantage** | — | **不存在** |

最后两行就是「隐式信号不需要奖励模型」这句话在代码里的完整样子，而它不是一个
开关：离线家族传给 `plan_assembly` 的是 `phases=F.OFFLINE_PHASES`
（`{dataset, prepare, train}`），框架据此**推导**出「这个家族拿不到
`rewards` / `advantages`」。于是误写 `requires: {rewards}` 的离线损失项
在**装配时**就报「没人能提供 rewards」，而不是训练跑到一半才炸。

`phases` 之外的差别只有一个：`advantage=None`、`processors=()`。
两个家族用的是同一个 `plan_assembly`，没有第二份装配逻辑。

**「哪些字段是框架无条件许诺的」现在是推导出来的，不是断言的，这本身就是一次修正。**
早先 `_FRAMEWORK_PROVIDERS` 是一张大表，等于宣称「所有家族都能拿到所有字段」——
在那张表下，离线损失项写 `requires: {rewards}` 会通过装配检查，然后在训练时
炸在「Batch 里没有 rewards」上。现在是按相位分组的
`_FRAMEWORK_PROVIDERS_BY_PHASE`，一个家族能拿到什么由它跑哪些相位推出来。

### 形状约定

`advantages` / `returns` / `*_logprobs` / `values` **一律逐 token `[B, L]`**，
归约一律靠 `response_mask`。序列级的量（GRPO 的序列级奖励等）由产出方广播到
`response_mask` 为 1 的位置（`core.tensor_ops.broadcast_sequence_to_tokens`）。

这一条消除了「逐 token 还是逐序列」这个最常见的形状 bug。

### 字段名不要硬编码

全部从 `core.interfaces` import 常量。字段改名只会影响那一个文件。

### 为什么 `rollout_logprobs` 与 `curr_logprobs` 是两个字段

PPO 的 ratio 是 `exp(logπ_θ − logπ_old)`，其中 `logπ_old` 必须是 **rollout 那一刻**
的值，在整个 epoch × minibatch 循环里恒定。

如果第 2 个 epoch 用更新后的参数重算 `logπ_old`：不报错、形状对得上、
ratio 恒等于 1、裁剪彻底失效、KL 项权重被静默放大 —— 训练能跑、曲线好看、结果不对。

三道防线，每个都必要：

1. **字段名物理分离** —— `Actor.logprobs()` 只写 `curr_logprobs`，绝不碰
   `rollout_logprobs`；
2. **`freeze()`** —— 相位 E 冻结之后，写 `rollout_logprobs` 直接抛 `KeyError`；
3. **专项测试** —— `tests/test_old_logprobs.py` 跑两个 epoch，把 loss term
   实际看到的张量抓下来逐元素比对。这是整个仓库里最值钱的测试。

---

## 二、GRPO 的归一化归属

**归一化唯一归 `RewardProcessor`。`AdvantageEstimator` 不许自己再做一次。**

```yaml
reward:
  scorer:     {type: my_scorer}          # 产出原始奖励，只此一处入口
  processors: [{type: group_normalize}]  # 归一化在这里
algorithm:
  advantage:  {type: broadcast}          # 只做「奖励 -> 优势」的搬运
```

两处都做会**归一化两次**，优势被压成噪声级 —— 而 loss 曲线看起来完全正常。

所以 `RewardProcessor` 用 `normalizes_rewards = True` 自报，
`AdvantageEstimator` 用 `assumes_normalized_rewards = True` 自报，
`plan_assembly` 检查两者同时成立就报错。

> ⚠️ **`assumes_normalized_rewards` 这个名字是反的。** 字面读作「我假定奖励已被
> 归一化」，实际语义是「**我这一层也做归一化**」。语义方向看检查逻辑就清楚：
> 一旦链上已经有 `normalizes_rewards=True` 的 processor，本属性再为 True 就是
> 做了两次。按字面理解去设这个标志，会正好设反。名字是历史遗留（改名要动公开
> 属性名，还没做），**以其定义处的注释为准**：`core/base_advantage.py`。
>
> 按约定归一化归 processor，所以绝大多数 advantage 保持默认的 `False` ——
> `GAEAdvantage` 就是这样。

```
归一化冲突：processor ['processor/fixture_group_normalize'] 会对奖励做归一化，
而 advantage advantage/fixture_my_gae 又声明它假定奖励已被归一化。
两者同时成立会归一化两次，优势被压成噪声级，而 loss 曲线看起来完全正常。
```

相关的两个坑，也已经被框架处理：

- **组内归一化必须在 `split()` 之前完成。** mini-batch 里组是残缺的，
  `group_view()` 在残缺组上会直接抛错而不是给出错误的统计量。
- **`num_generations=1` 时不能用组内归一化。** 只有一条采样时标准差恒为 0，
  归一化会把奖励全部压成 0。用全局归一化。

---

## 三、`response_mask` 的 shift 只有一个出口

因果 LM 的 off-by-one：位置 `i` 的 token 由位置 `i-1` 的 logits 预测。

这个 shift **只允许在 `core/tensor_ops.gather_token_logprobs` 里发生一次**。

```python
# 正确：不需要自己 shift，输出已经与 input_ids 位置对齐
logprobs = gather_token_logprobs(outputs.logits, batch[INPUT_IDS])
```

任何组件自己再切一刀，就会出现「一处切了、一处忘了」—— 形状依然对得上
（都是 `[B, L-1, V]`），只是语义错位一位。不崩、不 warning，只会让 KL 各项
差一点点、让 ratio 的基准整体偏移。

`Batch.from_rollout()` 是**唯一**决定 `response_mask` 对齐方式的地方：

- prompt 左 padding，使所有 prompt 右对齐 → 生成从同一列继续；
- `response` 右 padding，长度取本批最大值；
- `attention_mask` 在真实 token 上为 1；
- `response_mask` 只在生成的 token 上为 1。

因为 prompt 左 padding，所有 response token 都在 `i >= Lp` 的位置，而
`gather_token_logprobs` 在位置 0 填的那个 0 永远不会落进 response 里 ——
两个约定正好互相接上。

`tests/test_no_manual_shift.py` 会静态扫描 `models/` 与 `components/`，
发现手工 shift 就报错。

### `rollout_logprobs` 为什么不从 `generate` 的 scores 里拿

HF 的 `generate(output_scores=True)` 返回的 logprob 与事后独立前向算出的
**数值上对不上** —— 两者走了不同的 kernel 与精度路径。后果是 step 0 第一个
mini-batch 上 `ratio != 1`，PPO 的裁剪从训练第一刻起就是偏的。

所以 `HFRolloutEngine` 在生成之后对完整序列做一次 `no_grad` 前向重算 logprob。
代价是每步多一次前向，收益是 `ratio` 精确等于 1。
`train/ratio_first` 这个指标就是用来盯这件事的：**它应该永远精确等于 1.0**。

（`recompute_logprobs=False` 这条路径被有意实现成直接抛 `NotImplementedError`
并说明理由，而不是悄悄用一个有偏差的实现。）

---

## 四、新增一个组件的 SOP

以「加一项自定义损失」为例，**四步，trainer 一行都不用改**。

> 想直接看成品：`src/components/losses/` 下的 `policy_gradient.py` /
> `value_loss.py` / `kl_penalty.py`（含 `kl_k1` / `kl_k2` / `kl_k3` 三个估计量）
> 就是按下面这套 SOP 写出来的，而且各自带了一段「为什么这么写」的注释。
> 改一个比从头写快。
>
> 想看**最小**的成品：`cross_entropy.py`、`gspo.py` 与 `ppo_clip.py` 都只有几十行
> —— 第一个是五项就能写完的 `−masked_mean(curr_logprobs, mask)`，第二个是一个
> 独立的损失项，第三个是 `policy_gradient` 的子类（只改两个默认值）。
> 「加一项自定义损失」的实际工作量就是这个量级。
>
> `dpo.py` 稍长一些，因为它要讲清楚三件容易写错的事（用 `masked_sum` 而不是
> `masked_mean`、配对靠 `preference` 而不是行序、一个对必须整对进同一个前向）。
> 但它的**代码**仍然只有二十几行，其余全是注释。

**1. 建文件** `src/components/losses/my_custom.py`：

```python
from core import interfaces as F
from core.base_loss import LossTerm
from core.batch import Batch
from core.registry import register
from core.tensor_ops import masked_mean

import torch


@register("loss", "my_custom")          # 双参：category + name
class MyCustomLoss(LossTerm):
    requires    = frozenset({F.CURR_LOGPROBS, F.RESPONSE_MASK, F.ADVANTAGES})
    grad_fields = frozenset({F.CURR_LOGPROBS})   # 我要哪个字段可微

    def __init__(self, coef: float = 1.0) -> None:
        super().__init__()
        self.coef = float(coef)

    def name(self) -> str:
        return "my_custom"

    def compute(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        mask = batch[F.RESPONSE_MASK]
        loss = self.coef * masked_mean(batch[F.CURR_LOGPROBS] ** 2, mask)
        # 指标里的值记得 detach —— 它们只是给人看的
        return loss, {"mean_sq": float(masked_mean(batch[F.CURR_LOGPROBS], mask).detach())}
```

**2. 在 `src/components/losses/__init__.py` 里 import 它**（触发注册）。

**3. YAML 里加一行**：

```yaml
algorithm:
  losses:
    - {type: my_custom, weight: 0.3}
```

**4. 跑测试** `pytest tests/test_grad_flow.py -q` —— 它会自动发现这个新组件，
喂一个随机批次、真的做一次 `backward()`，断言 `grad_fields` 里的每个字段
都拿到了**非零且有限**的梯度。

> ⚠️ 第 2 步漏了不会报错，只会让新组件**隐形**：`test_grad_flow.py` 少参数化
> 一个用例，全部测试照常通过，而那个组件从未被检查过。所以
> `tests/test_registry.py::test_every_declared_component_is_actually_registered`
> 会用 AST 扫描 `src/components/**` 里所有 `@register` 声明，逐个查注册表 ——
> 漏了第 2 步它会红。这条守卫是补上的，原因是这条路径真的漏过一次。

### `LossTerm.compute` 是纯函数

不调模型、不 backward、不碰 optimizer。这不只是洁癖：如果 term 自己调 actor，
它就要自己决定 `no_grad` 与否、要不要 `zero_grad`、怎么和 trainer 的
`clip_grad_norm_` 配合 —— 这些是 trainer 的职责，散出去必然出现「某个 term
多调了一次前向导致显存爆掉」或「梯度被意外的 `zero_grad` 清掉」。

确实需要额外前向的合法需求（双前向 KL 估计、需要 reward model 打分的 term），
用 `Component.prepare(batch, ctx)` 钩子，`ctx` 暴露 actor / critic / reference
的只读引用。注意它**不参与**模型构建的推导（那是静态的），所以如果你在
`prepare` 里用了 `ctx.reference`，必须同时让 Reference 被构建，否则运行期拿到 `None`。

### 写新组件的常见坑

| 坑 | 后果 | 框架怎么帮你 |
|---|---|---|
| `requires` 写了 `"reward"`（漏了 s） | 运行到一半才炸 | 启动时报错并列出候选字段名 |
| `grad_fields` 声明了 `rollout_logprobs` | 策略梯度项恒为 0，loss 正常下降 | 启动时静态报错 |
| 归一化在 processor 和 advantage 各做一次 | 优势被压成噪声级 | 启动时报错 |
| 忘了在 `__init__.py` 里 import | 「组件未注册」但文件明明存在；更糟的是相关测试**静默漏掉**它 | 报错附带排查提示 + `test_registry.py` 的 AST 扫描 |
| 自己手工 shift 序列 | 形状对得上但语义错位一位 | 静态扫描 + 报错 |
| 按 padding 的**当前位置**而不是**来源位置**取 mask | 上一个 bug 的同族：形状对，语义错，不报错 | 无明显信号 —— 变长 mask 的对拍测试是唯一防线（见下） |
| 归一化属性 `assumes_normalized_rewards` 按字面理解设反 | 冲突检查失效，归一化做两次 | 该属性注释里写明了语义相反 |
| 指标里 `float(带梯度张量)` | `UserWarning` | `LossComposition` 统一 detach |
| 忘了声明 `transient_requires` | 中间张量占着显存，第一次跑就 OOM | 见下一节 |

### 显存：用后即弃的字段

`B=64 × n=8 × L=1024` 时中间张量有十几个 GB，这不是「以后再优化」的问题。
组件声明自己「只在执行期间需要」的字段：

```python
class MyGAE(AdvantageEstimator):
    requires = frozenset({F.REWARDS, F.ROLLOUT_VALUES, F.RESPONSE_MASK})
    transient_requires = frozenset({F.ROLLOUT_VALUES})   # 优势算完就不用了
```

trainer 的释放规则是「声明为 transient 的字段 **减去** 训练相位确实还需要
的字段」，所以 `rollout_logprobs` 即便被某个 advantage 声明为 transient，
只要 PPO 的 loss term 还需要它就不会被释放。规则自动推导，不需要手工维护。

---

## 五、category 对应表

`@register(category, name)` 的第一个参数。键是 `(category, name)` 二元组，
所以 `loss/kl` 与 `processor/kl` 可以同时存在。

| category | 基类 | 读 | 写 | YAML 里的位置 |
|---|---|---|---|---|
| `rollout` | `RolloutEngine` | `prompt_texts` | `input_ids` `attention_mask` `response_mask` `rollout_logprobs` `group_ids` | `rollout` |
| `dataset` | `BaseDataset` | **无** | `input_ids` `attention_mask` `response_mask` `group_ids`（+ 成对数据额外提供 `preference`） | `data` |
| `scorer` | `Scorer` | `prompt_texts` `response_texts` | `rewards` | `reward.scorer` |
| `processor` | `RewardProcessor` | `rewards` | `rewards`（就地改写） | `reward.processors[]` |
| `advantage` | `AdvantageEstimator` | `rewards` `response_mask` `rollout_values` | `advantages` `returns` | `algorithm.advantage` |
| `loss` | `LossTerm` | `curr_logprobs` `response_mask` … | 无（只产出标量） | `algorithm.losses[]` |
| `controller` | `Controller` | **无** | **无**（改写另一个组件的 Python 属性） | `algorithm.controllers[]` |
| `actor` | `Actor` | `input_ids` `attention_mask` | `curr_logprobs`（**可微**） | `actor` |
| `critic` | `Critic` | `input_ids` `attention_mask` | `rollout_values` / `values` | `critic` |
| `reference` | `Reference` | `input_ids` `attention_mask` | `ref_logprobs`（detached） | `reference` |
| `logger` | `Logger` | — | — | `loggers[]` |
| `checkpointer` | `Checkpointer` | — | — | `checkpointer` |

几个要点：

- **`scorer` 与 `processor` 分开**，是为了让「奖励怎么算」和「奖励怎么加工」
  各自独立可换。替换归一化策略不需要碰打分逻辑。
- **`actor` 与 `rollout` 是两个 category、一个模型对象。** trainer 只加载一次
  权重，然后把 `actor.module` 交给 rollout 共享 —— 于是不存在权重同步，
  也不存在「两份权重不一致」。
- **`normalizes_rewards`（processor）与 `assumes_normalized_rewards`（advantage）**
  是给冲突检查用的自报属性。注意后者的名字与语义相反（语义是「我也做归一化」），
  见[第二节](#二grpo-的归一化归属)。
- **`model` 是唯一一个「不是 category、但同样被注入」的东西。** `actor` /
  `reference` / `critic` 都从 `model:` 段拿 `model_config`，`scorer`（当它是
  `hf_reward_model` 时）从 `reward_model:` 段拿 —— 两者是同一个模式：
  组件声明要什么，框架在构建时注入。`hf_shared_value` 更进一步，它要的是
  actor 的**底座对象**本身，那个通过 `needs_actor_backbone` 类属性声明
  （不是 `requires` —— `requires` 是 Batch 字段依赖，参与装配计划推导；
  这是构建期的事，两者不能混）。
- **`advantage` 那一行的「读」里写着 `rollout_values`，这不是通用的** ——
  它正是 PPO 与 GRPO 的分界线。GAE 声明要读它，Critic 才会被构建；
  广播式的 advantage 不声明，Critic 就不存在。见[第六节](#六组件--模型构建映射表)。
- **`dataset` 与 `rollout` 是同一个位置的两种实现**：两者都提供
  `input_ids` / `attention_mask` / `response_mask` / `group_ids`，只是一个从
  策略采样、一个从固定数据集来。所以它们**都走 `Batch.from_rollout`** ——
  全仓库只有那一个地方决定 `response_mask` 怎么对齐。离线侧另写一条 padding
  路径不会报错，只会让对齐差一列、loss 与 RL 侧对不上。
- **`dataset` 的 `requires` 恒为空**，这不是「还没写」。数据集是 Batch 的**起点**，
  不读任何字段，所以它永远不可能是「某个模型被加载」的理由 —— 否则一个数据集
  组件就能把 Reference 拉起来，而那显然说不通。
- **`controller` 是这个设计里唯一一个「读/写都为空」的 category**，这不是遗漏。
  控制器的唯一动作是改**另一个组件的 Python 属性**（把新的 β 写回某个
  `LossTerm.coef`），它完全看不见 `Batch`。如果给它声明
  `requires={ref_logprobs}`，它会**错误地成为「拉起 Reference 模型」的原因** ——
  而它读的其实是别的东西算出来的指标。所以它只能靠鸭子类型检查目标项，
  `components/controllers/` 也因此不 import `components/losses/`
  （有一条 `test_no_cross_imports.py` 在守着）。

  这也是为什么「adaptive-KL PPO」这个变体**不需要新的 category 之外的任何东西**：
  它是 `algorithm.controllers` 下面的一个列表项，而 `algorithm` 本来就在
  那三个允许差异的块里。

---

## 六、组件 → 模型构建映射表

`plan_assembly()` 把「所有存活组件声明的 `needed` 并集」翻译成「要构建哪些模型」。
**这是「换算法 = 改 YAML」得以成立的机关。**

| 需要这个字段 | 于是必须构建 |
|---|---|
| `ref_logprobs` | **Reference** |
| `rollout_values` 或 `values` | **Critic** |
| 其它一切字段 | 无需额外模型（框架自己产出） |
| — | **Actor 永远构建** |

这张表在两个家族里是同一张，只是查表的输入不同。最干净的一对例子就在
`configs/` 里：

| 配置 | 损失项的 `requires` | → 构建的模型 |
|---|---|---|
| `sft.yaml` | `{curr_logprobs, response_mask}` | `['actor']` |
| `dpo.yaml` | `{curr_logprobs, ref_logprobs, response_mask, preference}` | `['actor', 'reference']` |

两个 YAML 里都**没有** `reference` 那一行 —— 它在 `configs/model/fixture.yaml`
里一直挂着（模型段在基座配置里，不在算法配置里）。
`plan.py --all` 会把这两段并排打出来：

```
sft.yaml                                  dpo.yaml
  将构建的模型：['actor']                    将构建的模型：['actor', 'reference']
    （没有组件需要 ref_logprobs，               reference <- ['loss/dpo']
      Reference 不会加载）
```

这就是「条件依赖」第二次成立，也再次说明「配置里写了某一行 ≠ 那个模型会被加载」。

推导顺序（不能颠倒）：

1. 构建**纯逻辑组件**（scorer / processor / advantage / loss term / controller）。
   它们很便宜 —— 只是 `__init__` 读配置，不加载任何权重。
2. 剔除**权重为 0** 的 loss term。这一步必须在求并集**之前**做，否则 YAML 里留一行
   `{type: kl_k3, weight: 0.0}` 会白白把 Reference 拉起来。
3. 遍历存活组件的 `needed`（**实例属性**，不是类属性 `requires`），求并集。
4. 剥掉训练相位字段（`curr_logprobs` / `values`）再查表，否则会得到
   「需要名为 `curr_logprobs` 的模型」这种荒谬结论。
5. 检查「被需要但无人提供」的字段，启动即失败。

### 两种「关掉一项」的方式，以及它们的代价

| 写法 | 效果 | 注意 |
|---|---|---|
| `weight: 0.0` | 这一项在**装配阶段**就被剔除；`compute()` 不会被调用 | 它的依赖也一并从并集里消失 |
| `coef: 0.0` | 这一项**仍在**组合里、每步仍被调用（返回 0） | 只释放它自己声明退掉的那几个字段 |

`coef: 0.0` 走的是第 3 步里那个「实例属性」的机制：组件在 `__init__` 里按配置
调 `self.release(F.REF_LOGPROBS)`，于是它的 `needed` **小于**类属性 `requires`。
`configs/rl/dapo.yaml` 用的就是这一条（DAPO 不用 KL 项），所以它的 Reference
根本不会被构建 —— 尽管 `kl_k3` 那一行还在配置里。

⚠️ **退掉依赖的一方必须同时停止读它。** `release` 只是改了装配计划的输入；
如果 `compute()` 还在无条件 `batch.require(REF_LOGPROBS)`，那个字段会因为
「装配计划认定无人需要」而根本没被算出来 —— 于是配置合法、装配计划自洽、
文件能加载，跑到运行期才炸。这个 bug 真实发生过一次，现在由
`tests/test_kl_estimators.py::test_zero_coef_term_runs_without_ref_logprobs`
钉住。

⚠️ **`coef: 0.0` 的项不能当控制器的目标。** `adaptive_kl` 要把 β 调上去时，
读的是它已经退掉的 `ref_logprobs`。`plan_assembly` 会在启动时拦住这种绑定
（要求目标项的 `coef > 0`），而不是等到训练中途 `KeyError`。

启动时打印的 `plan.describe()` 会说明**是哪个组件导致了哪个模型被构建**：

```
装配计划：
  将构建的模型：['actor', 'critic', 'reference']
    critic <- ['advantage/gae', 'loss/value_loss']
    reference <- ['loss/kl_k3']
  依赖字段并集：[...]
```

这条可观测性是必要的：奖励处理阶段的一个 processor 拉起一个几十亿参数的模型
是反直觉的，必须让人一眼看到原因。

### 为什么 `needed` 是实例属性

类属性 `requires` 只能表达**无条件**依赖。但真实需求里有一大类条件依赖：

```python
class KLPenalty(LossTerm):
    requires = frozenset({F.CURR_LOGPROBS, F.REF_LOGPROBS, F.RESPONSE_MASK})

    def __init__(self, coef: float = 1.0) -> None:
        super().__init__()
        self.coef = coef
        if coef == 0.0:
            self.release(F.REF_LOGPROBS)    # 系数为 0 时不该拉起 Reference
```

这些依赖只有在**读到配置之后**才知道。所以 trainer 的顺序是：
**先建便宜的组件 → 问它们要什么 → 再建贵的模型。**

---

## 七、离线家族：SFT 与 DPO

### 两个决策，先说清楚

**决策 1：`train_step()` 的单位是「一个 batch」，不是「一遍数据集」。**

`OfflineTrainer.train_step()` 取**一个** batch（大小 = `data.batch_size`），
内部再按 `algorithm.mini_batch_size` 切、按 `algorithm.epochs` 过几遍。
数据集过几遍是**调用方**的事 —— 与 `RLTrainer.train_step(prompts)` 里
「喂哪批 prompt 是调用方的事」完全对称。

这么定不是为了对称好看，是为了**避免一个名字两种语义**：`algorithm.epochs`
在 RL 里的意思是「同一批 rollout 数据复用几遍」（这正是 `rollout_logprobs`
必须冻结的原因）。如果离线侧把它解释成「数据集过几遍」，同一个配置键在两个
家族里含义就不同了。现在它两边都是「同一个 batch 过几遍」。

**决策 2：成对样本必须整对地进同一个 mini-batch。**

DPO 的损失是 `−log σ(β(Δ_w − Δ_l))` —— **一个对必须在同一次 forward 里**，
因为损失是两行 logprob 的差。按行 `split(shuffle=True)` 会把一个对切到两个
mini-batch 里，那半个对算不出损失。

「要不要按组切」由**损失项自己声明**（`LossTerm.needs_intact_groups`），
不由配置开关控制 —— 与 `RewardProcessor.changes_batch_size` 同一套做法：
写这个损失的人知道它需要什么，配 YAML 的人不一定。`mini_batch_size` 不是组大小
的整数倍时**直接报错**，不静默丢半个对。

### 数据

| 注册名 | 每行 | 一条样本占几行 |
|---|---|---|
| `jsonl_sft` | `{prompt, response}` | 1 |
| `jsonl_preference` | `{prompt, chosen, rejected}` | 2（chosen + rejected） |

两者共享 `JSONLDataset` 基类（读文件 / 字段映射 / 长度过滤 / 取样顺序 / 去重），
差别只有**编码什么**与**提供什么**。取样顺序在构造时按 seed 定下来，所以游标
只要一个**整数**：

```python
indices, cursor = dataset.next_batch_indices(cursor, size)
```

**游标单调递增，不取模。** 它的含义是「到目前为止一共消费了多少个样本」，
绕回是**取下标时**做的（`cursor % total`）。两者必须分开，否则「刚好走完一整圈」
的存档会记成 `sampler_pos = 0` —— 与「从没训过」无法区分，而那正是 resume 时
最需要分辨的一刻。游标存进 `TrainerState.sampler_pos`，不存的话续跑会从数据集
开头重训，**从 loss 曲线上完全看不出来**。

分词用 `tokenizer.apply_chat_template(..., add_generation_prompt=True)` 拿
prompt ids，再单独编码 response，最后走 `Batch.from_rollout` 对齐。
分词器是**注入**的（`dataset.bind_tokenizer(actor.tokenizer)`），不是构造参数：
数据集要在 `plan_assembly` **之前**建好（它以 `extra_components` 的身份参与依赖
推导），而 tokenizer 来自 actor，actor 在 `plan_assembly` **之后**才加载。
注入之前调 `build_batch` 会拿到一条说明这一步的错误，而不是裸的 `AttributeError`。

### DPO 的三个容易写错的地方

```python
L = -E[ log σ( β (Δ_w − Δ_l) ) ]
Δ_y = Σ_t ( log π_θ(y_t) − log π_ref(y_t) )     # 逐 token 累加，分母是 1
```

1. **`masked_sum` 而不是 `masked_mean`。** Δ 是一个 token 一个 token 累加的。
   写成 mean 会得到「每条回答的平均对数比」—— 形状对、数值小一个 token 数
   因子、不会报错、loss 照样降，只是训练的其实是另一个目标。
   有测试专门钉住：两个有效 token 的 `Δ = (1-0)+(2-0) = 3`，不是 1.5。

2. **配对的依据是 `preference` 字段，不是行在组里的位置。**
   `(Δ · sign)` 在组内求和，`sign` 是 `+1`（chosen）或 `-1`（rejected），
   所以结果恒等于 `Δ_w − Δ_l`，与两行的先后无关。靠位置（「第 0 行是 chosen」）
   的写法在任何一次 `split` / `filter` 打乱行序之后会**静默地训反**。
   有测试交换一个对内的两行，断言损失逐位不变。

3. **一个对必须在同一次前向里。** 见上面决策 2。切散了的话 `group_view`
   会抛错；如果实现成「切散了也照算」，损失会基于两个不相干的样本 ——
   而 `Δ_w − Δ_w = 0` 时 loss 恒为 `log 2`、梯度恒为 0，训练悄悄停摆。

### 四个长得像但完全不同的参数

| 参数 | 在哪 | 含义 |
|---|---|---|
| `weight` | YAML，所有损失项都有 | 这一项在总损失里的**组合权重**，由 `LossComposition` 管 |
| `beta` | `DPOLoss` 构造参数 | sigmoid 内部的**温度**，控制偏离参考模型的惩罚力度 |
| `label_smoothing` | `DPOLoss` 构造参数 | 把标签从 hard 0/1 软化，抗噪声偏好 |
| `coef` | KL 项的参数 | KL 惩罚的强度，DPO 没有这个参数 |

把 `beta` 写成 `0.01` 是「允许偏离参考模型很多」，把 `weight` 写成 `0.01` 是
「这一项只占 1% 的梯度」—— 后者看起来像学习率太小，实际上是把目标函数改掉了。

`beta=0` 是一个诚实的退化：`σ(0) = 1/2`，损失恒为 `log 2`，
**且对 `curr_logprobs` 的梯度恒为 0**。有测试直接断言这两件事。

### 配置

```yaml
# configs/dpo.yaml（sft.yaml 只差 data.type 与 losses）
defaults: [base, model/fixture]     # 框架公共 + 基座（注意路径相对本文件所在目录）
trainer: {kind: offline, run_name: dpo}
data:
  type: fixture_jsonl_preference   # 换成 jsonl_preference + 真实路径即可
  path: memory://preference
  batch_size: 4                    # 4 **对** = 8 行
algorithm:
  losses:
    - {type: dpo, weight: 1.0, beta: 0.1, label_smoothing: 0.0}
  epochs: 2
  mini_batch_size: 4               # 行数，必须是 2 的整数倍
```

`data.batch_size` 挂在**数据集**上而不是 trainer 上：多大的 batch 装得下，
取决于数据集本身（10 行的玩具数据与 100 万行的语料，合适的值不可能一样）。

`trainer.kind` **不做成注册表 category**。注册表装的是 `Component`：它们都吃
`Batch`、都能被 `plan_assembly` 推导依赖。训练器不是 —— 它**产生** Batch，
是整个流程的宿主。硬塞进去，「组件」这个词就被用坏了。
分派逻辑在 `engine/build.py`，只有三行。

### 实测行为

```
$ python -X utf8 scripts/train.py -a dpo -m fixture --steps 6
dpo.yaml  |  OfflineTrainer（kind=offline）
  基座：tiny-fixture（actor=fixture_tiny，critic=fixture_tiny_value）
  ...
  dataset/fixture_jsonl_preference  样本数=6  组大小=2  provides=[... 'preference' ...]
step    1  loss=   0.6796  grad_norm=  0.9643
...
step    6  loss=   0.3836  grad_norm=  0.6217
```

（`train_offline.py --config dpo` 打的是同一份东西，额外多一列**数据游标** ——
续跑相关的细节在那边的输出里看得更清楚。）

第一步的 0.6796 贴近 `log 2 = 0.6932` —— 训练开始前模型对 chosen / rejected
没有系统性偏好，损失应当就在这个值附近。这是「损失尺度对不对」的 sanity check：
第一次算出来是 5 或 0.01，多半是 Δ 少乘/多乘了一个 token 数。

**但 loss 下降本身说明不了什么** —— 一个 `Δ` 恒为 0 的退化实现也有一条平坦
的曲线。所以真正盯的是 `loss/dpo/implicit_reward_gap`（即 `β·margin`）：
它在训练中应当上升。`verify_e2e.py` 的那条 DPO 检查断言的正是这个量，
而不是损失。

---

## 组件速查

### `Batch` —— 全框架唯一的数据契约

`tensors` / `non_tensors` / `meta` / `frozen` 四个 dict。用 dict 而不是 dataclass
字段，是「自定义组件能加自定义字段」的直接要求。

方法分两类语义，务必分清：

```python
# 返回新对象
select / filter / filter_groups / repeat_interleave / repeat
concat / group_by_prompt / split / chunk / clone

# 就地修改并返回 self
to / detach / cast / pin_memory / freeze / unfreeze / drop
```

`select` 与朋友们一律返回**拷贝**而不是视图 —— 视图会产生「改子 batch 污染
父 batch」这种隐形数据污染，而研究原型下这点拷贝成本完全可以接受。

**行序契约**：`num_generations=n` 时行序必须等价于
`repeat_interleave(prompts, n)`，组是连续块，`group_index = row // n`。
`group_by_prompt` 走连续块快路径；若检测到 `group_ids` 非连续会退化为散射
分组并给出 warning（这通常意味着某个组件打乱了行序）。

### 组件的四个可覆写方法

```python
class MyComponent(Component):
    requires:   ClassVar[frozenset[str]]   # 读哪些字段
    provides:   ClassVar[frozenset[str]]   # 写哪些字段
    transient_requires: ClassVar[frozenset[str]]   # 用完即弃的字段

    def require(self, *fields) / release(self, *fields)   # 实例级调整依赖
    def metrics(self) -> dict[str, float]   # 自报指标，会被 logger 记录
    def prepare(self, batch, ctx) -> Batch  # 需要额外前向时的钩子
    def state_dict(self) / load_state_dict(self)   # 需要进 checkpoint 的额外状态
```

### checkpoint

`TrainerState` 一次打包五类对象（actor / critic / optimizer / lr_scheduler /
组件状态）+ 四路 RNG + 数据游标 + **配置指纹**，`Checkpointer` 一个点替换。

- **只能在 step 边界存。** `train_step` 内部是半完成状态，存下来无法正确恢复。
- **resume 时校验配置指纹。** 从 grpo 换到 ppo 却加载旧存档，权重与配置对不上，
  会以极难排查的方式失效 —— 所以直接报错。指纹只覆盖 `algorithm` / `reward` /
  `model` 三块，改 `run_name` 不会让 resume 无谓失败。
- **RNG 四路都要存**（python / numpy / torch / cuda）。少了它们，续跑出来的
  采样序列与不中断时不同，实验不可复现。

### 离线数据

离线家族的数据集层见[第七节](#七离线家族sft-与-dpo)。它和 `rollout` 是同一个
位置（相位 A）的两种实现，都产出 `input_ids` / `attention_mask` /
`response_mask` / `group_ids` 并走 `Batch.from_rollout` 对齐。

离线 **rollout 缓存**（把生成结果存下来复用）仍是预留的扩展点。接入方式是自定义
一个 `RolloutEngine`：`generate()` 的返回契约就是「一个满足
`ROLLOUT_OUTPUT_FIELDS` 的 `Batch`」，所以读缓存与真采样在上层看来完全一样。

### LoRA / PEFT

也是预留的接缝。关键点在 `Actor.state_dict()`：默认返回全参，LoRA 实现覆盖它
只返回 adapter + `base_model_hash`。于是「全参存盘」与「只存 adapter」的差异
只体现在这一个点上。

⚠️ `AdapterDisabledReference`（关掉 adapter 当参考模型）**只允许在
「actor 冻结、只训 adapter」的场景使用**。如果 actor 是全参微调的，参考策略会
随 actor 一起漂移，`rollout_logprobs - ref_logprobs` 变成近似自己减自己，
KL 惩罚恒接近于 0 —— 训练静默失效，而 loss 曲线看起来正常。

---

## 验证与已知限制

### 已验证

```
pytest tests/ -q                         582 passed
python scripts/verify_e2e.py --stage 1   31/31 通过（十个配置各跑满）
python scripts/plan.py --all             十个配置各一段
```

`configs/*.yaml` 指的是 `src/components/` 下的**真实算法实现**，所以阶段 1 跑的
不是「PPO 形状的 fixture」，而是十个配置各自真的构造、真的训三步、真的存读
checkpoint。下面是八个 RL 配置在第 1 步与第 6 步的实测指标（4 条 prompt，微型模型，
`train/ratio_first` 一列全部**精确**为 1）：

| 配置 | `advantage/std` | `clip_frac` | `seq_clip_frac` | `grad_norm` |
|---|---|---|---|---|
| `ppo` | 0.96 → 1.07 | 0.56 → 0.16 | — | 2.8 → 2.6 |
| `ppo_kl_k1` `ppo_kl_k2` | 同 ppo | 同 ppo | — | 同 ppo |
| `ppo_adaptive_kl` | 同 ppo | 同 ppo | — | 同 ppo |
| `grpo` | 1.008（恒定） | **0** | — | 1.5 |
| `dr_grpo` | **11.6 → 8.5** | 0 | — | **70 → 51** |
| `dapo` | 1.008（恒定） | 0.023 → 0 | — | 1.5 |
| `gspo` | 1.008（恒定） | — | **0.88 → 0.75** | 0.47 → 0.81 |

（三行「同 ppo」是字面意思：`kl_k1` / `kl_k2` 只换一个估计量，`adaptive_kl` 只多
一个控制器，它们对策略梯度项没有任何影响。这正是那三个配置的差量只有一个
叶子 / 一个块的可观测后果。）

三条值得看的：

1. **`ratio_first` 八个 RL 配置、每一步都精确为 1。** 这是「rollout 的 logprob 与训练
   前向算出来的是同一个数」的直接证据，也是 GAE 与 ratio 裁剪能成立的前提。
2. **`grpo` 的 `clip_frac` 恒为 0，`advantage/std` 恒为 1.008。** 前者说明「不裁剪」
   不是忘记配，后者说明组内归一化正好做了一次 —— 做两次会掉到 1e-3 量级。
   这两个数一起把 GRPO 的配方钉住了。
3. **`dr_grpo` 的 `grad_norm` 比 `grpo` 大三十多倍。** 这是 `divide_std: false` 的
   直接后果：不除 σ 之后奖励尺度回到原始量级（`reward_std` 11.8 对 1.03），
   梯度跟着放大。**这是一个真实的、需要重新调 `lr` 的差异** ——
   「改一个配置值」不等于「不用管别的」。README 里专门写这一条，是因为它比
   「差异只是一个布尔值」那句更有用。

`ppo` 的 `clip_frac` 从 0.56 降到 0.16 也值得一看：0.56 明显**高于**通常说的
健康区间（0.05 ~ 0.2），因为微型模型是随机初始化的，第 1 步的 ratio 本来就分散。
到第 6 步落回区间内。真实训练里它应当稳定在区间内小幅波动 —— 一直很高说明
`lr` 太大或者 `logπ_old` 被刷新了。

两个诚实的观察，都属于「看曲线时别误判」那一类：

- **`gspo` 的 `seq_clip_frac` 高达 0.75 ~ 0.88，这不是坏了。** GSPO 论文的
  `ε = 3e-4` 与 PPO 的 `0.2` **不在同一个尺度上**（一个是长度归一化后的序列比值，
  一个是逐 token 比值）。在未训练的微型模型上，序列比值 `s` 偏离 1 的幅度普遍
  超过 3e-4，所以大部分序列都落在裁剪区间外。如果拿 PPO 的 0.2 去读它，会误以为
  「GSPO 裁得比 PPO 狠得多」；两者根本不可比。想让它在这个 fixture 上不裁剪，
  把 `clip_eps` 调到 0.01 量级即可。
- **`gspo` 的 `grad_norm` 明显小于 `grpo`。** 同上：大部分序列被裁之后，
  目标函数对参数的依赖只剩裁剪边界那一段。真实训练里这个量会随 `ratio` 趋近 1
  而回升。
`frac_neg_adv` 在 0.3~0.6 之间（优势两个符号都有，没有退化）。

阶段 1 是**纯 torch 但真自动微分**：`tests/fixtures/tiny_lm.py` 是个约两万参数的
微型因果 LM，有真参数、走真 autograd。用假模型（返回常量张量的那种）能证明
「接口闭合、没有缺失依赖」，但**证明不了梯度正确性** —— 而这类框架里最危险的
bug 恰恰是「梯度悄悄为零」。所以这里必须是真模型。

覆盖的关键断言：

| 断言 | 在哪 |
|---|---|
| GRPO 配置不构建 Critic、PPO 构建 | `test_requires.py` `test_configs.py` |
| `train/ratio_first` 精确等于 1 | `test_train_step.py` `verify_e2e.py` |
| 两个 epoch 之间 `rollout_logprobs` 逐元素不变、`curr_logprobs` 变了 | `test_old_logprobs.py` |
| 每个 loss term 的 `grad_fields` 梯度非零且有限 | `test_grad_flow.py` |
| 损失的梯度只能流到训练相位的字段 | `test_grad_flow.py` |
| GAE 与朴素实现逐元素一致（**变长 mask，含区域内 padding**） | `test_gae.py` |
| padding 列的价值不泄漏进 TD 残差 | `test_gae.py` |
| 每个 `@register` 声明都真的被注册（SOP 第 2 步） | `test_registry.py` |
| 改子 batch 不污染父 batch | `test_batch.py` |
| checkpoint 存读续跑一致（含优化器状态与 RNG） | `test_checkpoint.py` |
| 配置指纹不符时拒绝 resume | `test_checkpoint.py` |
| `components/` 不横向 import、不反向依赖 `engine/` | `test_no_cross_imports.py` |
| 没有手工 shift（且扫描规则本身被反向测试过） | `test_no_manual_shift.py` |
| `configs/` 里的配置真的能构造出 trainer 并跑起来 | `test_configs.py` |
| **八个 RL 配置与默认链的差量都只落在三块里** | `test_configs.py` |
| 磁盘上每个可跑的配置都被 `iter_algorithm_configs` 覆盖（没有配置静默不跑） | `test_configs.py` |
| `ppo` 与 `rl/ppo` 指同一个文件、找不到时报出候选 | `test_model_axis.py` |
| **`-m` 换基座不留下旧基座的键，也不丢算法声明的键** | `test_model_axis.py` |
| **共享骨架 critic：值相等、对象不同、参数不共享、键不重叠** | `test_model_axis.py` `verify_e2e.py` |
| **只跑价值回归时 actor 一个参数都不许动**（deepcopy vs 别名） | `test_model_axis.py` |
| 声明 `hf_reward_model` 却拿不到基座时报出一句能看懂的话 | `test_model_axis.py` |
| **kl 变体与 ppo 的差量恰好是一个叶子路径** | `test_configs.py` |
| **`ppo_clip` 与 `policy_gradient` 的 PPO 默认值逐位相等** | `test_policy_gradient.py` |
| **三种 reduction 给出三个数，且 `seq_sum`/`token_mean` 只差一个批次标量** | `test_policy_gradient.py` |
| **三个 KL 估计量在真 KL 已知的分布上收敛到同一个值** | `test_kl_estimators.py` |
| **k2 收敛到的是有偏目标 `½(KL²+Var(ℓ))`，不是 KL** | `test_kl_estimators.py` |
| **`coef: 0` 的项真的不再读它释放掉的字段** | `test_kl_estimators.py` |
| **GSPO 与逐 token 裁剪在同一个 batch 上给出相反判断** | `test_gspo.py` |
| **GSPO 配上逐 token 优势会报错（不静默退化）** | `test_gspo.py` |
| **归一化不会被做两次（processor 与 advantage 的冲突检查）** | `test_requires.py` |
| **`dynamic_filter` 排在会做统计的 processor 之前（顺序被检查）** | `test_group_processors.py` |
| **空批次跳过：不推进 `global_step`、不报 loss、不调控制器** | `test_train_step.py` |
| **β 经过 save → resume 之后保持** | `test_controllers.py` |
| **README 里给的命令行真的能跑** | `test_scripts.py` |
| `train.py` 的 `-a` 分派与 `--model` 合并、续跑接得上 | `test_scripts.py` |

离线家族（第七节）新增的：

| 断言 | 在哪 |
|---|---|
| `RLTrainer` 里不再定义骨架方法（AST 扫描「没被复制成两份」） | `test_trainer_boundary.py` |
| 骨架基类的标识符里不出现 reward / rollout / preference / critic… | `test_trainer_boundary.py` |
| 第一步之后八个 RL 配置的 `_cfg_hash()` 逐字节不变 | `verify_e2e.py` + 实测 |
| SFT 配置**不构建** Reference、DPO 配置**构建** | `test_dpo.py` `test_offline_trainer.py` |
| 离线家族拿不到 `rewards` / `advantages`（`phases` 推导） | `test_requires.py` |
| **拿 SFT 数据配 DPO 损失，在装配期就被拦下** | `test_dpo.py` `verify_e2e.py` |
| **交换一个对内的两行，DPO 的损失逐位不变** | `test_dpo.py` |
| **Δ 是逐 token 累加（`masked_sum`）而不是均值** | `test_dpo.py` |
| **`beta=0` 时 loss 恒为 `log 2` 且梯度恒为 0** | `test_dpo.py` |
| `label_smoothing` 给出正的下界 `H(ε)`，极小值落在有限的 margin 上 | `test_dpo.py` |
| **一个对被切到两个 mini-batch 里，`compute()` 就报错** | `test_dpo.py` |
| `mini_batch_size` 不是组大小的整数倍时报错（不静默丢半个对） | `test_dpo.py` |
| 切分策略取自损失项的**并集**（加一个 `cross_entropy` 不会退回按行切） | `test_offline_trainer.py` |
| **游标单调递增、刚好走完一圈也不是 0**，进 checkpoint、resume 接得上 | `test_dataset.py` `test_offline_trainer.py` |
| 续跑取到的**下一批数据**与不中断时完全一致 | `test_offline_trainer.py` `verify_e2e.py` |
| 换 `data.path` / `beta` 之后指纹改变，resume 被拒绝 | `test_dpo.py` `test_offline_trainer.py` |
| `chosen == rejected` 的行被过滤且条数进指标 | `test_dataset.py` |
| 离线数据的对齐（prompt 左 padding / response 右 padding）与 RL 契约一致 | `test_dataset.py` |

### 未验证（诚实说明）

**`src/models/` 这一整层（HF 封装）没有在本机跑过。**

原因是本机环境问题，不是框架问题：安装的 torchvision 与 torch 2.14.0+cpu
ABI 不匹配（`RuntimeError: operator torchvision::nms does not exist`）。
transformers 的懒加载在导入任何建模类时都会连带 import `image_utils` →
torchvision，于是被挡住。

按约定**没有改动环境**（不重装、不修复 torchvision）。因此：

- `scripts/verify_e2e.py --stage 2` 在本机会**跳过**并打印原因；
  加上 `--require-hf` 时跳过变成失败（CI 上用这个）。
- 请在一台 torchvision 正常的机器上跑 `python scripts/verify_e2e.py`，
  它会用 `GPT2Config` 在内存里构造一个 1 层 1 头的 GPT-2（**不联网**）
  配一个手搓的离线分词器，验证：行序契约、`response_mask` 对齐、
  rollout logprob 与 actor 前向一致（`ratio == 1`）、reference 确实是独立副本、
  critic 的两种梯度语义、梯度能穿过 HF 模型。

### 哪些组件被真实验证过

| 层 | 状态 |
|---|---|
| `components/advantages/`（`gae` / `broadcast`） | **真实实现**，与朴素实现对拍 + padding 回归测试 |
| `components/processors/`（`group_normalize` / `dynamic_filter`） | **真实实现**，含「归一化只做一次」与链顺序的守卫 |
| `components/losses/`（九项） | **真实实现**，被 `test_grad_flow.py` 自动覆盖梯度 |
| `components/controllers/`（`fixed_kl` / `adaptive_kl`） | **真实实现**，含 β 进 checkpoint 的往返测试 |
| `components/loggers` `components/checkpointers` | **真实实现**（基础设施） |
| `data/jsonl.py`（`jsonl_sft` / `jsonl_preference`） | **真实实现**，读盘路径由 `tmp_path` 覆盖 |
| `components/scorers/` | 空 —— 任务相关，换任务才写 |
| `actor` / `critic` / `reference` / `rollout` | `configs/model/fixture.yaml` 指向 `tests/fixtures/` 的微型模型 |
| `src/models/`（HF 封装，含 `hf_shared_value`） | **已写，未在本机跑过**（见上一节） |

`components/losses/` 下的九个注册名：`policy_gradient`（GRPO / Dr.GRPO / DAPO 共用）、
`ppo_clip`（前者的子类）、`gspo`、`kl_k1` / `kl_k2` / `kl_k3`（共用一个基类）、
`value_loss`、`cross_entropy`、`dpo`。**「十个算法配置」与「九个损失项」
不是一一对应的** —— 这正是本框架的主张：算法数可以比组件数涨得快。

离线那两个损失项与前面七个**住在同一个目录、走同一条注册路径、由同一个
`LossComposition` 组合**。它们没有继承任何「离线损失」基类，也没有让 trainer
多一个分支 —— 差别只有 `requires` 里写了什么。

### 两个值得记下来的测试设计教训

写这套东西时踩到的，都留了守卫测试：

1. **「全部通过」可能是「一个都没检查」。** `tests/test_grad_flow.py` 在**收集
   阶段**枚举注册表，而 `conftest.py` 最初只 import 了 `tests.fixtures`、漏了
   `components`。结果 179 个测试全绿，而三个真实损失项一次都没被检查过梯度。
   现在 `conftest.py` 两条都 import，并且 `test_registry.py` 里加了一条 AST
   扫描 —— 任何写了 `@register` 却没被扇出 import 的类都会让测试变红。

2. **过宽的守卫测试会被关掉，那比没有更糟。** `test_no_manual_shift.py` 原本
   拦下任何 `[:, :-1]`，于是把两件不相干的事一并拦了：GAE 里
   `next_values[:, :-1] = values[:, 1:]`（把价值沿序列维右移去取 `V(s_{t+1})`），
   以及**组件文档字符串里讲解这个坑的示例代码**。现在规则是「切片 + 名字」
   同时匹配，并且扫描用 `tokenize` 跳过字符串字面量覆盖的行。

### 明确不做

分布式训练、vLLM / SGLang、权重同步、序列 packing、任何具体算法。
这是个**单机研究原型**的骨架。
>>>>>>> master
