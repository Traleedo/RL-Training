# SGLang 加速 rollout（实验分支 `sglang-rollout`）

## 一句话

`rollout` 从「和 actor 共享一个 HF 模型」换成「一个可插拔的推理后端」，默认实现仍是 HF，
新增 `sglang` 实现；框架契约、`Batch` 对齐规则、`num_generations` 分组顺序一个字没改。

## 为什么值得做

`time/rollout_s` 是 `train_step` 里最容易被忽略的一项。以 0.6B 策略 + `num_generations=8`
+ 512 token 为例，HF 的 `model.generate` 是**逐 token Python 循环**：每条序列 512 次前向，
8 条采样就是 4096 次。同一个 step 里，打分与反向的量级远小于它。

SGLang 用 paged KV cache + CUDA graph + 连续批处理把这段压下去；更关键的是它能在**采样时**
直接返回每个 token 的 logprob（`output_token_logprobs`），于是「重新前向一遍算 rollout_logprobs」
那次全序列前向也可以省掉。

## 设计

```
RolloutEngine (core/base_rollout.py)          ← 抽象：generate() 返回 Batch
├── HFRolloutEngine   ("rollout/hf")          ← 原有实现，与 actor 共享模型对象
└── SGLangRolloutEngine ("rollout/sglang")    ← 新增
        └── RolloutBackend (Protocol)
            ├── SGLangEngineBackend   同进程 sglang.Engine
            ├── SGLangHTTPBackend     sglang.launch_server 的 /generate
            └── HFMirrorBackend       本地 HF 生成（CI / 链路验证）
```

四条边界：

1. **引擎只做采样**。chat 模板、左 padding、`response_mask` 对齐、`group_ids` 连续块契约、
   logprob 放置，全部在 `models/rollout_common.py` 里，`hf` 与 `sglang` 共用同一份实现。
   这些规则一旦有两份，错的那份是**静默**错的：mask 少一格只会让梯度权重悄悄偏，
   logprob 错一位只会让 `train/ratio_first` 轻微偏离 1，看起来像数值误差。

2. **后端是 Protocol，不是基类**。测试与消融需要假后端，而假后端不该被迫继承生产基类
   —— 那会把「有没有实现全」变成继承检查而不是行为检查。

3. **sglang 是惰性依赖**。`import models` 不 import sglang：没装 sglang 的机器照样跑全部
   CPU 测试，只有真的选 `runtime=engine` 时才报错，并给出可执行的替代方案。

4. **权重生命周期是显式契约**。基类新增两个钩子：

   | 钩子 | 语义 | 默认 |
   |---|---|---|
   | `uses_external_weights` | rollout 是否持有独立于 actor 的权重副本 | `False` |
   | `sync_weights(actor_module)` | 每个优化步之后推送新权重 | no-op |
   | `close()` | 释放引擎 / 连接 / 子进程 | no-op |

   `RLTrainer` 在 `_train_epochs` 之后调用 `sync_weights`，并记 `time/weight_sync_s`。
   本地 HF rollout 与 actor 是**同一个模块对象**，这两个钩子都是零开销的 no-op，
   所以原有行为完全不变。

   > 为什么必须显式：不同步不会报错。训练到第 10 步时采样用的还是第 0 步的策略，
   > 而 `rollout_logprobs` 记的是「谁采的」—— PPO 的 ratio 从此恒等于 1，
   > 重要性采样静默失效，loss 曲线完全正常。所以 `weight_sync=none` 会在启动时打 WARNING。

## 三种 runtime

| runtime | 需要 | 权重同步 | 用途 |
|---|---|---|---|
| `engine` | Linux + CUDA + `pip install sglang[all]` | `update_weights_from_tensor`（全参） | 生产 / 性能实验 |
| `http` | 已起好的 `sglang.launch_server` | 不支持（后端返回 False 并告警） | 服务端独立部署 |
| `mirror` | 无（本地 actor） | 不需要（同一对象） | CI、链路验证、消融对照 |

## 用法

```bash
# 训练（需要 sglang）
python scripts/train.py -a sglang/grpo --steps 20

# 只验证链路：把 runtime 换成 mirror，不需要 sglang
python scripts/train.py -a sglang/grpo --steps 2 ...
```

配置在 `configs/sglang/grpo.yaml`，与 `configs/rl/grpo.yaml` **只差 rollout 块**
（有测试守着这条不变量）。它不在 `configs/rl/` 里，因为那不只是一个目录：
`tests/test_configs.py` 会对矩阵内每个配置真的构造 trainer 并跑一步，
而这一个多一个可选依赖，放进去会让「没装 sglang 的机器」的整套配置测试变红。

## 消融实验

两个轴：

| 轴 | 取值 | 想回答的问题 |
|---|---|---|
| `logprob_source` | `sglang` / `recompute` | 复用采样端 logprob 会不会改变语义？省下多少？ |
| `weight_sync` | `none` / `tensor` | 权重同步会吃掉多少收益？ |

### A. 正确性消融（CPU，可复现，无需 sglang）

```bash
python scripts/ablate_rollout.py smoke --batch-size 4 --num-generations 4 \
    --max-new-tokens 8 --repeats 5
```

方法上有一个必须控制的变量：两条 logprob 路径**必须吃同一串 token**。
若各自独立采样，比较的就是两段不同文本的 logprob，差异没有意义（首次实现就踩了这个坑，
量到 max |Δ| ≈ 1.4）。所以脚本先用微型模型采一次，把 token 与「采样时刻的 logprob」
一起记下来，再回放给两条路径。

实测（tiny 模型，仅代表方向，绝对耗时不具参考价值）：

| logprob 来源 | 每步耗时 | 相对 |
|---|---|---|
| `sglang`（复用采样端） | 0.003 s | 1.00× |
| `recompute`（actor 重算） | 0.007 s | 2.3× 慢 |

**一致性：max |Δlogprob| = 4.8e-7，ratio ∈ [1.000000, 1.000000]** —— 复用采样端 logprob
与重算在数值上是同一个量，所以省掉那次前向不会改变算法语义。

> 一个容易误判的细节：初次测量用了不等长 prompt，得到 max |Δ| ≈ 1.4。
> 原因不是对齐错位，而是 `tests/fixtures/tiny_lm.TinyCausalLM` 用 `arange(L)` 取位置嵌入、
> 不按 `attention_mask` 重算 position_ids —— 左 padding 会把真实 token 整体右移。
> 真实 HF 模型按 mask 算 position_ids，不存在这个问题。脚本里因此用等长 prompt
> 把这一项隔离掉，`PROMPTS_EQUAL` 上方有注释说明。

### B. 本地 GPU 消融（无需 sglang，Windows 可跑）

sglang 跑不了 Windows，但它的收益有一半在本地就能量出来：**server 顺手返回
采样时刻 logprob，省掉产线里「对完整序列再前向一遍」的重算**。`local` 档用
`BatchedHFBackend` 模拟这个返回契约（同一次 batched generate 里用
`output_scores` 取采样时 logprob），与产线路径同卡对比：

```bash
python scripts/ablate_rollout.py local --model Qwen/Qwen3-0.6B \
    --batch-size 8 --num-generations 8 --max-new-tokens 128 --repeats 3
```

三个产出：

- `hf` vs `hf-batched-sim` 的耗时差 = logprob 复用省下的部分；
- 差值 ≈ 单独量的重算前向耗时（`recompute_forward_s`，归因校验）；
- 真实权重上的一致性（fp32 小规模采样 + 重算，把 A 档微型模型的结论
  在真模型上复核一遍）。

**实测**（RTX 3060 Laptop 6GB，Qwen3-0.6B，8 prompts × n=8 × ≤128 tok）：

- 一致性：fp32 下 max |Δlogprob| = **3.6e-3** —— 放置逻辑正确。
  注意不要用 bf16 直接对拍：generate 走 KV-cache 核、重算走全序列前向，
  bf16 下两条核路径的舍入差能到 O(0.1~1)，那是精度噪声不是 bug；
- 重算前向：8192 tokens 约 **2.9~3.5s**，占产线 rollout 的 ~12-15%
  —— 这是 logprob 复用确定能省下的部分；
- generate 本身的加速比：**本档测不出来**。`hf-batched-sim` 依赖
  `output_scores=True` 保留全部分布（本例 ≈2.5GB），6GB 卡上显存压力
  把 generate 拖慢 3 倍 —— 这是 HF 模拟器的伪影（sglang server 返回
  logprob 几乎免费，且它有自己独立的显存），不代表 sglang 语义的成本。
  generate 侧收益必须用 C 档真 sglang 量。

附带产出：本档暴露了产线 recompute 的一个真 OOM —— 全 batch 的
`[B, L-1, V]` fp32 logits 一次性物化（64×150×151k ≈ 13GB）。已在
`core/tensor_ops.gather_token_logprobs` 改为按 batch 维分块，数值逐位等价。

### C. 吞吐消融（需要 GPU；sglang 需要 Linux）

```bash
python scripts/ablate_rollout.py bench --model Qwen/Qwen3-0.6B \
    --batch-size 8 --num-generations 8 --max-new-tokens 256 \
    --backends hf,sglang --runtime engine --logprob-sources sglang,recompute \
    --repeats 3
```

输出一张 markdown 表（rollout 墙钟、tokens、tok/s、相对 hf 的加速比），
同时写 `runs/ablation/bench-<时间戳>.json`。解读方式：

- **`sglang` vs `hf` 的加速比**：rollout 本身的收益；
- **`logprob_source=sglang` vs `recompute` 的差**：省掉全序列前向的收益（0.6B、512 token 时
  通常是这一步最大的一块）；
- **`weight_sync=tensor` 的 `time/weight_sync_s`**：每步要还的账，极端情况下能把收益吃光
  —— 这也是为什么它必须单独计时。

## 已知限制

1. **LoRA 不做权重同步**。只推 adapter 不会改变服务端底座，属「半同步」，比不同步更难查。
   `sync_weights` 检测到 LoRA 会明确拒绝并告警。全参微调才支持 `weight_sync=tensor`。
2. **`runtime=http` 没有张量级同步**。REST 端没有通用接口，`update_weights` 返回 False。
3. **`runtime=mirror` 的性能数据无意义**。它就是 HF 生成，只用来验证链路与做对照。
4. **同卡共处要留显存**。`mem_fraction_static` 默认 0.55；sglang 静态占用 + actor + 奖励模型
   同卡时容易 OOM，按实际调小或把奖励模型换小（0.6B 版）。
5. **Windows 不支持 `runtime=engine`**。sglang 依赖 triton/CUDA，本地只能跑 `mirror` 或纯测试。
