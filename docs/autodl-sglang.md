# 在 AutoDL 上跑 sglang rollout 消融

这份文档只解决一件事：**在 sm80+ 的卡上把 `sglang` 引擎真正跑起来，量出它相对 HF 产线 rollout 的加速比。**

Kaggle 那条路已经走到头（T4 = sm75，上游不再提供该架构的预编译 kernel），结论留在 `docs/sglang-rollout.md` 的附录里。这里是从零到出数的完整步骤。

---

## 1. 选机器

| 项 | 要求 | 说明 |
|---|---|---|
| GPU | **RTX 4090 24G**（sm89）或 A100 40G（sm80） | bench 档 24G 够；A100 便宜不了多少但显存宽裕 |
| 驱动 | **≥ 580** | 见下方「为什么卡驱动」 |
| 镜像 | Ubuntu 22.04 + 任意 Python ≥ 3.10 | 我们自建 venv，不用镜像里的 torch |
| 磁盘 | 系统盘 ≥ 30G，模型放 `/root/autodl-tmp` | venv + torch + sglang 约 12G；两个模型约 3G |

**不要选 2080Ti / V100 / T4 / P100。** 它们要么是 sm75 及以下（无预编译 kernel），要么没有 bf16，跑出来的数字没有参考价值。

### 为什么卡在驱动 580

`sglang==0.5.13.post1` 的依赖里有这两行：

```
flashinfer_python[cu13]==0.6.12
nvidia-cutlass-dsl[cu13]==4.5.2
```

`[cu13]` 表示这两个 wheel 是按 **CUDA 13** 编的，而 CUDA 13 要求驱动 ≥ 580。torch 本身（2.11 + cu128）只要 525+ 就能跑，但 flashinfer 会先炸，报错文本是 `CUDA driver version is insufficient for CUDA runtime version` —— 看起来像驱动没装，其实是版本不够。

AutoDL 上怎么看：实例详情里「驱动/CUDA」那一列，或者 `nvidia-smi` 右上角的 `CUDA Version` 要 ≥ 13.0。

驱动只有 550/570 的机器：换机器，别硬拼。真要在旧驱动上跑，退回 cu12 时代的 `sglang==0.5.9`（本仓库没有验证过这条路径，需要自己确认 API 兼容性）。

---

## 2. 版本矩阵

这套版本不是随手挑的，是**互相锁死**的：sglang 的 `pyproject.toml` 把 torch / transformers / kernel 全 pin 死了，任何一项单独升级都会 ABI 崩。

| 组件 | 版本 | 来源 | 为什么是这个 |
|---|---|---|---|
| Python | 3.12 | uv 自动获取 | sglang 要求 ≥ 3.10；kernel wheel 是 `cp310-abi3`，3.10+ 通用 |
| sglang | `0.5.13.post1` | 清华 PyPI | 4090 上有第三方实测锚点 |
| torch | `2.11.0` (cu128) | sglang 自动带入 | sglang 硬 pin，不要手动改 |
| transformers | `5.8.1` | sglang 自动带入 | 同上。本仓库已兼容 5.x，见第 6 节 |
| **sglang-kernel** | `0.4.3+cu129` | **docs.sglang.ai 官方索引** | PyPI 版只带 sm100 变体，A100/4090 加载不了 |
| flashinfer | `0.6.12[cu13]` | sglang 自动带入 | 需要驱动 580 |
| modelscope | `>=1.40.1` | 清华 PyPI | 下载权重 |

### 这张表里最容易错的一行

PyPI 上的 `sglang-kernel` 是给 Hopper/Blackwell 编的，wheel 里只有 `sm90/`、`sm100/` 目录。在 4090 上 `import sgl_kernel` 直接报：

```
[sgl_kernel] CRITICAL: Could not load any common_ops library!
GPU Info: Compute capability: 89 | Expected variant: SM89
Attempted locations: .../sgl_kernel/sm100/common_ops.abi3.so
```

**修法不是降版本，是换索引**：

```bash
uv pip install --no-deps --reinstall \
  "sglang-kernel==0.4.3+cu129" \
  --index-url https://docs.sglang.ai/whl/cu129/
```

这个索引里 0.4.0 → 0.4.6.post1 都有，每个版本带全架构构建。

---

## 3. 部署（三步）

### 3.1 传代码

```bash
# 本地
git push origin sglang-rollout
# AutoDL 上
cd /root/autodl-tmp && git clone -b sglang-rollout <你的仓库地址> RL-Training
cd RL-Training
```

### 3.2 一键装环境

```bash
bash scripts/setup_autodl.sh
```

脚本做四件事：检查驱动 → 装 uv → 建独立 venv → 装 sglang 并**替换 kernel**。全程不碰系统 python 和 conda。

想装到别处：

```bash
VENV=/root/autodl-tmp/venv-sglang bash scripts/setup_autodl.sh
```

末尾会打印 `sglang / torch / transformers` 的实际版本和下一步命令。看到 `sgl_kernel 加载成功` 就说明架构匹配上了 —— 这一步过了，后面基本不会再出环境问题。

### 3.3 下模型

```bash
source /root/autodl-tmp/venv-sglang/bin/activate
python scripts/download_models_ms.py --only policy
```

ModelScope 国内直连，Qwen3-0.6B 约 1.5 GB。bench 只需要策略模型。

---

## 4. 量加速比

先跑 CPU 档确认代码链路没坏（不需要 GPU，20 秒内出结果）：

```bash
python scripts/ablate_rollout.py smoke
```

再跑真模型：

```bash
python scripts/ablate_rollout.py bench \
    --model "$POLICY_PATH" \
    --backends hf,sglang \
    --runtime engine \
    --logprob-sources sglang,recompute \
    --batch-size 8 --num-generations 8 --max-new-tokens 256 \
    --mem-fraction 0.45 --repeats 3
```

其中 `$POLICY_PATH` 是上一步下载脚本打印的 `model.name_or_path:` 那一行（ModelScope 的缓存层级会随版本变化，别手拼路径）：

```bash
POLICY_PATH=$(python scripts/download_models_ms.py --only policy | sed -n 's/^  model.name_or_path: *//p')
```

参数含义：

| 参数 | 建议值 | 说明 |
|---|---|---|
| `--mem-fraction` | 24G 卡 0.45，40G 卡 0.55 | 即 `mem_fraction_static`，给 KV cache 的静态显存比例。设太高会 OOM |
| `--num-generations` | 8 | GRPO 的 group size，与算法配置对齐才有意义 |
| `--max-new-tokens` | 256 | 越大越能放大加速比（HF 是逐 token 解码，sglang 是批内并行） |
| `--repeats` | 3 | 取均值；第一次是冷启动，脚本已单独 warmup |
| `--engine-kwargs` | 一般不用传 | 若 attention backend 报错，加 `--engine-kwargs '{"attention_backend":"triton"}'` |

输出是一张 markdown 表 + `runs/ablation/bench-<时间戳>.json`。读数看三行：

1. **`sglang / hf` 加速比** —— 这次要量的核心数。HF 逐 token 解码，batch 8 × 8 generations 时理论上有数倍差距
2. **`logprob_source=sglang` vs `recompute`** —— 省掉的那次全序列前向。本地实测这部分占 rollout 12~15%，单独看是 ~1.1×
3. **`weight_sync=tensor` 每步耗时** —— 训练时每个优化步的额外开销，bench 里 `weight_sync=none` 不计入

> 判读注意：如果 `sglang` 那行反而比 `hf` 慢，先确认 `--num-generations` 不是 1、`--max-new-tokens` 不是 32 —— 小规模下 sglang 的进程内调度开销盖过收益，这是正常现象，不代表引擎没生效。

---

## 5. 完整训练（可选，先算显存）

bench 只装策略模型。训练档还要奖励模型，显存账完全不同：

| 组件 | 0.6B 策略 + **8B** 奖励 | 0.6B 策略 + **0.6B** 奖励 |
|---|---|---|
| actor (bf16) | 1.2 GB | 1.2 GB |
| reference (bf16, 冻结) | 1.2 GB | 1.2 GB |
| reward model | **16.1 GB** | 1.2 GB |
| sglang KV 静态（0.45 × 24G） | 10.8 GB | 10.8 GB |
| **合计** | **≈ 29 GB → 24G 卡装不下** | ≈ 14.4 GB → 可以 |

`configs/model/qwen3-0.6b.yaml` 默认的奖励模型是 8B。所以在 4090 上跑训练，**必须**换成小号：

```bash
python scripts/download_models_ms.py --only policy,reward-small
```

然后在配置里覆盖 `reward_model.name_or_path` 指向本地路径。或者直接上 A100 40G 用 8B。

训练命令：

```bash
python scripts/train.py -a sglang/grpo --steps 20 \
    --set reward_model.name_or_path=/root/autodl-tmp/models/Skywork/Skywork-Reward-V2-Qwen3-0.6B
```

训练档要额外装 TRL（bench 不需要）：

```bash
uv pip install "trl==1.16.0" "datasets>=2.19"
```

⚠️ TRL 1.16 是在 `transformers 5.16` 上验证过的，`5.8.1` 没测过。如果 import 报 API 不匹配，先确认是 TRL 还是本仓库的问题 —— 不要在没定位前就把 transformers 降回 4.x，那会让 sglang 起不来。

---

## 6. 本仓库为 transformers 5.x 做的两处适配

AutoDL 上会是 transformers 5.8.1，本机是 4.x。两处差异已修：

| 差异 | 现象 | 修在哪 |
|---|---|---|
| `apply_chat_template(tokenize=True)` 返回 `BatchEncoding` 而非 `list[int]` | `ValueError: invalid literal for int() with base 10: 'input_ids'` | `models/rollout_common.py::chat_prompt_ids` 归一化 dict/tensor/嵌套三种形态 |
| `from_pretrained` 的 `torch_dtype` 改名 `dtype` | 满屏 `FutureWarning`，容易误判成错误 | `scripts/ablate_rollout.py::_load_hf` 按大版本选关键字 |

`requirements.txt` 的 `transformers` 上限已从 `<5` 放宽到 `<6`，否则装 sglang（要 5.8.1）和装本项目会互相顶。

---

## 7. 排错

| 症状 | 原因 | 处理 |
|---|---|---|
| `Could not load any common_ops library! Expected variant: SM89` | kernel 还是 PyPI 那个 sm100 版 | 重跑第 3.2 节的替换命令 |
| `undefined symbol: _ZN3c104cuda29c10_cuda_check...` | kernel 与 torch 的 C++ ABI 不符 | 删 venv 重来，**不要**用 `--no-deps` 装 sglang |
| `CUDA driver version is insufficient` | 驱动 < 580，cu13 组件跑不了 | 换机器，或退回 sglang 0.5.9 |
| Engine 初始化 `CUDA out of memory` | `mem_fraction_static` 太高 | 0.45 → 0.30，或先只跑 bench |
| `ImportError: libcudart.so.13` | 容器里没装 CUDA 13 runtime | 装 sglang 时会带入 `nvidia-*` 包，确认没被 `--no-deps` 跳过 |
| `ModuleNotFoundError: No module named 'tvm_ffi'` | 缺 `apache-tvm-ffi`（包名与 import 名不同） | `uv pip install apache-tvm-ffi==0.1.9` |
| sglang 比 hf 还慢 | 规模太小，调度开销盖过收益 | 加大 `--max-new-tokens` / `--num-generations` |
| `chat template` 相关 ValueError | transformers 5.x 返回形态差异 | 确认代码是 `sglang-rollout` 分支最新版 |

---

## 8. 验收标准

跑完 bench 应该能回答三个问题：

- [ ] `sglang/hf` 加速比是多少（预期 2× 以上，取决于 `max_new_tokens` 与 batch）
- [ ] `logprob_source=sglang` 省掉多少时间（预期 10~15%）
- [ ] `weight_sync=tensor` 每步同步开销（0.6B 模型预期 < 50 ms）

把 `runs/ablation/bench-*.json` 拉回本地，直接替换 `docs/sglang-rollout.md` 里的 T4 数字，那份文档的结论就从「只验证了 logprob 复用」升级成「generate 加速已实测」。
