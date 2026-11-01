# 本地（Windows + RTX 3060）用 uv 跑 rollout 消融

这份文档对应「没有 sm80+ Linux 机器、又想在本地拿到数」的场景：**不装 sglang**，用 uv 管住全部依赖，跑 `smoke` 与 `local` 两档。

sglang 本身在 Windows 上装不起来（没有 win 轮子，`sgl_kernel` / `flashinfer` / `triton` 都只发 Linux），所以本地能回答的是：

- 链路是否正确（`smoke`，CPU，20 秒）
- **logprob 复用能省多少**（`local`，GPU 真模型，与产线同一串 token 对照）
- 数值一致性（fp32 下两路 logprob 应完全相同）

「generate 本身的加速比」本地拿不到，那一项必须 sm80+ Linux。

---

## 1. 版本怎么定

本机：RTX 3060 Laptop 6G（sm86）、驱动 560.94 → CUDA 上限 **12.6**。

| 组件 | 版本 | 为什么 |
|---|---|---|
| Python | 3.12 | uv 自动下载，不碰系统解释器 |
| torch | `2.7.1+cu126`（**必须带 `+cu126` 后缀**） | 驱动 560 正好对应 12.6；PyPI 上的 `torch` 在 Windows 是 **CPU 版**，装了 `torch.cuda.is_available()` 会是 False |
| transformers | `4.57.1` | 4.x 分支，Qwen3 支持完整；5.x 也支持但本地没必要引入差异 |
| numpy | `<3` | torch 2.7 的 ABI 要求 |

### Windows 上最容易踩的一个坑

`uv pip install torch` 从 PyPI 装到的是 `2.7.1+cpu`，跑起来直接 `AssertionError: Torch not compiled with CUDA enabled`。Windows 的 CUDA 轮子只在 PyTorch 自己的索引里，而 `download.pytorch.org` 在部分网络下不通（本机就是：curl 6 秒超时）。用国内镜像的直链：

```bash
uv pip install --python .venv-ablate/Scripts/python.exe \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --reinstall-package torch \
    "https://mirrors.aliyun.com/pytorch-wheels/cu126/torch-2.7.1%2Bcu126-cp312-cp312-win_amd64.whl"
```

（2.7 GB。`%2B` 就是 `+` 的 URL 编码，别写成空格。）

驱动与 CUDA 的对应关系：`nvidia-smi` 右上角 `CUDA Version` 就是驱动支持的最高运行时版本。**装等于或低于它的 wheel**，装高了会在第一次 `.cuda()` 时报错，而报错信息通常不含「版本」二字，很难认。

---

## 2. 装环境

```bash
cd /d/work/RL-Training

# 1) 建 venv（uv 会把 CPython 也下好，不依赖系统 python）
uv venv --python 3.12 .venv-ablate

# 2) 装依赖（走清华源）。注意 torch 先别装，见下一段
uv pip install --python .venv-ablate/Scripts/python.exe \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    "transformers==4.57.1" "numpy<3" "omegaconf>=2.3" "PyYAML>=6.0"

# 3) torch 走 cu126 直链（PyPI 的 Windows 版是 CPU 编译的）
uv pip install --python .venv-ablate/Scripts/python.exe \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    "https://mirrors.aliyun.com/pytorch-wheels/cu126/torch-2.7.1%2Bcu126-cp312-cp312-win_amd64.whl"

# 4) 验证
.venv-ablate/Scripts/python.exe -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

只装这五个包，**不装** `requirements.txt` 里的 `datasets` / `modelscope` / `wandb` —— 消融用不到，省 1G+ 带宽。要跑训练再补。

`.venv-ablate` 已写进 `.gitignore`，不会污染 `git status`。

---

## 3. 两档消融

```bash
PY=.venv-ablate/Scripts/python.exe
export PYTHONPATH=src            # Git Bash；PowerShell 用 $env:PYTHONPATH="src"
export HF_HUB_OFFLINE=1          # 模型已在缓存里，断网也能跑，避免挂在 HEAD 请求上
```

### 3.1 smoke：链路正确性（CPU）

```bash
$PY scripts/ablate_rollout.py smoke
```

微型模型 + 全链路，验证 Batch 对齐、mask、group_ids、logprob 放置。期望 `max|Δ|` 在 `1e-6` 量级。

### 3.2 local：真模型量收益（GPU）

```bash
MODEL=~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca

$PY scripts/ablate_rollout.py local --model "$MODEL" \
    --batch-size 4 --num-generations 4 --max-new-tokens 96 \
    --dtype bfloat16 --repeats 3
```

6G 显存下的参数取舍：

| 参数 | 6G 上的建议 | 说明 |
|---|---|---|
| `--batch-size` | 4（8 也行，慢了就降） | prompt 条数；实际序列数 = batch × num-generations |
| `--num-generations` | 4 | 显存按 `batch×n` 增长，不是按 batch |
| `--max-new-tokens` | 96~128 | 越长越能放大差异，但时间线性涨 |
| `--dtype` | `bfloat16` | sm86 支持 bf16；`float16` 也行，但一致性对比要用 `float32` |

要看严格的数值一致性，加 `--dtype float32`（慢，但两路 logprob 应完全相等）。

OOM 时先降 `--batch-size`，再降 `--num-generations`；`--max-new-tokens` 对显存影响小、对时间影响大，别先动它。

---

## 4. 本地实测（2026-11-01，RTX 3060 Laptop 6G / torch 2.7.1+cu126 / bf16）

| 形状 | 后端 | 耗时 | tokens | tok/s | 比值 |
|---|---|---|---|---|---|
| 4 prompts × n=4 × ≤96 tok | hf（产线，重算 logprob） | 14.664 s | 1536 | 104.7 | 1.00× |
| 同上 | hf-batched-sim（复用采样 logprob） | 14.336 s | 1536 | 107.1 | **1.02×** |
| 8 prompts × n=4 × ≤128 tok | hf | 19.247 s | 4096 | 212.8 | 1.00× |
| 同上 | hf-batched-sim | 19.953 s | 4096 | 205.3 | **0.96×** |

- 重算前向单次耗时：0.29 s（1536 tok）→ 0.52 s（4096 tok），线性于 token 数
- 两路一致性 `max|Δlogprob|`：**3.62e-03**（bf16 + temperature=1、top_p=1，同一串 token）

### 怎么读这组数

**省下来的那部分 = 重算前向**，只有 0.3~0.5 s，占 rollout 的 **2~3%**。它随 token 数线性增长，但在 6G 卡上永远只是零头。

**比值会在 1 附近抖动，甚至掉到 0.96×**，原因是模拟路径的 `hf-batched-sim` 要把整批 `scores` 留在显存里（sglang server 就是这么返回的），6G 显存下这会触发更多的显存换入换出，把省下的一次前向吃掉。这不是「sglang 没用」，而是**用 HF 假装 sglang 时的失真**：真 sglang 的 logprob 是引擎内部算好的，不存在这份额外驻留。

所以本地这组数只能得出一个结论：**logprob 复用的收益在 2~3% 量级（小 batch）/ 更小（大 batch，显存压力上升）**。generate 本身的加速比本地测不出来，需要 sm80+ Linux 上真 sglang。

> 判读注意：`repeats` 太小时抖动会盖过信号。上面 4 prompts 那行是 `--repeats 3` 的均值，8 prompts 那行是 `--repeats 2`；两者不可直接横向比较不同形状的比值。
