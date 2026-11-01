#!/usr/bin/env bash
# =====================================================================
# 任意 Linux + NVIDIA 机器上一键搭 sglang rollout 环境（AutoDL 只是其中之一）
#
# 用途：把「装 sglang」这件事里唯一容易踩死的两步固化下来
#   1) sglang 与 torch / transformers 的版本必须成套（它 pin 得很死）
#   2) PyPI 上的 sglang-kernel 只带 sm100 变体，sm80/sm89 要从官方索引装
#
# 用法：
#   bash scripts/setup_autodl.sh              # venv 自动选址（见下方 VENV 解析）
#   VENV=$PWD/.venv bash scripts/setup_autodl.sh
#   SKIP_PROJECT=1 bash scripts/setup_autodl.sh   # 只装 sglang，不装本项目依赖
#
# 注意：脚本不碰系统 python，也不改 conda 环境，全部装进独立 venv。
# 它不依赖仓库放在哪个目录，clone 到哪都能跑。
# =====================================================================
set -euo pipefail

SGLANG_VERSION="${SGLANG_VERSION:-0.5.13.post1}"
# 与 SGLANG_VERSION 绑定的 kernel 版本，见 sglang 的 pyproject.toml。
# 换 sglang 版本时这两个要一起换，否则 import 期就会报版本断言失败。
KERNEL_VERSION="${KERNEL_VERSION:-0.4.3+cu129}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
KERNEL_INDEX="${KERNEL_INDEX:-https://docs.sglang.ai/whl/cu129/}"
SKIP_PROJECT="${SKIP_PROJECT:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# venv 放在哪：torch + sglang 装完 6~8 GB，放系统盘容易把盘塞满。
# 优先用显式指定的 $VENV；否则 AutoDL 这种有独立数据盘的机器用大盘，
# 其余平台退回 $HOME。都不合适就自己传 VENV=...
if [ -z "${VENV:-}" ]; then
  if [ -d /root/autodl-tmp ] && [ -w /root/autodl-tmp ]; then
    VENV=/root/autodl-tmp/venv-sglang
  else
    VENV="$HOME/venv-sglang"
  fi
fi

c_ok()   { printf '\033[32m[ OK ]\033[0m %s\n' "$*"; }
c_info() { printf '\033[36m[ .. ]\033[0m %s\n' "$*"; }
c_warn() { printf '\033[33m[WARN]\033[0m %s\n' "$*"; }
c_die()  { printf '\033[31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------
# 0. 前置检查：显卡、驱动
#
# sglang 0.5.13 的依赖里有 flashinfer_python[cu13] 与 nvidia-cutlass-dsl[cu13]，
# 这两组 wheel 是按 CUDA 13 编的，驱动必须 >= 580。驱动不够时装得上、跑不起来，
# 报错会伪装成 "CUDA driver version is insufficient for CUDA runtime version"。
# 与其等到 Engine 启动才炸，不如在这里直接说清楚。
# ---------------------------------------------------------------------
c_info "检查 GPU 与驱动…"
command -v nvidia-smi >/dev/null 2>&1 || c_die "找不到 nvidia-smi：这台机器没有可用 GPU，或没装驱动"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
CUDA_MAX="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -1)"
# 万一 nvidia-smi 输出不是纯数字（老驱动/裁剪镜像），别让下面的 -lt 直接报语法错
case "$DRIVER_MAJOR" in ''|*[!0-9]*) DRIVER_MAJOR=0 ;; esac
c_ok "GPU: ${GPU_NAME} | 驱动: ${DRIVER_MAJOR}.x | CUDA(驱动上限): ${CUDA_MAX:-?}"

if [ "${DRIVER_MAJOR:-0}" -lt 580 ]; then
  c_warn "驱动主版本 ${DRIVER_MAJOR} < 580。sglang ${SGLANG_VERSION} 的 flashinfer/cutlass 是 cu13 构建，"
  c_warn "在旧驱动上大概率起不来。三个选择："
  c_warn "  1) 换一台驱动 >= 580 的机器（租机器时看实例详情里的「驱动/CUDA」列）"
  c_warn "  2) 改用 cu12 时代的 sglang：SGLANG_VERSION=0.5.9 KERNEL_VERSION=0.3.21 SGLANG_CU12=1 bash $0"
  c_warn "  3) 继续（只在你确定驱动够用的情况下）"
  c_warn "继续安装，10 秒后开始；想中止就 Ctrl-C。"
  sleep 10
fi

# ---------------------------------------------------------------------
# 1. uv
# ---------------------------------------------------------------------
c_info "准备 uv…"
if ! command -v uv >/dev/null 2>&1; then
  # 优先 pip（走国内源，最稳）；失败再走官方安装脚本。
  # 新版 Debian/Ubuntu 的 pip 会因 PEP 668 直接拒装，所以带上 --user 兜底。
  if command -v pip3 >/dev/null 2>&1; then
    pip3 install -q -i "$PYPI_INDEX" uv 2>/dev/null \
      || pip3 install -q --user --break-system-packages -i "$PYPI_INDEX" uv 2>/dev/null \
      || true
    export PATH="$HOME/.local/bin:$PATH"
  fi
  if ! command -v uv >/dev/null 2>&1 && command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi
command -v uv >/dev/null 2>&1 || c_die "uv 安装失败：手动装一次（pip install uv），或把 uv 放进 PATH 后重跑"
c_ok "uv $(uv --version | awk '{print $2}')"

# ---------------------------------------------------------------------
# 2. venv（独立环境，不动系统 python / conda）
# ---------------------------------------------------------------------
if [ -x "$VENV/bin/python" ]; then
  c_ok "复用已有环境 $VENV"
else
  c_info "创建 venv：$VENV (python $PYTHON_VERSION)"
  mkdir -p "$(dirname "$VENV")" 2>/dev/null \
    || c_die "建不了 $(dirname "$VENV")（权限或磁盘）。换位置重跑：VENV=\$HOME/venv-sglang bash $0"
  # 装完约 6~8 GB，先看一眼剩余空间，免得装到一半 No space left
  FREE_G="$(df -Pk "$(dirname "$VENV")" 2>/dev/null | awk 'NR==2{printf "%d", $4/1048576}')"
  if [ -n "${FREE_G:-}" ] && [ "$FREE_G" -lt 15 ]; then
    c_warn "$(dirname "$VENV") 只剩 ${FREE_G}G，torch+sglang 装完要 6~8G，"
    c_warn "容器重启后还会重下。换到大盘：VENV=/your/big/disk/venv-sglang bash $0"
    c_warn "继续安装，8 秒后开始；想中止就 Ctrl-C。"
    sleep 8
  fi
  # uv 会优先用系统里已有的同版本解释器；没有才去下载。
  # 国内下 CPython 慢时可以设 UV_PYTHON_INSTALL_MIRROR。
  if ! uv venv --python "$PYTHON_VERSION" "$VENV"; then
    c_warn "拿不到 python $PYTHON_VERSION，退回系统 python3"
    uv venv --python "$(command -v python3)" "$VENV"
  fi
fi
PY="$VENV/bin/python"
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
  || c_die "python 版本过低（sglang 要求 >=3.10）"
c_ok "$($PY -V)"

# ---------------------------------------------------------------------
# 3. sglang 本体
#
# 不装 [all]：那个 extra 会拖 diffusers / trimesh / vsa 一整套视频生成依赖，
# 与本项目无关，白白多几百 MB 和一堆版本冲突。
# 也不加 --no-deps：让 uv 按 sglang 自己的 pin 解析 torch/transformers，
# ABI 才能和预编译 kernel 对上（Kaggle 上就是因为 --no-deps 用了镜像的
# 另一个 torch 版本，才出现 undefined symbol）。
# ---------------------------------------------------------------------
c_info "安装 sglang==${SGLANG_VERSION}（含 torch / transformers，约 4~6 GB，耐心等）…"
uv pip install --python "$PY" --index-url "$PYPI_INDEX" "sglang==${SGLANG_VERSION}"
c_ok "sglang 安装完成"

# ---------------------------------------------------------------------
# 4. 换掉 sglang-kernel（本脚本存在的唯一理由）
#
# PyPI 上的 sglang-kernel 是给 Hopper/Blackwell 编的，wheel 里只有
# sm90/sm100 的 common_ops；在 A100(8.0)/4090(8.9) 上 import 时报：
#   [sgl_kernel] CRITICAL: Could not load any common_ops library!
#   Expected variant: SM89
# 官方把各架构的构建放在 docs.sglang.ai/whl/cu129/。
# ---------------------------------------------------------------------
c_info "替换 sglang-kernel -> ${KERNEL_VERSION} (${KERNEL_INDEX})…"
uv pip install --python "$PY" --no-deps --reinstall \
  --index-url "$KERNEL_INDEX" "sglang-kernel==${KERNEL_VERSION}"

c_info "验证 kernel 里带了哪些架构…"
"$PY" - <<'PY'
import os, sgl_kernel
root = os.path.dirname(sgl_kernel.__file__)
sm_dirs = sorted(d for d in os.listdir(root) if d.startswith("sm"))
print("  kernel 版本 :", getattr(sgl_kernel, "version", "?"))
print("  sm 目录     :", sm_dirs or "(未找到)")
if not sm_dirs:
    raise SystemExit("kernel 里没有 smXX 目录，架构仍不匹配")
PY

# 真正 import 一次：上面的目录检查只能证明文件在，不能证明 .so 能加载
c_info "加载 sgl_kernel（会 dlopen 对应架构的 .so）…"
"$PY" -c "import sgl_kernel; print('  sgl_kernel 加载成功')" \
  || c_die "sgl_kernel 加载失败：多半是 KERNEL_VERSION 与 SGLANG_VERSION 不配套，或驱动/CUDA 版本不匹配"
c_ok "kernel 就绪"

# ---------------------------------------------------------------------
# 5. 本项目依赖
# ---------------------------------------------------------------------
if [ "$SKIP_PROJECT" = "1" ]; then
  c_warn "SKIP_PROJECT=1，跳过项目依赖"
else
  c_info "安装项目依赖…"
  uv pip install --python "$PY" --index-url "$PYPI_INDEX" -r "$REPO_ROOT/requirements.txt"
  # sglang 已经把 transformers 钉在 5.8.x，这里确认没被 requirements 拽回 4.x
  "$PY" - <<'PY'
import transformers
print("  transformers:", transformers.__version__)
major = int(transformers.__version__.split(".")[0])
if major < 5:
    raise SystemExit("transformers 被降级到 4.x，sglang 会 import 失败；检查 requirements 的版本上限")
PY
  c_ok "项目依赖就绪"
fi

# ---------------------------------------------------------------------
# 6. 汇总
# ---------------------------------------------------------------------
cat <<EOF

================= 环境就绪 =================
  venv        : $VENV
  激活        : source $VENV/bin/activate
  sglang      : $("$PY" -c 'import sglang;print(sglang.__version__)' 2>/dev/null || echo '?')
  torch       : $("$PY" -c 'import torch;print(torch.__version__, "| cuda", torch.version.cuda, "| avail", torch.cuda.is_available())')
  transformers: $("$PY" -c 'import transformers;print(transformers.__version__)')

下一步：
  # 1) 下模型（ModelScope，国内直连）。输出里的 "model.name_or_path:" 就是下面要用的路径
  python scripts/download_models_ms.py --only policy

  # 2) 先跑 CPU 档确认链路（不需要 GPU）
  python scripts/ablate_rollout.py smoke

  # 3) 量加速比（$POLICY_PATH 换成上一步打印的路径）
  POLICY_PATH=\$(python scripts/download_models_ms.py --only policy | sed -n 's/^  model.name_or_path: *//p')
  python scripts/ablate_rollout.py bench \\
      --model "\$POLICY_PATH" \\
      --backends hf,sglang --runtime engine \\
      --logprob-sources sglang,recompute \\
      --batch-size 8 --num-generations 8 --max-new-tokens 256 \\
      --mem-fraction 0.45 --repeats 3
============================================
EOF
