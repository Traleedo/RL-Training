"""KL 惩罚项 —— 把策略拴在参考模型附近。三个估计量，一个基类。

.. math::

    L = \\beta \\cdot \\mathbb{E}_t\\left[\\widehat{\\mathrm{KL}}(\\pi_\\theta \\| \\pi_{\\text{ref}})\\right]

三种估计量（Schulman, *Approximating KL Divergence*, 2020）
--------------------------------------------------------
记 ``log_ratio = log(π_θ / π_ref) = logπ_θ − logπ_ref``（下称 ``ℓ``）。

============  ====================  ==========  ==========================================
注册名        公式                  非负？      什么时候用
============  ====================  ==========  ==========================================
``kl_k1``     ``E[ℓ]``              否          要无偏、且不在乎曲线可读性时
``kl_k2``     ``E[½ ℓ²]``           是          方差最低，但**有偏**（见下节）
``kl_k3``     ``E[r − 1 − log r]``  是          **本框架的默认**，也是 PPO / GRPO 用的
============  ====================  ==========  ===========================================

其中 ``ℓ = log(π_θ/π_ref)``，而 k3 里的 ``r = π_ref/π_θ = exp(−ℓ)``。

谁无偏、谁有偏
-------------
**只有 k1 与 k3 以 KL 为期望，k2 不是。** ``E[½ℓ²] = ½(KL² + Var(ℓ))``，
比 KL 高出一个 ``½KL² + ½Var`` —— 在典型的 ``KL ≈ 0.1``、``Var(ℓ)`` 也是
零点几的量级下，这个偏置有 5% ~ 10%，**大大超过任何采样噪声**。
所以别把 k2 当成「k3 的低方差版本」：它换来的低方差是用一个不可忽略的
偏置买来的。``E[½ℓ²]`` 与 ``E[ℓ]`` 是两个不同的矩，跟 ½ 有没有写无关。
（``tests/test_kl_estimators.py`` 用 ``ℓ`` 分布有闭式的正态案例把这三者的
期望分别钉死了。）

**k3 的 r 是哪个方向，不是可以随便选的。** k3 的无偏性来自
``E_{π_θ}[r] = ∫π_ref = 1``；写成 ``r = π_θ/π_ref`` 之后 ``E[r]`` 变成
``1 + χ²``（卡方散度），减完就不再是 KL。两个方向都恒非负、都在 ``ℓ = 0``
处取 0，从指标上完全看不出区别 —— k1/k2 对 ℓ 的符号反而是对称的（k1 线性、
k2 平方），所以这个坑只在 k3 上存在。

选哪个
------
需要无偏就 k1 或 k3；k1 方差大、且可正可负（当监控量不好看），所以 **k3**
是默认。k2 只在你明确想要「惩罚大偏离、对小幅偏离不敏感」这一形状时用 ——
那时你要的是它这个形状，而不是「KL 的估计」。

**k2 里那个 ½ 不能省。** 少了它就是 ``E[ℓ²]``，期望正好是 KL 的两倍 ——
一个纯粹的常数倍偏差。它不会让训练崩，只会让「KL 应该是多少」这个判断一直
差一倍，而监控曲线上完全看不出来。

为什么默认 k3 而不是 k1
---------------------
``E[ℓ]`` 的样本均值可正可负（期望为 0），在监控曲线上分不清「KL 很小」和
「符号写反了」；k3 恒非负。所以 k3 既当损失也当监控量，而 k1 只适合做诊断 ——
用它的时候请同时看 ``mean_log_ratio``，那个量才能告诉你偏离的方向。

KL 放在 loss 侧，不放在 reward 侧
---------------------------------
另一种常见写法是把 KL 逐 token 从奖励里扣掉（``r_t -= β·kl_t``）。**本框架选 loss 侧**，
理由不只是偏好：reward 侧要求 KL 在**相位 B 之前**算出来，但 Reference 的前向是
相位 C（``MODEL_FOR_FIELD[REF_LOGPROBS] = reference``，在打分之后才跑）。
选 reward 侧就得改相位划分，而相位划分决定梯度归属 —— 那等于改整个框架。

附带的好处是 critic 完全看不到 KL，不会去拟合一个含 KL 的目标。

条件依赖
--------
``coef == 0`` 时调用 ``release()`` 把自己对 ``ref_logprobs`` 的依赖退掉，
于是 **Reference 模型根本不会被加载**。这是 ``needed`` 做成实例属性、而非类属性的
直接收益 —— 类属性表达不了「这取决于构造时的配置」。

⚠️ 这条与 ``controller/adaptive_kl`` 有一处冲突：自适应控制器要在运行期把 β
调上去，所以它的目标项**不能**以 ``coef: 0`` 起步 —— 那时 Reference 根本没被
构建，控制器写回 β 之后第一次读 ``ref_logprobs`` 就会 ``KeyError``。
``engine.assembly`` 会在启动时拦住这个配置。

关于 ``coef`` 与 YAML 里的 ``weight``
------------------------------------
两者会相乘。建议 ``weight`` 固定为 1.0，把 β 只写在 ``coef`` 里 ——
这样 ``loss/kl_k3`` 这个指标的含义是唯一的（β × 平均 KL），
而不是两个权重混在一个数里。自适应控制器也只改 ``coef``，不改 ``weight``。
"""

from __future__ import annotations

import torch

from core import interfaces as F
from core.base_loss import LossTerm
from core.batch import Batch
from core.registry import register
from core.tensor_ops import masked_mean

__all__ = ["KLEstimator", "KLK1Loss", "KLK2Loss", "KLK3Loss"]


class KLEstimator(LossTerm):
    """KL 估计量的公共骨架：依赖声明、β 的条件释放、逐 token 公式的挂载点。"""

    requires = frozenset({F.CURR_LOGPROBS, F.REF_LOGPROBS, F.RESPONSE_MASK})

    grad_fields = frozenset({F.CURR_LOGPROBS})

    #: 子类覆写：估计量的短名（k1 / k2 / k3），只用于 ``describe()``。
    estimator = "?"

    def __init__(self, coef: float = 0.01) -> None:
        super().__init__()
        self.coef = float(coef)
        if self.coef == 0.0:
            # 系数为 0 时这一项恒等于 0。不退掉依赖的话，Reference 会被
            # 白白加载（多一份完整模型的显存），而它对训练毫无贡献。
            self.release(F.REF_LOGPROBS)

    def weight_hint(self) -> float:
        return 1.0

    def describe(self) -> str:
        return f"{super().describe()}  估计量={self.estimator} coef={self.coef}"

    # ------------------------------------------------------------------
    def estimate(self, log_ratio: torch.Tensor) -> torch.Tensor:
        """逐 token 的估计值。``log_ratio = log(π_θ / π_ref)``，形状同 mask。

        子类只实现这一个函数 —— 方向（哪边是分子）已经由基类统一钉死，
        三个估计量不可能出现「某个的符号写反了」这种不一致。
        """
        raise NotImplementedError

    def compute(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        if self.coef == 0.0:
            # ``coef`` 为 0 时 ``__init__`` 已经 release 掉 ``ref_logprobs``，
            # 所以这里**读不到它** —— 不是「不想读」，而是它可能根本没被算出来
            # （``plan_assembly`` 据此判定无人需要参考模型，于是不构建
            # Reference）。曾经这里无条件 ``require`` 它，结果是把 ``coef: 0.0``
            # 写成配置的人会在运行期撞上 KeyError —— 一个「配置合法、装配计划
            # 也自洽、跑到第三个 minibatch 才炸」的失败。
            #
            # 返回一个与计算图无关的 0：本项对总损失的贡献恒等于 0，所以不接进
            # 图里是等价的，还省一次前向。device/dtype 从 mask 取，保证与总损失
            # 相加时不触发类型提升。
            #
            # 指标的第二个参数是**空字典**，这也是刻意的。报
            # ``mean_kl: 0.0`` 会是一个**捏造的测量值**：真实 KL 并不为 0，
            # 我们只是选择不去测量它。而 KL 曲线正是监控策略漂移的主要信号，
            # 一条恒为 0 的假曲线比没有曲线更糟 —— 它看起来像「KL 完美收敛」。
            #
            # 注意这与 ``LossComposition`` 报出的 ``loss/kl_k3 = 0.0`` 不矛盾：
            # 那说的是「本项对总损失的贡献是 0」（真的），
            # 而 ``loss/kl_k3/mean_kl`` 会声称「测到的 KL 是 0」（假的）。
            # 前者报、后者不报，正是因为这两句话的真假不同。
            #
            # 不会有组件去读这个不存在的指标：绑定控制器的目标项不允许
            # ``coef <= 0``（plan_assembly 的绑定校验拦住），而 adaptive_kl
            # 读的正是 ``loss/<term>/mean_kl``。
            mask = batch[F.RESPONSE_MASK]
            return torch.zeros((), device=mask.device, dtype=torch.float32), {}

        batch.require(
            F.CURR_LOGPROBS, F.REF_LOGPROBS, F.RESPONSE_MASK, who=type(self).__name__
        )
        mask = batch[F.RESPONSE_MASK]

        # 方向：log_ratio = logπ_θ − logπ_ref，三个 estimate() 共享它。
        # 各子类自己负责把方向用对（k3 那个是唯一不对称的，理由见 KLK3Loss）。
        delta = batch[F.REF_LOGPROBS] - batch[F.CURR_LOGPROBS]
        log_ratio = -delta

        per_token = self.estimate(log_ratio)

        with torch.no_grad():
            metrics = {
                "mean_kl": float(masked_mean(per_token, mask)),
                # mean 会被长尾平均掉，KL 爆炸往往先在这里露头。
                # 注意 k1 的这个量可以是负的（见模块 docstring）。
                "max_kl": float((per_token * mask).max()),
                # 逐 token 的对数比均值。它应当接近 0 附近小幅波动；
                # 单调走同一个方向意味着策略在系统性地偏离参考模型。
                # 想看方向就看它，别指望 mean_kl（k1/k2 都不带方向信息）。
                "mean_log_ratio": float(masked_mean(log_ratio, mask)),
            }
        return self.coef * masked_mean(per_token, mask), metrics


@register("loss", "kl_k1")
class KLK1Loss(KLEstimator):
    """``E[logπ_θ − logπ_ref]`` —— 无偏，但样本均值可正可负。"""

    estimator = "k1"

    def name(self) -> str:
        return "kl_k1"

    def estimate(self, log_ratio: torch.Tensor) -> torch.Tensor:
        return log_ratio


@register("loss", "kl_k2")
class KLK2Loss(KLEstimator):
    """``E[½ ℓ²]``，``ℓ = log(π_θ/π_ref)`` —— 方差最低，但期望不是 KL。"""

    estimator = "k2"

    def name(self) -> str:
        return "kl_k2"

    def estimate(self, log_ratio: torch.Tensor) -> torch.Tensor:
        # ½ 不能省：少了它就是 KL 的两倍（见模块 docstring）。
        return 0.5 * log_ratio * log_ratio


@register("loss", "kl_k3")
class KLK3Loss(KLEstimator):
    """``E[r − 1 − log r]``，其中 ``r = π_ref/π_θ`` —— 恒非负，默认。"""

    estimator = "k3"

    def name(self) -> str:
        return "kl_k3"

    def estimate(self, log_ratio: torch.Tensor) -> torch.Tensor:
        # 注意 r 的**方向**：这里是 r = π_ref/π_θ = exp(−log_ratio)，不是 exp(log_ratio)。
        #
        # 为什么必须是这个方向：k3 的「无偏」说的是
        #     E_{x∼π_θ}[ r − 1 − log r ] = E[r] − 1 − KL(π_θ‖π_ref) = 1 − 1 + KL = KL
        # 而 E_{x∼π_θ}[π_ref/π_θ] = ∫π_ref = 1。反过来（r = π_θ/π_ref）得到的是
        # E[π_θ/π_ref] = 1 + χ²（卡方散度），减完之后**不是** KL。
        #
        # 两个方向都恒非负、都在 ℓ = 0 处取 0，所以从指标上完全看不出来。
        # tests/test_kl_estimators.py 用真 KL 已知的分布把它钉住了。
        return torch.exp(-log_ratio) - 1.0 + log_ratio
