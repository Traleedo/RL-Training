"""控制器 —— 每个训练步调整**别处**系数的有状态组件。

为什么需要这个 category
----------------------
其它 category 都是「Batch 的纯函数」：读字段、写字段，算法差异体现在写什么。
但有一类算法机制无法这样表达 —— **它要根据本步的观测反过来调整另一个组件的参数**。
最典型的是自适应 KL：

.. math::

    \\beta \\leftarrow \\mathrm{clip}\\bigl(\\beta + \\eta \\cdot (\\mathrm{KL}_{\\text{obs}} - \\mathrm{KL}_{\\text{target}}),\\ \\beta_{\\min},\\ \\beta_{\\max}\\bigr)

这里 ``beta`` 是**跨 step 存活**的状态，``KL_obs`` 来自本步的日志，而它的作用点是
**另一个组件的 Python 属性**。塞进任何现有 category 都会歪：它不是 loss（不改梯度）、
不是 processor（不碰 Batch）、不是 advantage（不碰奖励）。

所以给它一个 category：**有状态的系数调度器**。

``requires`` / ``provides`` 刻意为空
----------------------------------
这不是偷懒。控制器唯一做的事情是改另一个组件的属性，它既不读也不写 Batch。
如果为了「看起来自洽」而声明 ``requires = {REF_LOGPROBS}``，它会**错误地成为
拉起 Reference 模型的原因** —— 而真正需要 Reference 的是它绑定的那个 KL 项。
需求归属必须留在损失项上，否则 ``plan_assembly`` 的 ``model_reasons`` 会指向
一个根本没碰过那个字段的组件，排查时会被带偏。

绑定用 ``LossTerm.name()``，不是 YAML 的 ``type``
------------------------------------------------
指标键 ``loss/{name}/mean_kl`` 就是用 ``name()`` 拼出来的（见 ``core.base_loss``
里 ``LossComposition.__call__`` 的 ``prefix``）。用同一个标识符做**绑定**和
**读数**，能消掉一整类「绑对了项、却读错了键」的 bug —— 那种错会让控制器读到
``KeyError`` 或（更糟）一个恒为 0 的默认值，然后安静地把系数调到界外。

所以 YAML 里写 ``term: kl_k3``（注册名恰好也是 ``kl_k3``，但语义上是 ``name()``）。
不写 ``term`` 时会自动探测名字以 ``kl`` 开头的项，**候选不唯一就报错**，不猜。

β 的生效时机（容易误解，必须知道）
--------------------------------
``on_train_step_end`` 在一个 train_step 的**最后**被调用，而 ``LossComposition``
是在 epoch × mini-batch 循环**内部**逐块调用的。所以第 k 步写进去的系数，
要到**第 k+1 步**才影响损失 —— 第 k 步后续的 mini-batch 不会受影响。

这是刻意的，不是缺陷：要让新系数在当前步的剩余 mini-batch 生效，就得把控制器
塞进 ``LossComposition`` 的循环里，那会让「一项损失算多少次」和「系数改多少次」
纠缠在一起，且每个 mini-batch 读到不同的 β，梯度不再是无偏的同一个目标。
"""

from __future__ import annotations

from typing import Any

from core.base_loss import LossTerm
from core.component import Component

__all__ = ["Controller", "resolve_target", "check_target"]

#: 不写 ``term:`` 时的自动探测前缀。候选不唯一即报错，不猜。
_AUTO_PREFIX = "kl"


def resolve_target(
    controller: "Controller", terms: dict[str, LossTerm]
) -> tuple[str | None, list[str]]:
    """解析控制器要调整的损失项名，返回 ``(项名 或 None, 问题列表)``。

    纯函数 —— 不改控制器状态，也不改 ``terms``。``plan_assembly`` 用它一次收齐
    全部控制器的全部问题（而不是修一个报一个）；``Controller.bind`` 用它做运行期
    兜底（支持不经 ``plan_assembly`` 手工组装 trainer 的用法）。
    """
    spec = controller.term_spec
    if spec is not None:
        if spec in terms:
            return spec, []
        return None, [
            f"目标损失项 {spec!r} 不存在。当前存活的损失项名：{sorted(terms)}。\n"
            f"    注意 term: 要填 LossTerm.name()（日志里 loss/<名字>/... 的那个名字），"
            f"不是 YAML 里的 type。\n"
            f"    另一个常见原因：该项的 weight 被设成了 0.0 —— 零权重的损失项在装配前"
            f"就被剔除了，控制器无法绑定一个不参与训练的项。"
        ]

    candidates = sorted(n for n in terms if n.startswith(_AUTO_PREFIX))
    if len(candidates) == 1:
        return candidates[0], []
    if not candidates:
        return None, [
            f"没有写 term:，也自动探测不到候选（名字以 {_AUTO_PREFIX!r} 开头的项）。"
            f"当前存活的损失项名：{sorted(terms)}。请显式写 term:。"
        ]
    return None, [
        f"没有写 term:，而自动探测到多个候选 {candidates}，无法确定要调整哪一个。"
        f"请显式写 term:。"
    ]


def check_target(name: str, target: LossTerm) -> list[str]:
    """检查目标项是否**可被调整**。返回问题列表。"""
    if not hasattr(target, "coef"):
        return [
            f"目标损失项 {name!r} 没有 coef 属性，控制器无处写入。"
            f"控制器改的是损失项自身的 coef，不是 YAML 里的 weight。"
        ]
    coef = target.coef
    # bool 是 int 的子类，先排掉，否则 coef=True 会被当成 1.0 放过去
    if isinstance(coef, bool) or not isinstance(coef, (int, float)):
        return [f"目标损失项 {name!r} 的 coef 不是数值（{coef!r}），控制器无法调整。"]

    if float(coef) <= 0.0:
        # 这是本项目最隐蔽的一个启动期陷阱，报错信息必须把因果链写全。
        return [
            f"目标损失项 {name!r} 的 coef = {float(coef)}，控制器必须从一个**正**系数起步。\n"
            f"    因果链：coef == 0 时该损失项会在 __init__ 里 release(ref_logprobs)"
            f"（省掉一份完整模型的显存）-> plan_assembly 判定没有任何组件需要 "
            f"ref_logprobs -> **Reference 模型根本不会被构建** -> 控制器把系数调上去"
            f"之后再去读 ref_logprobs 就是 KeyError。\n"
            f"    修法：把该项的 coef 设成一个正数。控制器会在第一次观测之后接管它，"
            f"所以初值只需要 > 0，不必是最终值。"
        ]
    return []


class Controller(Component):
    """有状态的系数调度器。子类实现 ``on_train_step_end``。"""

    category = "controller"

    #: 刻意为空，理由见模块 docstring：控制器不碰 Batch，声明字段会让它错误地
    #: 成为某个模型被构建的原因。
    requires = frozenset()
    provides = frozenset()

    def __init__(self, term: str | None = None) -> None:
        super().__init__()
        self._term_spec: str | None = None if term is None else str(term)
        self._target: LossTerm | None = None

    # ------------------------------------------------------------------
    # 绑定
    # ------------------------------------------------------------------
    @property
    def term_spec(self) -> str | None:
        """YAML 里显式写的目标项名；``None`` 表示让框架自动探测。"""
        return self._term_spec

    @property
    def target(self) -> LossTerm:
        """已被绑定的目标损失项。"""
        if self._target is None:
            raise RuntimeError(
                f"{type(self).__name__} 还没有绑定损失项。"
                f"绑定由 trainer 在装配阶段完成（_bind_controllers）。"
                f"如果你是手工组装 trainer，记得自己调一次 bind()。"
            )
        return self._target

    @property
    def target_name(self) -> str:
        return self.target.name()

    def bind(self, terms: dict[str, LossTerm]) -> None:
        """解析并锁定要调整的损失项。

        ``terms`` 是 ``{name(): LossTerm}``，**只含权重非零的存活项**
        （零权重项在装配前已被剔除）。

        这里重跑一遍 ``plan_assembly`` 已经做过的检查：不是冗余，而是服务于
        「手工组装 trainer、不经过 plan_assembly」的用法。两条路径的检查逻辑
        共用同一组纯函数，所以不会出现两处判断不一致。
        """
        name, problems = resolve_target(self, terms)
        if name is not None:
            problems = problems + check_target(name, terms[name])
        if problems:
            raise ValueError(
                f"控制器 {type(self).__name__} 绑定失败：\n  - " + "\n  - ".join(problems)
            )
        self._target = terms[name]

    # ------------------------------------------------------------------
    # 系数读写：目标项是唯一真相来源
    # ------------------------------------------------------------------
    @property
    def current_coef(self) -> float:
        """目标项当前的系数。"""
        return float(self.target.coef)

    @current_coef.setter
    def current_coef(self, value: float) -> None:
        self.target.coef = float(value)

    # ------------------------------------------------------------------
    # 钩子
    # ------------------------------------------------------------------
    def on_train_step_end(self, metrics: dict[str, float]) -> None:
        """一个 train_step 结束后被调用。子类在这里根据 ``metrics`` 调整系数。

        ``metrics`` 里已经包含本步的 ``loss/{term}/mean_kl`` 等全部指标，可以直接
        **就地修改**它来把控制器自己的状态上报到同一行日志。

        注意 β 的生效时机（见模块 docstring）：这里写进去的系数在第 k+1 步才生效。
        """
        raise NotImplementedError(f"{type(self).__name__} 必须实现 on_train_step_end")

    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """把可调系数存进 checkpoint。

        只存系数和目标项名 —— 系数的轨迹就是自适应控制器全部的产物。resume 后
        弹回 YAML 初值会让惩罚不连续、与不中断的训练不等价，那正是本仓库拒绝
        接受的静默偏差（对照 ``cfg_hash`` 的存在理由）。
        """
        if self._target is None:
            return {}
        return {"term": self.target_name, "coef": self.current_coef}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        saved_term = state.get("term")
        if saved_term is not None and saved_term != self.target_name:
            # cfg_hash 已经覆盖 algorithm 块（含 controllers），所以正常情况下
            # 走不到这里。留着是因为「系数被写到一个语义不同的项上」不会有任何
            # 报错，只会让训练悄悄偏离 —— 值得多一道便宜的防线。
            raise ValueError(
                f"checkpoint 里 {type(self).__name__} 绑定的是 {saved_term!r}，"
                f"当前配置绑定的是 {self.target_name!r}。拒绝把系数套到一个语义不同的项上。"
            )
        self.current_coef = float(state["coef"])

    # ------------------------------------------------------------------
    def metrics(self) -> dict[str, float]:
        """控制器**不用**这个方法上报。

        它不会被 trainer 轮询（见 ``RLTrainer.train_step`` 的 G 相位）—— 那一步在
        ``on_train_step_end`` 之前，轮询到的会是**上一步**的旧系数。上报请在
        ``on_train_step_end`` 里就地写 ``metrics``。
        """
        return {}

    def describe(self) -> str:
        if self._target is None:
            return f"{super().describe()}  term=(未绑定: {self._term_spec!r})"
        return f"{super().describe()}  term={self.target_name}  coef={self.current_coef}"
