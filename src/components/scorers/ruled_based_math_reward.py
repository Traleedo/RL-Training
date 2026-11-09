"""基于规则的可验证数学奖励（rule-based verifiable math reward）。

一次 rollout 被拆成四个可加的分项（权重可在构造时配置）：

==========  ========  =======================================================
分项        默认权重   含义
==========  ========  =======================================================
accuracy     0.4      ``math_verify`` 判定回答与标准答案是否数学等价
reasoning    0.3      推理步骤标记（``Step 1:`` / 编号列表 / ``First,``）的密度
length       0.2      组内相对长度整形：同组更短的正确回答加分、更长的扣分
format       0.1      严格匹配 ``<think>...</think>`` + ``<answer>...</answer>``
==========  ========  =======================================================

奖励的符号约定
--------------
辅助项（reasoning / format）只在**答对**时给正分；length 项对错误回答只可能
给出 ``<= 0`` 的惩罚。于是答错的样本恒有 ``reward <= 0``，答对的样本恒有
``reward > 0``。这不是为了好看，而是为了防止「答错但格式漂亮、关键词堆满」
的回答拿到与正确答案相当的奖励 —— 那会直接教模型刷关键词而不是解题。
构造时传 ``gate_auxiliary_on_correct=False`` 可以退回四路独立加权的旧行为。

Batch 契约
----------
读：

- ``F.RESPONSE_TEXTS``：模型输出，``list[str]``，长度 B。
- ``F.RESPONSE_MASK``：``[B, L]``。用它的行和作为长度（token 数），而不是
  字符数 —— token 数才是训练成本，也不会因为中文/空白比例而失真。
- ``F.SOLUTIONS``：每行的标准答案，``list[str]``，长度 B。必须与
  ``RESPONSE_TEXTS`` **逐行对齐**（即已经按 ``num_generations`` 展开），
  并且只放最终答案本身、不要放整段解题过程 —— 解析器取第一个匹配。
- ``F.GROUP_IDS``（可选）：``[B]``。有它时长度项在**组内**归一化；没有时
  退化为整批归一化。

写：

- ``F.REWARDS``：``torch.Tensor``，``[B]``，float32，设备跟随 ``input_ids``。
  **必须是张量**：下游 ``RewardProcessor`` 直接对它做 ``mean()/std()``，
  advantage 还要与 ``response_mask`` 做逐 token 广播。写 ``list[float]``
  会被 ``Batch.__setitem__`` 放进 ``non_tensors``，然后在 processor 里
  以 ``AttributeError: 'list' object has no attribute 'mean'`` 的形式炸掉。

两个容易踩的坑
--------------
1. **正确性只算一次。** ``parse`` + ``verify`` 是这条链上最贵的操作，而且
   accuracy 与 length 两个分项都需要「答对没有」。原实现各算一遍、还用了
   不同的 extraction config，于是同一个回答可能「accuracy 判错、length 判对」，
   长度奖励和正确性奖励互相打架 —— 这种不一致不会报错。
2. **答案从 ``<answer>`` 标签里取。** ``math_verify`` 的锚点只认识
   ``\\boxed{}`` 与 "answer is" 这类词，**不认识 ``<answer>`` 标签**。直接拿
   整段 response 去 parse，一个严格照格式作答的模型（``<answer>42</answer>``，
   没有 ``\\boxed``）会解析失败、准确率奖励恒为 0。所以这里先抽出
   ``<answer>`` 的内容再解析；没有标签时才退回 ``math_verify`` 的原始锚点
   （兼容 ``\\boxed{}`` 写法）。
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from types import SimpleNamespace

import torch

from core import interfaces as F
from core.base_scorer import Scorer
from core.batch import Batch
from core.registry import register

__all__ = ["RuledBasedMathReward"]

logger = logging.getLogger(__name__)
_MATH_VERIFY: SimpleNamespace | None = None


def _math_verify() -> SimpleNamespace:
    """首次使用时加载解析器，并构造两套 extraction config。

    - ``inner``：内容已从 ``<answer>`` 标签中定位，不再要求锚点；
    - ``anchored``：没有标签时的后备，只认 ``\\boxed{}`` / "answer is"。
    """
    global _MATH_VERIFY
    if _MATH_VERIFY is not None:
        return _MATH_VERIFY

    try:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import (
            ExprExtractionConfig,
            LatexExtractionConfig,
            parse,
            verify,
        )
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise ImportError(
            "RuledBasedMathReward 需要 math-verify 与 latex2sympy2-extended。"
            "安装：pip install 'math-verify>=0.9' 'latex2sympy2-extended>=1.11'"
        ) from exc

    normalization = NormalizationConfig(
        nits=False,
        malformed_operators=False,
        basic_latex=True,
        equations=True,
        boxed="all",  # 只能是 "all" / "last" / "none"；写 True 会静默失效
        units=True,
    )
    inner = [
        LatexExtractionConfig(
            normalization_config=normalization,
            boxed_match_priority=0,
            try_extract_without_anchor=True,
        ),
        ExprExtractionConfig(try_extract_without_anchor=True),
    ]
    anchored = [
        LatexExtractionConfig(
            normalization_config=normalization,
            boxed_match_priority=0,
            try_extract_without_anchor=False,
        ),
        ExprExtractionConfig(try_extract_without_anchor=False),
    ]
    _MATH_VERIFY = SimpleNamespace(
        parse=parse, verify=verify, inner=inner, anchored=anchored
    )
    return _MATH_VERIFY


_ANSWER_BLOCK = re.compile(
    r"<answer>\s*(?P<body>.*?)\s*</answer>", re.DOTALL | re.IGNORECASE
)

#: 用 ``fullmatch`` + ``\A/\Z`` 锚定整段文本。
#:
#: 原实现带了 ``re.MULTILINE``：那会让 ``^`` / ``$`` 匹配**任意行的首尾**，
#: 于是响应中间恰好嵌着一段 ``<think>...\n</think>\n<answer>...</answer>``
#: 也会被判为格式正确 —— 这是安静的误判，不是无害的宽松。
_FORMAT_PATTERN = re.compile(
    r"\A\s*<think>\s*\n.*?\n\s*</think>\s*\n\s*<answer>\s*\n.*?\n\s*</answer>\s*\Z",
    re.DOTALL,
)

#: 推理步骤标记。``^[ \t]*(?:\d+[.)]|[-*])`` 必须带 ``MULTILINE`` 才能在每行
#: 行首匹配；原实现的 ``^\d+\.`` 没有 ``MULTILINE``，只能在整段开头匹配，
#: 等于漏掉了绝大多数编号列表。
_REASONING_PATTERN = re.compile(
    r"(?:^[ \t]*(?:\d+[.)]|[-*])[ \t]+"
    r"|\bStep[ \t]*\d+[ \t]*:"
    r"|\b(?:First|Second|Third|Next|Then|Finally)[ \t]*,)",
    re.IGNORECASE | re.MULTILINE,
)


@register("scorer", "ruled_based_math_reward")
class RuledBasedMathReward(Scorer):
    """数学奖励：正确性 + 推理步骤 + 组内相对长度 + 输出格式。

    参数
    ----
    accuracy_weight / reasoning_weight / length_weight / format_weight:
        四个分项的权重，必须非负且不能全为 0。
    gate_auxiliary_on_correct:
        为 True（默认）时 reasoning / format 只在答对时计入总分。
    reasoning_steps_for_full_credit:
        多少个推理标记算「推理项满分」，默认 3。
    """

    requires = frozenset({F.RESPONSE_TEXTS, F.RESPONSE_MASK, F.SOLUTIONS})

    def __init__(
        self,
        accuracy_weight: float = 0.4,
        reasoning_weight: float = 0.3,
        length_weight: float = 0.2,
        format_weight: float = 0.1,
        *,
        gate_auxiliary_on_correct: bool = True,
        reasoning_steps_for_full_credit: int = 3,
    ) -> None:
        super().__init__()
        weights = {
            "accuracy_weight": float(accuracy_weight),
            "reasoning_weight": float(reasoning_weight),
            "length_weight": float(length_weight),
            "format_weight": float(format_weight),
        }
        for name, value in weights.items():
            if value < 0:
                raise ValueError(f"{name} 必须 >= 0，收到 {value}")
            setattr(self, name, value)
        if sum(weights.values()) == 0:
            raise ValueError("四个权重不能全为 0，否则奖励恒为 0、训练毫无信号")
        if int(reasoning_steps_for_full_credit) < 1:
            raise ValueError(
                "reasoning_steps_for_full_credit 必须 >= 1，"
                f"收到 {reasoning_steps_for_full_credit}"
            )
        self.gate_auxiliary_on_correct = bool(gate_auxiliary_on_correct)
        self.reasoning_steps_for_full_credit = int(reasoning_steps_for_full_credit)
        self._last_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 纯文本分项
    # ------------------------------------------------------------------
    def format_reward(self, response_texts: list[str]) -> list[float]:
        """严格匹配 ``<think>...</think>`` + ``<answer>...</answer>`` 的模板。"""
        return [
            1.0 if _FORMAT_PATTERN.fullmatch(text) is not None else 0.0
            for text in response_texts
        ]

    def reasoning_steps_reward(self, response_texts: list[str]) -> list[float]:
        """推理步骤标记的密度，3 个标记即满分。

        这个分项本质上是可刷的（堆 "First, Next, Finally," 就能拿满），
        所以权重不该给高，并且默认只在答对时计入。
        """
        full_credit = self.reasoning_steps_for_full_credit
        return [
            min(1.0, len(_REASONING_PATTERN.findall(text)) / full_credit)
            for text in response_texts
        ]

    # ------------------------------------------------------------------
    # 正确性：全批次只判定一次
    # ------------------------------------------------------------------
    def _correctness(
        self, response_texts: list[str], solutions: list[str]
    ) -> tuple[list[bool], dict[str, int]]:
        gold_cache: dict[str, list] = {}
        correct: list[bool] = []
        failed_gold = failed_answer = failed_verify = 0

        for response, solution in zip(response_texts, solutions):
            # GRPO 同一个 prompt 有 n 条采样，标准答案完全相同 —— 缓存掉重复解析
            if solution not in gold_cache:
                gold_cache[solution] = self._parse_gold(solution)
            gold = gold_cache[solution]
            if not gold:
                failed_gold += 1
                correct.append(False)
                continue

            answer = self._parse_answer(response)
            if not answer:
                failed_answer += 1
                correct.append(False)
                continue

            try:
                # math_verify.verify(gold, target)：顺序是「标准答案在前」。
                # 原实现两处顺序相反，同一个判定在 accuracy / length 里可能不同。
                correct.append(bool(_math_verify().verify(gold, answer)))
            except Exception:  # noqa: BLE001 — sympy/timeout 都可能，统一按答错
                failed_verify += 1
                logger.debug(
                    "verify 抛错，按答错处理：gold=%r answer=%r",
                    gold,
                    answer,
                    exc_info=True,
                )
                correct.append(False)

        if failed_verify:
            logger.warning(
                "%d/%d 条回答在 verify 时抛错，已按答错处理。",
                failed_verify,
                len(response_texts),
            )
        return correct, {"gold": failed_gold, "answer": failed_answer}

    @staticmethod
    def _parse_gold(solution: str) -> list:
        """标准答案用 ``math_verify`` 的默认（宽松）配置。

        数据侧通常只存答案本身（``42`` / ``\\frac{1}{2}``），没有 ``\\boxed``
        也没有锚点，所以这里不能要求锚点。默认配置同时包含 latex 与普通表达式。
        """
        try:
            return _math_verify().parse(str(solution), extraction_mode="first_match")
        except Exception:  # noqa: BLE001
            logger.debug("标准答案解析失败：%r", str(solution)[:200], exc_info=True)
            return []

    @staticmethod
    def _parse_answer(response: str) -> list:
        """优先取最后一个 ``<answer>`` 标签里的内容，没有再退回锚点解析。"""
        blocks = list(_ANSWER_BLOCK.finditer(response))
        if blocks:
            candidate = blocks[-1].group("body")
            extraction = _math_verify().inner
        else:
            candidate = response
            extraction = _math_verify().anchored
        try:
            return _math_verify().parse(
                candidate,
                extraction_config=extraction,
                extraction_mode="first_match",
            )
        except Exception:  # noqa: BLE001
            logger.debug("回答解析失败：%r", response[-200:], exc_info=True)
            return []

    # ------------------------------------------------------------------
    # 长度整形
    # ------------------------------------------------------------------
    @staticmethod
    def _length_terms(
        lengths: list[int],
        correct: list[bool],
        group_ids: list[int] | None,
    ) -> list[float]:
        """组内相对长度项，落在 ``[-0.5, 0.5]``。

        只在**同一组**（同一个 prompt 的 n 条采样）内比较长度：不同题目的
        标准回答长短本来就不同，跨组取 min/max 会让「这道题答案就该长」的样本
        被无端惩罚。组大小为 1 时无相对信息，该项为 0。

        答对：最短 +0.5，最长 -0.5；答错：只保留 ``<= 0`` 的惩罚。
        """
        n = len(lengths)
        if n == 0:
            return []

        groups: dict[int, list[int]] = defaultdict(list)
        if group_ids is not None and len(group_ids) == n:
            for index, group_id in enumerate(group_ids):
                groups[int(group_id)].append(index)
        else:
            groups[0] = list(range(n))

        terms = [0.0] * n
        for members in groups.values():
            low = min(lengths[i] for i in members)
            high = max(lengths[i] for i in members)
            if high == low:
                continue
            span = high - low
            for i in members:
                lambda_val = 0.5 - (lengths[i] - low) / span
                terms[i] = lambda_val if correct[i] else min(0.0, lambda_val)
        return terms

    # ------------------------------------------------------------------
    # 组件入口
    # ------------------------------------------------------------------
    def score(self, batch: Batch) -> Batch:
        batch.require(
            F.RESPONSE_TEXTS, F.RESPONSE_MASK, F.SOLUTIONS,
            who=type(self).__name__,
        )
        response_texts = [str(text) for text in batch[F.RESPONSE_TEXTS]]
        solutions = [str(solution) for solution in batch[F.SOLUTIONS]]
        if len(solutions) != len(response_texts):
            # zip 会静默截断，然后奖励行数与 batch 对不上 —— 越早炸越好
            raise ValueError(
                f"{F.SOLUTIONS} 的长度 {len(solutions)} 与 "
                f"{F.RESPONSE_TEXTS} 的长度 {len(response_texts)} 不一致。"
                f"标准答案必须与 rollout 展开后的每一行逐行对齐。"
            )

        device = (
            batch[F.INPUT_IDS].device
            if F.INPUT_IDS in batch.tensors
            else torch.device("cpu")
        )

        if not response_texts:
            batch[F.REWARDS] = torch.zeros(0, dtype=torch.float32, device=device)
            self._last_metrics = {}
            return batch

        correct, parse_failures = self._correctness(response_texts, solutions)
        lengths = [int(length) for length in batch[F.RESPONSE_MASK].sum(dim=1).tolist()]
        if len(lengths) != len(response_texts):
            raise ValueError(
                f"{F.RESPONSE_MASK} 的行数 {len(lengths)} 与 "
                f"{F.RESPONSE_TEXTS} 的长度 {len(response_texts)} 不一致"
            )
        group_ids = (
            [int(gid) for gid in batch[F.GROUP_IDS].detach().to("cpu").tolist()]
            if F.GROUP_IDS in batch.tensors
            else None
        )

        reasoning = self.reasoning_steps_reward(response_texts)
        formats = self.format_reward(response_texts)
        length_terms = self._length_terms(lengths, correct, group_ids)

        total: list[float] = []
        for is_correct, reason, length_term, fmt in zip(
            correct, reasoning, length_terms, formats
        ):
            reward = (
                self.accuracy_weight * float(is_correct)
                + self.length_weight * length_term
            )
            if is_correct or not self.gate_auxiliary_on_correct:
                reward += (
                    self.reasoning_weight * reason
                    + self.format_weight * fmt
                )
            total.append(reward)

        batch[F.REWARDS] = torch.tensor(
            total, dtype=torch.float32, device=device
        )

        n = len(response_texts)
        self._last_metrics = {
            "reward/accuracy": sum(correct) / n,
            "reward/format": sum(formats) / n,
            "reward/reasoning": sum(reasoning) / n,
            "reward/length_term": sum(length_terms) / n,
            "reward/unparsed_gold": float(parse_failures["gold"]),
            "reward/unparsed_answer": float(parse_failures["answer"]),
        }
        if parse_failures["gold"]:
            logger.warning(
                "%d/%d 条标准答案解析失败（准确率项按 0 处理）。"
                "检查 %s 里是否混入了整段解题过程或非答案文本。",
                parse_failures["gold"], n, F.SOLUTIONS,
            )
        return batch

    def metrics(self) -> dict[str, float]:
        return dict(self._last_metrics)
