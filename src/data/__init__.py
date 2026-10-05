"""数据层 —— 离线家族的样本来源。

与 ``components/`` 分开，是因为这里的实现要**读文件**（IO）。
``components/`` 是「纯逻辑、只依赖 core」的那一层，把读盘混进去会让
「任意组件都能被拔掉换成另一个」这条不变量变得含糊 —— 一个组件到底有没有副作用，
看一眼目录名就该知道。

这里的实现同样是**注册的唯一扇出点**：新增一个数据集后端，除了写文件和加
``@register("dataset", ...)``，还必须在这个文件里 import 它。忘了的症状与
``components/`` 完全一样：报错说「未注册 xxx」而文件明明存在。
"""

from __future__ import annotations

import data.json  # noqa: F401  触发 json_sft / json_preference 的注册
import data.jsonl  # noqa: F401  触发 jsonl_sft / jsonl_preference 的注册

__all__ = ["json", "jsonl"]
