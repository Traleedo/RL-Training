from __future__ import annotations

from collections import defaultdict
from typing import Any, TypeVar

from omegaconf import OmegaConf

__all__ = ["register", "build", "get", "list_registered", "categories", "is_registered"]

#: category -> {name -> class}
_REGISTRY: dict[str, dict[str, type]] = defaultdict(dict)

#: 合法的 category 清单。写错 category 是最常见的低级错误，提前拦住。
_VALID_CATEGORIES = (
    "rollout",
    "scorer",
    "processor",
    "advantage",
    "loss",
    "controller",
    "dataset",
    "actor",
    "critic",
    "reference",
    "logger",
    "checkpointer",
)

#: 报错时附带的排查提示。使用者 90% 的「未注册」是因为忘了扇出 import。
_SCAN_HINT = (
    "提示：只有被 import 过的实现才会出现在注册表里。"
    "如果你刚新增了组件，请确认已在 src/components/__init__.py（或其子包的 "
    "__init__.py）里 import 它。"
)

T = TypeVar("T")


def register(category: str, name: str, *, override: bool = False):
    if category not in _VALID_CATEGORIES:
        raise ValueError(
            f"未知的 category {category!r}；合法取值：{list(_VALID_CATEGORIES)}"
        )

    def deco(cls: type) -> type:
        existing = _REGISTRY[category].get(name)
        if existing is not None and not override:
            raise KeyError(
                f"[{category}] 名字 {name!r} 已被 "
                f"{existing.__module__}.{existing.__qualname__} 占用。"
                f"如需覆盖请显式传 override=True。"
            )
        _REGISTRY[category][name] = cls
        cls._registry_key = (category, name)  # 供调试与自省
        return cls

    return deco


def is_registered(category: str, name: str) -> bool:
    return name in _REGISTRY.get(category, {})


def get(category: str, name: str) -> type:
    """按名字取出组件类；不存在则报错并列出所有候选。"""
    table = _REGISTRY.get(category)
    if table is None or name not in table:
        available = sorted(table) if table else []
        raise KeyError(
            f"[{category}] 未注册名为 {name!r} 的组件。\n"
            f"已注册的 {category} 组件：{available}\n{_SCAN_HINT}"
        )
    return table[name]


def list_registered(category: str | None = None) -> list[str] | dict[str, list[str]]:
    """列出已注册的组件名。不传 category 则返回全部 category 的字典。"""
    if category is None:
        return {cat: sorted(names) for cat, names in _REGISTRY.items()}
    return sorted(_REGISTRY.get(category, {}))


def categories() -> list[str]:
    return sorted(_REGISTRY)


def build(category: str, cfg: Any, **overrides: Any) -> Any:
    if isinstance(cfg, str):
        cfg = OmegaConf.create({"type": cfg})
    elif isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    if not OmegaConf.is_config(cfg):
        # 已经构建好的实例（或任意对象）—— 原样传回。
        return cfg

    raw = dict(cfg.items())          # 叶子已解析；嵌套仍是 DictConfig，留给子构建
    type_name = raw.pop("type", None)
    if type_name is None:
        raise KeyError(
            f"[{category}] 配置缺少 'type' 字段。收到的键：{sorted(raw)}"
        )
    if not isinstance(type_name, str):
        raise TypeError(
            f"[{category}] 'type' 必须是字符串，收到 {type(type_name).__name__}。"
        )

    raw.pop("weight", None)          # 归 LossComposition，不进构造函数

    cls = get(category, type_name)
    raw.update(overrides)
    return cls(**raw)
