from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

__all__ = [
    "load_config",
    "find_config",
    "iter_algorithm_configs",
    "CONFIG_SUBDIRS",
    "ALGORITHM_SUBDIRS",
]

#: 在 configs/ 下按顺序搜索**任意**配置名的子目录。空串 = 根目录。
CONFIG_SUBDIRS = ("rl", "model", "")

#: 可运行的**算法**配置所在的子目录（不含 model/ —— 那是基座，不是算法）。
ALGORITHM_SUBDIRS = ("rl", "")


def load_config(path: str | Path, _stack: tuple[Path, ...] = ()) -> DictConfig:
    resolved = Path(path).resolve()
    if resolved in _stack:
        chain = " -> ".join(str(p.name) for p in (*_stack, resolved))
        raise ValueError(f"配置继承出现环：{chain}")

    if not resolved.exists():
        raise FileNotFoundError(f"配置文件不存在：{resolved}")

    own = OmegaConf.load(resolved)
    if not isinstance(own, DictConfig):
        raise TypeError(f"{resolved.name} 的顶层应是映射（键值对），收到 {type(own).__name__}")

    defaults = own.pop("defaults", []) or []
    if isinstance(defaults, str):
        defaults = [defaults]

    merged = OmegaConf.create({})
    for name in defaults:
        base_path = resolved.parent / f"{name}.yaml"
        merged = OmegaConf.merge(
            merged, load_config(base_path, (*_stack, resolved))
        )

    return OmegaConf.merge(merged, own)


#: configs/ 下不是「算法配置」的名字。base 是被继承的公共块，model/ 是基座。
_NOT_AN_ALGORITHM = {"base"}


def find_config(name: str, root: str | Path) -> Path:
    """把一个配置名解析成路径。

    接受三种写法：``ppo``、``rl/ppo``、以及任何存在的显式路径
    （``configs/rl/ppo.yaml`` 或绝对路径）。搜索顺序见 ``CONFIG_SUBDIRS``。

    放在 engine 里而不是各个脚本里，是因为这套目录约定被 plan / verify_e2e /
    train / 若干个测试同时需要 —— 各拼一份路径的话，目录一挪就会有脚本静默地
    找不到配置。
    """
    root = Path(root)

    candidate = Path(name)
    if candidate.exists():
        return candidate
    if candidate.suffix != ".yaml":
        with_suffix = candidate.with_suffix(".yaml")
        if with_suffix.exists():
            return with_suffix

    # 名字里已经带了子目录（``rl/ppo``）时直接拼，否则按 CONFIG_SUBDIRS 搜。
    if candidate.parent != Path("."):
        path = root / f"{candidate}.yaml"
        if path.exists():
            return path
    else:
        for subdir in CONFIG_SUBDIRS:
            path = root / subdir / f"{candidate}.yaml"
            if path.exists():
                return path

    available = sorted(p.stem for p in iter_algorithm_configs(root))
    raise FileNotFoundError(
        f"找不到配置 {name!r}（在 {root} 下按 {list(CONFIG_SUBDIRS)} 搜索）。\n"
        f"可用的算法配置：{available}"
    )


def iter_algorithm_configs(root: str | Path) -> list[Path]:
    """列出所有**可运行的**算法配置：``configs/rl/*.yaml`` 加 ``configs/*.yaml``。

    排除 ``base.yaml``（公共块，不是算法）与 ``model/``（基座，不是算法）。
    名字按路径排序，保证遍历顺序稳定。
    """
    root = Path(root)
    found: list[Path] = []
    for subdir in ALGORITHM_SUBDIRS:
        directory = root / subdir
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.yaml")):
            if path.stem in _NOT_AN_ALGORITHM:
                continue
            found.append(path)
    return found
