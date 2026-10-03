from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

__all__ = [
    "load_config",
    "find_config",
    "iter_algorithm_configs",
    "resolve_model_config",
    "assemble_with_model",
    "own_of",
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


def resolve_model_config(name: str, root: str | Path) -> Path:
    """把基座参数解析成配置文件路径。

    两种写法都收：短名（``qwen2.5-1.5b`` → 在 ``configs/model/`` 下找），
    以及显式路径（``tests/fixtures/model.yaml``）。先按原样试一次，
    失败再当作短名补 ``model/`` 前缀 —— 无条件拼前缀会让显式路径永远找不到，
    而本机的冒烟测试正需要指向 tests/ 下那份 fixture 基座。
    """
    try:
        return find_config(name, root)
    except FileNotFoundError:
        return find_config(f"model/{name}", root)


def own_of(path: str | Path) -> DictConfig:
    """算法配置**自己**写了什么 —— 去掉 ``defaults`` 之后剩下的部分。

    换基座时不能简单地把新基座 merge 到已经展开好的配置上：那样**旧**基座
    写下的 ``vocab_size`` 之类的字段会留下来，被 ``build()`` 原样当成构造
    参数传给 ``hf_causal_lm``，报一个与真正原因无关的 TypeError。

    所以改成「配方」而不是「覆写」：``base + 选中的基座 + 算法自己的增量``。
    """
    raw = OmegaConf.load(Path(path))
    defaults = raw.get("defaults", None)
    if defaults is None or len(defaults) != 2 or "model" not in str(defaults[1]):
        raise ValueError(
            f"{Path(path).name} 的 defaults 不是「base + 一个基座」的形状"
            f"（实际是 {defaults}），无法安全地替换基座。\n"
            f"请手动编辑该文件的 defaults，或去掉这个基座参数。"
        )
    own = OmegaConf.to_container(raw, resolve=False)
    own.pop("defaults", None)
    return OmegaConf.create(own)


def assemble_with_model(
    algorithm_path: str | Path, model_path: str | Path | None, root: str | Path
) -> DictConfig:
    """``base + 基座 + 算法自己的增量``。

    ``model_path`` 为 None 时退化成 ``load_config``：算法配置自己的
    ``defaults`` 链就是答案。两个脚本（train / train_offline）共用这一份，
    免得「换基座」的语义在两处慢慢分叉。
    """
    algorithm_path = Path(algorithm_path)
    if model_path is None:
        return load_config(algorithm_path)
    return OmegaConf.merge(
        load_config(Path(root) / "base.yaml"),
        load_config(model_path),
        own_of(algorithm_path),
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
