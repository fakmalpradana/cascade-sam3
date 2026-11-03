"""Config: default.yaml + configs/local.yaml (atau --config) + --set key=val."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).with_name("default.yaml")
LOCAL_PATH = Path("configs/local.yaml")


def deep_merge(base: dict, other: dict) -> dict:
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def set_key(cfg: dict, dotted: str, value) -> None:
    *parents, last = dotted.split(".")
    node = cfg
    for p in parents:
        node = node.setdefault(p, {})
    node[last] = value


def parse_set(item: str) -> tuple[str, object]:
    if "=" not in item:
        raise ValueError(f"--set butuh format key=val, dapat: {item!r}")
    k, v = item.split("=", 1)
    return k.strip(), yaml.safe_load(v)


def load_config(path: str | Path | None = None, sets=(), overrides: dict | None = None,
                use_local: bool = True) -> dict:
    cfg = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
    if path is None and use_local and LOCAL_PATH.exists():
        path = LOCAL_PATH
    if path is not None:
        deep_merge(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    if overrides:
        deep_merge(cfg, overrides)
    for item in sets:
        set_key(cfg, *parse_set(item))
    cfg.setdefault("cities", {})
    cfg["work_dir"] = str(Path(cfg["work_dir"]).expanduser().resolve())
    return cfg


def work(cfg: dict, *parts) -> Path:
    return Path(cfg["work_dir"]).joinpath(*parts)


def cities(cfg: dict, only: str | None = None) -> dict:
    cs = cfg.get("cities") or {}
    if only:
        if only not in cs:
            raise KeyError(f"kota {only!r} tidak ada di config (ada: {list(cs)})")
        return {only: cs[only]}
    return cs
