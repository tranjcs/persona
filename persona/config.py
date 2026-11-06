"""Config loading with dotted-key CLI overrides, plus small JSONL helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must look like a.b=value, got {item!r}")
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def set_dotted(cfg: dict[str, Any], key: str, value: Any) -> None:
    node = cfg
    *parents, leaf = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value


def merged(base: dict[str, Any], **updates: Any) -> dict[str, Any]:
    out = copy.deepcopy(base)
    out.update(updates)
    return out


class Paths:
    def __init__(self, cfg: dict[str, Any], slug: str):
        p = cfg.get("paths", {})
        self.data = Path(p.get("data_dir", "data")) / slug
        self.out = Path(p.get("output_dir", "outputs")) / slug
        self.prompts = self.data / "prompts.jsonl"
        self.eval_prompts = self.data / "eval_prompts.jsonl"
        self.convos = self.data / "convos.jsonl"
        self.judged = self.data / "judged.jsonl"
        self.sft = self.data / "sft.jsonl"
        self.sft_val = self.data / "sft_val.jsonl"
        self.dpo = self.data / "dpo.jsonl"
        self.sft_model = self.out / "sft"
        self.dpo_model = self.out / "dpo"
        self.eval_dir = self.out / "eval"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
