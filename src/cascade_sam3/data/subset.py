"""Subset train COCO tanpa menyalin gambar (valid/test tetap utuh, agar sebanding antar run)."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SPLITS = ("train", "valid", "test")


def _city(file_name: str) -> str:
    return Path(file_name).name.split("_", 1)[0]


def sample_images(images: list[dict], n: int, seed: int) -> list[dict]:
    """Ambil `n` gambar acak, proporsional per kota (prefiks nama file), urutan asli dipertahankan."""
    if n >= len(images):
        return list(images)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, im in enumerate(images):
        groups[_city(im["file_name"])].append(i)
    rng = np.random.default_rng(seed)
    quota = {c: int(round(n * len(ix) / len(images))) for c, ix in groups.items()}
    quota[max(groups, key=lambda c: len(groups[c]))] += n - sum(quota.values())
    keep: set[int] = set()
    for c, ix in groups.items():
        keep.update(int(i) for i in rng.choice(ix, quota[c], replace=False))
    return [im for i, im in enumerate(images) if i in keep]


def make_train_subset(coco_dir: Path, n: int, seed: int = 42) -> Path:
    """Tulis `<coco_dir>_train<n>/{train,valid,test}/_annotations.coco.json` dengan file_name relatif ke gambar asli."""
    coco_dir = Path(coco_dir)
    out = coco_dir.with_name(f"{coco_dir.name}_train{n}")
    for split in SPLITS:
        src = coco_dir / split / "_annotations.coco.json"
        if not src.exists():
            continue
        d = json.loads(src.read_text(encoding="utf-8"))
        if split == "train":
            d["images"] = sample_images(d["images"], n, seed)
            ids = {im["id"] for im in d["images"]}
            d["annotations"] = [a for a in d["annotations"] if a["image_id"] in ids]
        for im in d["images"]:
            im["file_name"] = f"../../{coco_dir.name}/{split}/{im['file_name']}"
        (out / split).mkdir(parents=True, exist_ok=True)
        (out / split / "_annotations.coco.json").write_text(json.dumps(d), encoding="utf-8")
        log.info("subset %s: %d gambar, %d anotasi", split, len(d["images"]), len(d["annotations"]))
    return out
