"""Fine-tune RF-DETR-Seg pada dataset COCO hasil build-dataset."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..config import work
from ..models.detector import detector_class
from ..utils.gpu import limit_vram

log = logging.getLogger(__name__)


def resume_checkpoint(out: Path) -> Path:
    """Checkpoint Lightning berstatus lengkap (optimizer, scheduler, epoch) untuk resume, rfdetr 1.11.2:
    `last.ckpt` bila ada, selain itu `checkpoint_<epoch>.ckpt` terbaru. File *.pth hanya berisi bobot."""
    if (out / "last.ckpt").exists():
        return out / "last.ckpt"
    eps = [(int(m.group(1)), p) for p in out.glob("checkpoint_*.ckpt")
           if (m := re.fullmatch(r"checkpoint_(\d+)\.ckpt", p.name))]
    if not eps:
        raise FileNotFoundError(f"tidak ada checkpoint .ckpt untuk resume di {out}")
    return max(eps)[1]


def train_detector(cfg: dict, name: str | None = None, resume: bool = False, epochs: int | None = None):
    dc = dict(cfg["detector"])
    if dc["resolution"] % 24:
        raise ValueError(f"detector.resolution {dc['resolution']} harus kelipatan 24")
    dataset_dir = work(cfg, "dataset", "coco")
    if not (dataset_dir / "train" / "_annotations.coco.json").exists():
        raise FileNotFoundError(f"{dataset_dir} belum ada; jalankan build-dataset")
    if dc.get("max_train_images"):
        from ..data.subset import make_train_subset

        dataset_dir = make_train_subset(dataset_dir, int(dc["max_train_images"]), cfg["data"]["seed"])
    out = work(cfg, "runs", "detector", name or dc["variant"])
    out.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        dataset_dir=str(dataset_dir), output_dir=str(out), epochs=epochs or dc["epochs"],
        batch_size=dc["batch_size"], grad_accum_steps=dc["grad_accum_steps"], lr=dc["lr"],
        lr_encoder=dc["lr_encoder"], num_workers=dc["num_workers"], resolution=dc["resolution"],
        early_stopping=dc["early_stopping"], early_stopping_patience=dc["early_stopping_patience"],
        checkpoint_interval=dc["checkpoint_interval"], tensorboard=True, run_test=True,
        seed=cfg["data"]["seed"], **(dc.get("extra") or {}),
    )
    if resume:
        kwargs["resume"] = str(resume_checkpoint(out))
    (out / "cascade_train_args.json").write_text(json.dumps(dict(variant=dc["variant"], **kwargs), indent=2),
                                                  encoding="utf-8")
    log.info("RF-DETR %s -> %s", dc["variant"], out)
    limit_vram(dc.get("max_vram_fraction"))
    from . import rfdetr_patches

    rfdetr_patches.apply()
    init = dc.get("init_weights")
    cls = detector_class(dc["variant"])
    model = cls(pretrain_weights=str(Path(init)), resolution=dc["resolution"]) if init else cls()
    model.train(**kwargs)
    return out
