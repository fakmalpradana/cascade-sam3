"""Fine-tune RF-DETR-Seg pada dataset COCO hasil build-dataset."""

from __future__ import annotations

import json
import logging

from ..config import work
from ..models.detector import detector_class

log = logging.getLogger(__name__)


def train_detector(cfg: dict, name: str | None = None, resume: bool = False, epochs: int | None = None):
    dc = dict(cfg["detector"])
    if dc["resolution"] % 24:
        raise ValueError(f"detector.resolution {dc['resolution']} harus kelipatan 24")
    dataset_dir = work(cfg, "dataset", "coco")
    if not (dataset_dir / "train" / "_annotations.coco.json").exists():
        raise FileNotFoundError(f"{dataset_dir} belum ada; jalankan build-dataset")
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
        ck = next((out / f for f in ("checkpoint.pth", "last.ckpt") if (out / f).exists()), None)
        if ck is None:
            raise FileNotFoundError(f"tidak ada checkpoint untuk resume di {out}")
        kwargs["resume"] = str(ck)
    (out / "cascade_train_args.json").write_text(json.dumps(dict(variant=dc["variant"], **kwargs), indent=2),
                                                  encoding="utf-8")
    log.info("RF-DETR %s -> %s", dc["variant"], out)
    model = detector_class(dc["variant"])()
    model.train(**kwargs)
    return out
