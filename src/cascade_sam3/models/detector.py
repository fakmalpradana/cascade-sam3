"""RF-DETR-Seg (Apache-2.0). Impor rfdetr secara lazy."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .instances import Instances

VARIANTS = {
    "seg-nano": "RFDETRSegNano",
    "seg-small": "RFDETRSegSmall",
    "seg-medium": "RFDETRSegMedium",
    "seg-large": "RFDETRSegLarge",
    "seg-xlarge": "RFDETRSegXLarge",
    "seg-2xlarge": "RFDETRSeg2XLarge",
}


def detector_class(variant: str):
    import rfdetr

    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r} tidak dikenal; pilih {list(VARIANTS)}")
    return getattr(rfdetr, VARIANTS[variant])


def load_detector_model(checkpoint: str | Path, variant: str):
    cls = detector_class(variant)
    if hasattr(cls, "from_checkpoint"):
        model = cls.from_checkpoint(str(checkpoint))
    else:
        model = cls(pretrain_weights=str(checkpoint))
    try:
        model.optimize_for_inference()
    except Exception:  # noqa: BLE001 - opsional
        pass
    return model


def boxes_to_masks(xyxy: np.ndarray, h: int, w: int) -> np.ndarray:
    m = np.zeros((len(xyxy), h, w), bool)
    for i, (x1, y1, x2, y2) in enumerate(np.round(xyxy).astype(int)):
        m[i, max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = True
    return m


class Detector:
    def __init__(self, checkpoint, variant: str, threshold: float):
        self.model = load_detector_model(checkpoint, variant)
        self.threshold = threshold

    def __call__(self, rgb: np.ndarray) -> Instances:
        det = self.model.predict(rgb, threshold=self.threshold)  # input RGB
        h, w = rgb.shape[:2]
        if len(det) == 0:
            return Instances.empty(h, w)
        masks = det.mask if det.mask is not None else boxes_to_masks(det.xyxy, h, w)
        return Instances(masks, det.confidence, ["det"] * len(det))
