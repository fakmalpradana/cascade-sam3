"""Prediktor per tile: detector | concept | cascade."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..config import work
from .fusion import fuse, mask_nms, refine, resolve_overlaps
from .instances import Instances

log = logging.getLogger(__name__)
MODES = ("cascade", "detector", "concept")


class CascadePredictor:
    """__call__(rgb uint8 HxWx3, valid HxW bool, gsd) -> Instances."""

    def __init__(self, cfg: dict, mode: str, detector=None, concept=None, refiner=None):
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} tidak dikenal; pilih {MODES}")
        self.ic, self.mode = cfg["infer"], mode
        self.detector, self.concept, self.refiner = detector, concept, refiner

    def __call__(self, rgb: np.ndarray, valid: np.ndarray, gsd: float) -> Instances:
        ic = self.ic
        h, w = rgb.shape[:2]
        inst = self.detector(rgb) if self.mode in ("detector", "cascade") else Instances.empty(h, w)
        if self.concept is not None and (self.mode == "concept" or ic["use_concept"]):
            con = self.concept(rgb)
            inst = con if self.mode == "concept" else fuse(inst, con, ic)
        if self.mode == "cascade" and self.refiner is not None and ic["use_refine"]:
            inst = refine(inst, rgb, self.refiner, ic)
        inst = mask_nms(inst, ic["nms_iou"], ic["nms_containment"])
        return resolve_overlaps(inst, valid, ic["min_area_m2"] / gsd**2)


def latest_detector_ckpt(cfg: dict) -> Path | None:
    cands = sorted(work(cfg, "runs", "detector").glob("**/checkpoint_best_total.pth"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def build_predictor(cfg: dict, mode: str, detector_path=None, sam3_path=None, zero_shot: bool = False):
    from .sam3 import BoxRefiner, ConceptSegmenter, load_tracker

    ic, sc = cfg["infer"], cfg["sam3"]
    det = con = ref = None
    if mode in ("detector", "cascade"):
        from .detector import Detector

        detector_path = detector_path or latest_detector_ckpt(cfg)
        if detector_path is None:
            raise FileNotFoundError("checkpoint detektor tidak ditemukan di runs/detector; pakai --detector")
        log.info("detektor: %s", detector_path)
        det = Detector(detector_path, cfg["detector"]["variant"], ic["det_threshold"])
    if mode == "concept" or (mode == "cascade" and ic["use_concept"]):
        con = ConceptSegmenter(sc["model_id"], ic["concept_prompts"], ic["concept_threshold"], sc["amp_dtype"])
    if mode == "cascade" and ic["use_refine"]:
        if not zero_shot:
            sam3_path = Path(sam3_path or work(cfg, "runs", "sam3", "sam3_decoder_best.pt"))
            if not sam3_path.exists():
                raise FileNotFoundError(f"{sam3_path} tidak ada; latih dulu atau pakai --zero-shot-sam3")
        model, proc = load_tracker(sc["model_id"], sc["amp_dtype"], decoder_ckpt=None if zero_shot else sam3_path)
        ref = BoxRefiner(model, proc, ic["batch_boxes"])
    return CascadePredictor(cfg, mode, det, con, ref)
