"""Setel ambang inferensi pada split `valid` (bukan area uji). Keluaran model di-cache per tile, lalu tiap kombinasi
ambang hanya mengulang fusi/NMS: IoU piksel & F1 objek dihitung pada mask tile diperkecil 2x."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import work
from ..models.cascade import build_predictor
from ..models.fusion import apply_refinement, fuse, mask_iou, mask_nms, pad_boxes, resolve_overlaps, smooth_masks
from ..models.instances import Instances

log = logging.getLogger(__name__)
STRIDE = 2
GRID = {
    "det_threshold": [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6],
    "concept_add_threshold": [0.6, 0.7, 0.8, 0.9, 1.01],  # 1.01 = tak pernah menambah konsep
    "refine_smooth_px": [0, 5, 9, 13, 17],
    "refine_consistency_iou": [0.3, 0.5, 0.7, 0.85],
    "refine_min_iou_pred": [0.6, 0.7, 0.8, 0.9, 1.01],  # terakhir: 1.01 = refine mati, parameter refine lain tak berefek
}
DETECTOR_KEYS = ("det_threshold",)


@dataclass
class TileCache:
    gt: np.ndarray
    det: Instances
    con: Instances
    refined: dict = field(default_factory=dict)  # hash mask -> (mask refine, iou_pred)


def _key(mask: np.ndarray) -> int:
    return hash(mask.tobytes())


def _small(masks: np.ndarray) -> np.ndarray:
    return masks[:, ::STRIDE, ::STRIDE]


def cache_tile(gt: np.ndarray, det: Instances, con: Instances, new=None, iou_pred=None) -> TileCache:
    """Semua mask sudah ukuran penuh; `new`, `iou_pred` berurutan sesuai det lalu con."""
    d = Instances(_small(det.masks), det.scores, det.sources)
    c = Instances(_small(con.masks), con.scores, con.sources)
    refined = {}
    if new is not None:
        small = _small(new)
        for i, m in enumerate(np.concatenate([d.masks, c.masks])):
            refined[_key(m)] = (small[i], float(iou_pred[i]))
    return TileCache(_small(gt), d, c, refined)


def predict_cached(tile: TileCache, ic: dict, mode: str, min_area_px: float) -> Instances:
    inst = tile.det.select(np.flatnonzero(tile.det.scores >= ic["det_threshold"]))
    if mode == "cascade" and ic["use_concept"]:
        inst = fuse(inst, tile.con, ic)
    if mode == "cascade" and ic["use_refine"] and len(inst):
        new, ious = zip(*(tile.refined[_key(m)] for m in inst.masks), strict=True)
        px = ic.get("refine_smooth_px", 0) // STRIDE
        inst = apply_refinement(inst, smooth_masks(np.stack(new), px | 1 if px else 0), np.array(ious), ic)
    inst = mask_nms(inst, ic["nms_iou"], ic["nms_containment"])
    return resolve_overlaps(inst, None, min_area_px)


def score(tiles: list[TileCache], ic: dict, mode: str, min_area_px: float) -> dict:
    tp = fp = fn = obj_tp = n_pred = n_gt = 0
    for t in tiles:
        inst = predict_cached(t, ic, mode, min_area_px)
        pred, gt = inst.masks.any(0) if len(inst) else np.zeros(t.gt.shape[1:], bool), t.gt.any(0)
        tp += int((pred & gt).sum())
        fp += int((pred & ~gt).sum())
        fn += int((~pred & gt).sum())
        n_pred += len(inst)
        n_gt += len(t.gt)
        if len(inst) and len(t.gt):
            iou = mask_iou(inst.masks, t.gt, stride=2)[0]
            r, c = linear_sum_assignment(-iou)
            obj_tp += int((iou[r, c] >= 0.5).sum())
    prec, rec = obj_tp / max(n_pred, 1), obj_tp / max(n_gt, 1)
    return dict(pixel_iou=tp / max(tp + fp + fn, 1), obj_f1=2 * prec * rec / max(prec + rec, 1e-9),
                obj_precision=prec, obj_recall=rec)


def sweep(tiles: list[TileCache], ic: dict, mode: str, min_area_px: float, grid: dict | None = None) -> dict:
    """Satu putaran per parameter, urut GRID: pilih nilai dengan IoU piksel tertinggi, parameter lain tetap."""
    grid = grid or GRID
    keys = DETECTOR_KEYS if mode == "detector" else tuple(grid)
    ic = dict(ic)
    start = score(tiles, ic, mode, min_area_px)
    table, best_m = [], start
    for k in keys:
        best_v, best_m = ic.get(k, 0), None
        for v in grid[k]:
            m = score(tiles, {**ic, k: v}, mode, min_area_px)
            table.append(dict(param=k, value=v, **m))
            log.info("%s=%s: IoU %.4f, F1 %.4f", k, v, m["pixel_iou"], m["obj_f1"])
            if best_m is None or m["pixel_iou"] > best_m["pixel_iou"]:
                best_v, best_m = v, m
        ic[k] = best_v
    return dict(start=start, best=best_m, params={k: ic[k] for k in keys}, table=table)


def tune_infer(cfg: dict, mode: str, detector_path=None, sam3_path=None, n_tiles: int = 300,
               name: str | None = None) -> Path:
    from ..train.train_sam3 import masks_from_anns

    low = copy.deepcopy(cfg)
    low["infer"]["det_threshold"] = min(GRID["det_threshold"])
    pred = build_predictor(low, mode, detector_path, sam3_path)
    d = work(cfg, "dataset", "coco", "valid")
    coco = json.loads((d / "_annotations.coco.json").read_text(encoding="utf-8"))
    by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    idx = np.random.default_rng(cfg["data"]["seed"]).choice(len(coco["images"]), min(n_tiles, len(coco["images"])),
                                                            replace=False)
    tiles = []
    for n, i in enumerate(sorted(idx)):
        im = coco["images"][i]
        rgb = np.ascontiguousarray(cv2.imread(str(d / im["file_name"]))[..., ::-1])
        h, w = rgb.shape[:2]
        gt = masks_from_anns(by_img.get(im["id"], []), h, w).astype(bool)
        det = pred.detector(rgb)
        con = pred.concept(rgb) if pred.concept is not None else Instances.empty(h, w)
        new = iou_pred = None
        if pred.refiner is not None and len(det) + len(con):
            both = Instances.cat(det, con)
            new, iou_pred = pred.refiner(rgb, pad_boxes(both.boxes(), h, w))
        tiles.append(cache_tile(gt, det, con, new, iou_pred))
        if (n + 1) % 25 == 0:
            log.info("cache %d/%d tile", n + 1, len(idx))
    min_area_px = cfg["infer"]["min_area_m2"] / (cfg["data"]["target_gsd_m"] * STRIDE) ** 2
    res = sweep(tiles, cfg["infer"], mode, min_area_px)
    res.update(mode=mode, n_tiles=len(tiles), split="valid", detector=str(detector_path), sam3=str(sam3_path))
    out = work(cfg, "reports", f"tune_{name or mode}.json")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    log.info("awal IoU %.4f, F1 %.4f -> terbaik IoU %.4f, F1 %.4f: %s", res["start"]["pixel_iou"],
             res["start"]["obj_f1"], res["best"]["pixel_iou"], res["best"]["obj_f1"], res["params"])
    return out
