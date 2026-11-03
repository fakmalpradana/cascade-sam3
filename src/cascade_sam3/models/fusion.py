"""Fusi detektor + konsep, refinement kotak, mask NMS, resolusi overlap."""

from __future__ import annotations

import numpy as np

from .instances import Instances


def mask_iou(a: np.ndarray, b: np.ndarray, stride: int = 4):
    """-> iou (Na,Nb), inter, area_a, area_b (di grid stride)."""
    A = a[:, ::stride, ::stride].reshape(len(a), -1).astype(np.float32)
    B = b[:, ::stride, ::stride].reshape(len(b), -1).astype(np.float32)
    inter = A @ B.T
    aa, ab = A.sum(1), B.sum(1)
    iou = inter / np.maximum(aa[:, None] + ab[None] - inter, 1)
    return iou, inter, aa, ab


def fuse(det: Instances, con: Instances, ic: dict) -> Instances:
    w = ic["concept_score_weight"]
    if not len(con):
        return det
    if len(det):
        iou = mask_iou(det.masks, con.masks)[0]
        j = iou.argmax(1)
        matched = iou[np.arange(len(det)), j] >= ic["match_iou"]
        sd = det.scores.copy()
        sd[matched] = sd[matched] + (1 - sd[matched]) * w * con.scores[j[matched]]
        src = ["det+concept" if m else s for s, m in zip(det.sources, matched, strict=True)]
        det = Instances(det.masks, sd, src, det.refined)
        con_matched = (iou >= ic["match_iou"]).any(0)
    else:
        con_matched = np.zeros(len(con), bool)
    add = np.flatnonzero(~con_matched & (con.scores >= ic["concept_add_threshold"]))
    if not len(add):
        return det
    extra = con.select(add)
    extra = Instances(extra.masks, extra.scores * w, ["concept"] * len(add))
    return Instances.cat(det, extra)


def pad_boxes(boxes: np.ndarray, h: int, w: int, frac: float = 0.03, min_px: float = 2) -> np.ndarray:
    b = boxes.astype(np.float32).copy()
    px = np.maximum((b[:, 2] - b[:, 0]) * frac, min_px)
    py = np.maximum((b[:, 3] - b[:, 1]) * frac, min_px)
    b[:, 0] -= px
    b[:, 2] += px
    b[:, 1] -= py
    b[:, 3] += py
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    return b


def refine(inst: Instances, rgb: np.ndarray, refiner, ic: dict) -> Instances:
    """refiner(rgb, boxes) -> (masks (N,H,W) bool, iou_pred (N,))."""
    if not len(inst):
        return inst
    h, w = rgb.shape[:2]
    new, iou_pred = refiner(rgb, pad_boxes(inst.boxes(), h, w))
    inter = np.logical_and(new, inst.masks).sum((1, 2))
    union = np.logical_or(new, inst.masks).sum((1, 2))
    cons = inter / np.maximum(union, 1)
    ok = (np.asarray(iou_pred) >= ic["refine_min_iou_pred"]) & (cons >= ic["refine_consistency_iou"])
    masks = inst.masks.copy()
    masks[ok] = new[ok]
    return Instances(masks, inst.scores, inst.sources, inst.refined | ok)


def mask_nms(inst: Instances, iou_thr: float, cont_thr: float) -> Instances:
    if len(inst) < 2:
        return inst
    iou, inter, aa, _ = mask_iou(inst.masks, inst.masks)
    cont = inter / np.maximum(np.minimum(aa[:, None], aa[None]), 1)
    sup = np.zeros(len(inst), bool)
    keep = []
    for i in np.argsort(-inst.scores, kind="stable"):
        if sup[i]:
            continue
        keep.append(i)
        sup |= (iou[i] > iou_thr) | (cont[i] > cont_thr)
    return inst.select(keep)


def resolve_overlaps(inst: Instances, valid: np.ndarray | None = None, min_area_px: float = 0) -> Instances:
    """Tiap piksel milik instance skor tertinggi; potong ke piksel valid; buang yang kecil."""
    if not len(inst):
        return inst
    label = np.full(inst.masks.shape[1:], -1, np.int32)
    for i in np.argsort(inst.scores, kind="stable"):
        label[inst.masks[i]] = i
    if valid is not None:
        label[~valid] = -1
    masks = np.stack([label == i for i in range(len(inst))])
    out = Instances(masks, inst.scores, inst.sources, inst.refined)
    return out.select(np.flatnonzero(masks.sum((1, 2)) >= max(min_area_px, 1)))
