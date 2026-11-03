"""Metrik piksel, objek (Hungarian), AP gaya COCO, boundary F1."""

from __future__ import annotations

import numpy as np
import shapely
from rasterio.features import rasterize
from scipy.optimize import linear_sum_assignment


def _div(a, b):
    return float(a / b) if b else float("nan")


def pixel_confusion(gt, pred, area, transform, shape) -> dict:
    def r(geoms):
        geoms = [g for g in geoms if g is not None and not g.is_empty]
        if not geoms:
            return np.zeros(shape, bool)
        return rasterize(((g, 1) for g in geoms), out_shape=shape, transform=transform, dtype=np.uint8) > 0

    m = r([area])
    g, p = r(gt)[m], r(pred)[m]
    return dict(tp=int((g & p).sum()), fp=int((~g & p).sum()), fn=int((g & ~p).sum()), tn=int((~g & ~p).sum()))


def pixel_scores(tp, fp, fn, tn) -> dict:
    n = tp + fp + fn + tn
    prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
    iou_b, iou_bg = _div(tp, tp + fp + fn), _div(tn, tn + fp + fn)
    po = _div(tp + tn, n)
    pe = _div((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), n * n)
    return dict(oa=po, precision=prec, recall=rec, f1=_div(2 * tp, 2 * tp + fp + fn), iou=iou_b, iou_bg=iou_bg,
                miou=float(np.nanmean([iou_b, iou_bg])), kappa=_div(po - pe, 1 - pe))


def iou_matrix(gt, pred) -> np.ndarray:
    gt, pred = np.asarray(gt, dtype=object), np.asarray(pred, dtype=object)
    m = np.zeros((len(gt), len(pred)), np.float32)
    if not len(gt) or not len(pred):
        return m
    gi, pj = shapely.STRtree(pred).query(gt, predicate="intersects")
    inter = shapely.area(shapely.intersection(gt[gi], pred[pj]))
    union = shapely.area(gt)[gi] + shapely.area(pred)[pj] - inter
    m[gi, pj] = inter / np.maximum(union, 1e-9)
    return m


def hungarian(m: np.ndarray, thr: float):
    """Pencocokan satu-ke-satu; -> (gt_idx, pred_idx) dengan IoU >= thr."""
    if not m.size:
        return np.zeros(0, int), np.zeros(0, int)
    w = np.where(m >= thr, m, 0)
    r, c = linear_sum_assignment(w, maximize=True)
    ok = m[r, c] >= thr
    return r[ok], c[ok]


def object_scores(tp, fp, fn) -> dict:
    return dict(tp=tp, fp=fp, fn=fn, precision=_div(tp, tp + fp), recall=_div(tp, tp + fn),
                f1=_div(2 * tp, 2 * tp + fp + fn))


def object_counts(m: np.ndarray, thr: float) -> dict:
    r, c = hungarian(m, thr)
    return dict(tp=len(r), fp=m.shape[1] - len(r), fn=m.shape[0] - len(r),
                iou_sum=float(m[r, c].sum()) if len(r) else 0.0)


def average_precision(m: np.ndarray, scores, thr: float) -> float:
    """COCO 101 titik, pencocokan greedy berurutan skor."""
    n_gt = m.shape[0]
    if n_gt == 0:
        return float("nan")
    if m.shape[1] == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="stable")
    matched = np.zeros(n_gt, bool)
    tp = np.zeros(len(order))
    for k, j in enumerate(order):
        col = np.where(matched, -1.0, m[:, j])
        i = int(col.argmax())
        if col[i] >= thr:
            matched[i] = True
            tp[k] = 1
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
    return float(np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0).mean())


def boundary_f1(gt, pred, tol: float) -> dict:
    gb = shapely.union_all(shapely.boundary(np.asarray(gt, dtype=object))) if len(gt) else shapely.LineString()
    pb = shapely.union_all(shapely.boundary(np.asarray(pred, dtype=object))) if len(pred) else shapely.LineString()
    p = _div(pb.intersection(gb.buffer(tol)).length, pb.length) if not pb.is_empty else float("nan")
    r = _div(gb.intersection(pb.buffer(tol)).length, gb.length) if not gb.is_empty else float("nan")
    if pb.is_empty and not gb.is_empty:
        p = 0.0
    if gb.is_empty and not pb.is_empty:
        r = 0.0
    f = _div(2 * p * r, p + r) if (p + r) else 0.0
    return dict(precision=p, recall=r, f1=f)
