"""Mask -> poligon, regularisasi, dan penggabungan antar tile."""

from __future__ import annotations

import numpy as np
import shapely
from rasterio.features import shapes
from shapely.geometry import shape


def mask_to_polygon(mask: np.ndarray, transform):
    """Poligon terbesar dari mask, dalam koordinat dunia; None bila kosong."""
    best = None
    for geom, _ in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        p = shape(geom)
        if best is None or p.area > best.area:
            best = p
    return best


def regularize(poly, ic: dict):
    if poly is None or poly.is_empty:
        return poly
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    if ic.get("regularize", True) and not poly.interiors:
        rect = poly.minimum_rotated_rectangle
        if rect.area > 0 and poly.area / rect.area >= ic["rect_threshold"]:
            return rect
    return poly.simplify(ic["simplify_tolerance_m"], preserve_topology=True)


def poly_nms(recs: list[dict], iou_thr: float, cont_thr: float) -> list[dict]:
    if len(recs) < 2:
        return recs
    recs = sorted(recs, key=lambda r: -r["score"])
    g = np.array([r["geometry"] for r in recs], dtype=object)
    a = shapely.area(g)
    i, j = shapely.STRtree(g).query(g, predicate="intersects")
    m = i < j
    i, j = i[m], j[m]
    inter = shapely.area(shapely.intersection(g[i], g[j]))
    iou = inter / np.maximum(a[i] + a[j] - inter, 1e-9)
    cont = inter / np.maximum(np.minimum(a[i], a[j]), 1e-9)
    bad = (iou > iou_thr) | (cont > cont_thr)
    nbr: dict[int, list[int]] = {}
    for x, y in zip(i[bad], j[bad], strict=True):
        nbr.setdefault(int(x), []).append(int(y))
    sup = np.zeros(len(recs), bool)
    for k in range(len(recs)):  # sudah urut skor turun
        if not sup[k]:
            for y in nbr.get(k, []):
                sup[y] = True
    return [r for r, s in zip(recs, sup, strict=True) if not s]


def stitch(recs: list[dict], ic: dict) -> list[dict]:
    """Gabung poligon antar tile. recs: dict(geometry, score, source, refined, partial)."""
    full = poly_nms([r for r in recs if not r["partial"]], 0.3, 0.6)
    part = [r for r in recs if r["partial"]]
    if full and part:
        fg = np.array([r["geometry"] for r in full], dtype=object)
        tree = shapely.STRtree(fg)
        keep = []
        for r in part:
            hit = tree.query(r["geometry"], predicate="intersects")
            covered = shapely.union_all(fg[hit]).intersection(r["geometry"]).area if len(hit) else 0.0
            if covered <= 0.5 * r["geometry"].area:
                keep.append(r)
        part = keep
    if not part:
        return full
    g = np.array([r["geometry"] for r in part], dtype=object)
    a = shapely.area(g)
    parent = list(range(len(part)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    i, j = shapely.STRtree(g).query(g, predicate="intersects")
    m = i < j
    i, j = i[m], j[m]
    inter = shapely.area(shapely.intersection(g[i], g[j]))
    link = inter > 0.05 * np.minimum(a[i], a[j])
    for x, y in zip(i[link], j[link], strict=True):
        parent[find(int(x))] = find(int(y))
    groups: dict[int, list[int]] = {}
    for k in range(len(part)):
        groups.setdefault(find(k), []).append(k)
    merged = []
    for idx in groups.values():
        best = max(idx, key=lambda k: part[k]["score"])
        geom = regularize(shapely.union_all(g[idx]), ic)
        merged.append(dict(part[best], geometry=geom, refined=any(part[k]["refined"] for k in idx),
                           partial=len(idx) == 1))
    return full + merged
