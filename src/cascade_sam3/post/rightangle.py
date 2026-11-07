"""Regularisasi sudut siku (v2, v3) sebagai post-process poligon bangunan.

Port dari `regularize_v2.py` dan `regularize_v3.py` milik user. Algoritma dan parameter tidak diubah; fungsi
yang identik di kedua berkas digabung, parameter tiap versi disimpan di `V2`/`V3`. Bekerja dalam satuan meter
(CRS proyeksi). Urutan default pipeline: v2 -> v3 -> v3 (`infer.rightangle_chain`).
"""

from __future__ import annotations

import logging
import math

import numpy as np
from shapely.affinity import rotate as shp_rotate
from shapely.geometry import LinearRing, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

log = logging.getLogger(__name__)

V2 = dict(ANG_CLASSIFY_DEG=25.0, ANG_FORCE_DEG=45.0, MERGE_LINE_TOL=0.40, COLLINEAR_TOL_DEG=2.0, CLOSE_TOL=0.30,
          SMOOTH_TOL=0.10, GRID_SNAP=0.05, MAX_SNAP_OFFSET=1.80, TEETH_MAX_LEN=1.00, STAIR_SEG_MAX=1.50,
          STAIR_MIN_PAIRS=2, STAIR_MAX_DEPTH=0.60, AREA_TOL=1e-4, AREA_MATCH=True)
V3 = dict(STAIR_SEG_MAX=2.5, STAIR_MIN_PAIRS=2, STAIR_MAX_DEPTH=0.8, SPIKE_ANGLE_MAX=70.0, SPIKE_DEPTH_MAX=0.75,
          BEVEL_DIAG_MAX=2.0, RIGHT_TOL_DEG=15.0, CORNER_SHIFT_MAX=1.8, ANG_CLASSIFY_DEG=25.0, ANG_FORCE_DEG=45.0,
          MERGE_LINE_TOL=0.40, GRID_SNAP=0.05, COLLINEAR_TOL_DEG=2.0, CLOSE_TOL=0.30, SMOOTH_TOL=0.10, AREA_TOL=1e-4,
          AREA_MATCH=True)


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def dominant_angle_deg(geom) -> float:
    mrr = geom.minimum_rotated_rectangle
    if not isinstance(mrr, Polygon):
        return 0.0
    xs, ys = mrr.exterior.coords.xy
    pts = np.array(list(zip(xs, ys, strict=True)))
    edges = np.diff(pts, axis=0)[:4]
    lens = np.hypot(edges[:, 0], edges[:, 1])
    if len(lens) == 0 or np.max(lens) == 0:
        return 0.0
    v = edges[np.argmax(lens)]
    return (math.degrees(math.atan2(v[1], v[0])) + 180) % 180


def _merge_vals(vals, tol):
    if not vals:
        return []
    vals = sorted(vals)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - groups[-1][-1]) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(g)) for g in groups]


def _remove_colinear_ring(xy, tol_deg=2.0):
    pts = np.asarray(xy, float)
    if len(pts) < 4:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = out[-1], pts[i], pts[i + 1]
        v1, v2 = b - a, c - b
        if np.allclose(v1, 0) or np.allclose(v2, 0):
            continue
        ang = abs(math.degrees(math.atan2(_cross2(v1, v2), np.dot(v1, v2))))
        if abs(180 - ang) <= tol_deg or ang <= tol_deg:
            continue
        out.append(b)
    out.append(pts[-1])
    if (out[0] != out[-1]).any():
        out.append(out[0])
    return np.asarray(out)


def _line_runs_classify(coords, ang_class=25, ang_force=45):
    pts = np.asarray(coords, float)
    H, V = 0, 1
    labs = []
    for i in range(len(pts) - 1):
        dx, dy = pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            labs.append(H if (not labs or labs[-1] == H) else V)
            continue
        a = abs(math.degrees(math.atan2(dy, dx)))
        a = min(a, 180 - a)
        if a <= ang_class:
            labs.append(H)
        elif abs(a - 90) <= ang_class:
            labs.append(V)
        else:
            labs.append(H if a < ang_force else V)
    return labs


def _apply_runs_snap(coords, labels, merge_tol=0.3, grid_snap=0.0, max_shift=None, collinear_tol=2.0):
    pts = np.asarray(coords, float)
    n = len(pts) - 1
    runs, s = [], 0
    for i in range(1, n):
        if labels[i] != labels[i - 1]:
            runs.append((s, i - 1, labels[i - 1]))
            s = i
    runs.append((s, n - 1, labels[-1]))
    xs, ys = [], []
    for a, b, lab in runs:
        seg = pts[a:b + 2]
        if lab == 0:
            ys.append(np.mean(seg[:, 1]))
        else:
            xs.append(np.mean(seg[:, 0]))
    xs, ys = _merge_vals(xs, merge_tol), _merge_vals(ys, merge_tol)
    out = pts.copy()
    for a, b, lab in runs:
        if lab == 0:
            y0 = np.mean(out[a:b + 2, 1])
            y0 = min(ys, key=lambda yv, y0=y0: abs(yv - y0)) if ys else y0
            for k in range(a, b + 2):
                if max_shift is None or abs(y0 - out[k, 1]) <= max_shift:
                    out[k, 1] = y0
        else:
            x0 = np.mean(out[a:b + 2, 0])
            x0 = min(xs, key=lambda xv, x0=x0: abs(xv - x0)) if xs else x0
            for k in range(a, b + 2):
                if max_shift is None or abs(x0 - out[k, 0]) <= max_shift:
                    out[k, 0] = x0
    if grid_snap > 0:
        out[:, 0] = np.round(out[:, 0] / grid_snap) * grid_snap
        out[:, 1] = np.round(out[:, 1] / grid_snap) * grid_snap
    return _remove_colinear_ring(out, collinear_tol)


def _flatten_teeth(coords, teeth_max=1.0):
    pts = np.asarray(coords, float).copy()

    def lab(a, b):
        return 0 if abs(b[1] - a[1]) < 1e-12 else 1

    i = 0
    while i < len(pts) - 2:
        L0 = float(np.hypot(*(pts[i + 1] - pts[i])))
        L1 = float(np.hypot(*(pts[i + 2] - pts[i + 1])))
        if L0 <= teeth_max and L1 <= teeth_max and lab(pts[i], pts[i + 1]) != lab(pts[i + 1], pts[i + 2]):
            pts = np.vstack([pts[:i + 1], pts[i + 2:]])
            if i > 0:
                i -= 1
            continue
        i += 1
    if (pts[0] != pts[-1]).any():
        pts = np.vstack([pts, pts[0]])
    return pts


def _flatten_staircase(coords, seg_max=1.5, min_pairs=2, max_depth=0.6, collinear_tol=2.0):
    pts = np.asarray(coords, float)
    if len(pts) < 4:
        return pts

    def seg_lab_len(a, b):
        v = b - a
        return (0 if abs(v[1]) < abs(v[0]) else 1), float(np.hypot(*v))

    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j, labs, xs, ys = i, [], [pts[i][0]], [pts[i][1]]
        while j < len(pts) - 1:
            lab, L = seg_lab_len(pts[j], pts[j + 1])
            if L > seg_max or (labs and lab == labs[-1]):
                break
            labs.append(lab)
            j += 1
            xs.append(pts[j][0])
            ys.append(pts[j][1])
        if len(labs) // 2 >= min_pairs:
            dx, dy = max(xs) - min(xs), max(ys) - min(ys)
            if min(dx, dy) <= max_depth:
                if dx >= dy:
                    out.append(np.array([pts[j][0], float(np.median(ys))]))
                else:
                    out.append(np.array([float(np.median(xs)), pts[j][1]]))
                i = j
                continue
        out.append(pts[i + 1])
        i += 1
    out = np.asarray(out)
    if (out[0] != out[-1]).any():
        out = np.vstack([out, out[0]])
    return _remove_colinear_ring(out, collinear_tol)


def _remove_spikes_by_depth(coords, angle_max=70.0, depth_max=0.75, collinear_tol=2.0):
    P = np.asarray(coords, float).copy()
    changed = True
    while changed:
        changed = False
        for i in range(1, len(P) - 1):
            a, b, c = P[i - 1], P[i], P[i + 1]
            v1, v2 = a - b, c - b
            if float(np.hypot(*v1)) < 1e-9 or float(np.hypot(*v2)) < 1e-9:
                continue
            ang = abs(math.degrees(math.atan2(_cross2(v1, v2), np.dot(v1, v2))))
            ang = min(ang, 360 - ang)
            AC = c - a
            LAC = float(np.hypot(*AC))
            if LAC < 1e-9:
                continue
            depth = abs(_cross2(AC, b - a)) / LAC
            if ang < angle_max and depth <= depth_max:
                P = np.vstack([P[:i], P[i + 1:]])
                changed = True
                break
    if (P[0] != P[-1]).any():
        P = np.vstack([P, P[0]])
    return _remove_colinear_ring(P, collinear_tol)


def _fix_short_bevels(coords, diag_max=2.0, right_tol=15.0, max_shift=1.8, collinear_tol=2.0):
    def is_H(v):
        ang = abs(math.degrees(math.atan2(v[1], v[0])))
        return ang <= right_tol or abs(180 - ang) <= right_tol

    def is_V(v):
        return abs(abs(math.degrees(math.atan2(v[1], v[0]))) - 90) <= right_tol

    P = np.asarray(coords, float)
    if len(P) < 5:
        return P
    Q = [P[0]]
    for i in range(1, len(P) - 2):
        a, b, c, d = P[i - 1], P[i], P[i + 1], P[i + 2]
        v_prev, v_diag, v_next = b - a, c - b, d - c
        if float(np.hypot(*v_diag)) <= diag_max and \
                ((is_H(v_prev) and is_V(v_next)) or (is_V(v_prev) and is_H(v_next))):
            corner = np.array([c[0] if is_V(v_next) else b[0], b[1] if is_H(v_prev) else c[1]], float)
            if np.hypot(*(corner - b)) <= max_shift and np.hypot(*(corner - c)) <= max_shift:
                Q.append(corner)
                continue
        Q.append(b)
    Q.extend([P[-2], P[-1]])
    Q = np.asarray(Q)
    if (Q[0] != Q[-1]).any():
        Q = np.vstack([Q, Q[0]])
    return _remove_colinear_ring(Q, collinear_tol)


def _area_match(poly, target, p):
    if not p["AREA_MATCH"] or poly.area >= target * (1.0 - p["AREA_TOL"]):
        return poly
    lo, hi = 0.0, max(0.05, (target - poly.area) / max(poly.length, 1e-9) * 2.0)

    def area_of(d):
        return poly.buffer(d, join_style=2, mitre_limit=5.0).area

    while area_of(hi) < target and hi < 5.0:
        hi *= 2.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if area_of(mid) < target:
            lo = mid
        else:
            hi = mid
    return poly.buffer(hi, join_style=2, mitre_limit=5.0)


def _prepare(poly, p, angle):
    q = poly
    if p["SMOOTH_TOL"] > 0:
        q = q.buffer(p["SMOOTH_TOL"], join_style=2).buffer(-p["SMOOTH_TOL"], join_style=2)
    if p["CLOSE_TOL"] > 0:
        q = q.buffer(p["CLOSE_TOL"], join_style=2).buffer(-p["CLOSE_TOL"], join_style=2)
    return shp_rotate(make_valid(q).buffer(0), -angle, origin="centroid", use_radians=False)


def _ring_v2(arr, p):
    labs = _line_runs_classify(arr, p["ANG_CLASSIFY_DEG"], p["ANG_FORCE_DEG"])
    ct = p["COLLINEAR_TOL_DEG"]
    arr = _apply_runs_snap(arr, labs, p["MERGE_LINE_TOL"], p["GRID_SNAP"], p["MAX_SNAP_OFFSET"], ct)
    arr = _flatten_teeth(arr, p["TEETH_MAX_LEN"])
    return _flatten_staircase(arr, p["STAIR_SEG_MAX"], p["STAIR_MIN_PAIRS"], p["STAIR_MAX_DEPTH"], ct)


def _ring_v3(arr, p):
    if len(arr) < 4:
        return np.array([])
    ct = p["COLLINEAR_TOL_DEG"]
    labs = _line_runs_classify(arr, p["ANG_CLASSIFY_DEG"], p["ANG_FORCE_DEG"])
    arr = _apply_runs_snap(arr, labs, p["MERGE_LINE_TOL"], p["GRID_SNAP"], p["CORNER_SHIFT_MAX"], ct)
    arr = _flatten_teeth(arr, teeth_max=p["STAIR_SEG_MAX"] * 0.6)
    arr = _flatten_staircase(arr, p["STAIR_SEG_MAX"], p["STAIR_MIN_PAIRS"], p["STAIR_MAX_DEPTH"], ct)
    arr = _remove_spikes_by_depth(arr, p["SPIKE_ANGLE_MAX"], p["SPIKE_DEPTH_MAX"], ct)
    arr = _fix_short_bevels(arr, p["BEVEL_DIAG_MAX"], p["RIGHT_TOL_DEG"], p["CORNER_SHIFT_MAX"], ct)
    if len(arr) < 4 or not LinearRing(arr).is_valid:
        return np.array([])
    return arr


def _ortho_v2(poly, angle, p):
    rot = _prepare(poly, p, angle)

    def one(pg):
        ext = _ring_v2(np.asarray(pg.exterior.coords), p)
        holes = [h for h in (_ring_v2(np.asarray(r.coords), p) for r in pg.interiors) if len(h) >= 4]
        return Polygon(ext, holes).buffer(0) if len(ext) >= 4 else None

    if isinstance(rot, Polygon):
        out = one(rot)
    else:
        parts = [x for x in (one(pg) for pg in rot.geoms) if x is not None and not x.is_empty]
        out = unary_union(parts).buffer(0)
    if out is None or out.is_empty:
        return out
    return make_valid(shp_rotate(out, angle, origin="centroid", use_radians=False)).buffer(0)


def _ortho_v3(poly, angle, p):
    rot = _prepare(poly, p, angle)

    def one(pg):
        ext = _ring_v3(np.asarray(pg.exterior.coords), p)
        if len(ext) < 4:
            return None
        holes = [h for h in (_ring_v3(np.asarray(r.coords), p) for r in pg.interiors) if len(h) >= 4]
        try:
            q = Polygon(ext, holes)
            return None if not q.is_valid or q.is_empty else q.buffer(0)
        except Exception:  # noqa: BLE001 - sama dengan skrip asli
            return None

    if isinstance(rot, Polygon):
        out = one(rot)
    else:
        parts = [x for x in (one(pg) for pg in rot.geoms) if x is not None and not x.is_empty]
        out = unary_union(parts).buffer(0) if parts else None
    if out is None:
        return None
    return make_valid(shp_rotate(out, angle, origin="centroid", use_radians=False)).buffer(0)


def regularize_v2(geom, p=V2):
    if geom is None or geom.is_empty:
        return geom
    ortho = _ortho_v2(geom, dominant_angle_deg(geom), p)
    if ortho is None or ortho.is_empty:
        return geom
    return _area_match(ortho, geom.area, p).buffer(0)


def regularize_v3(geom, p=V3):
    if geom is None or geom.is_empty:
        return geom
    ortho = _ortho_v3(geom, dominant_angle_deg(geom), p)
    if ortho is None or ortho.is_empty:
        return geom
    return _area_match(ortho, geom.area, p).buffer(0)


STEPS = {"v2": regularize_v2, "v3": regularize_v3}


def _polygonal(g):
    """make_valid bisa menghasilkan GeometryCollection; simpan bagian poligonnya saja."""
    if g is None or g.is_empty or isinstance(g, Polygon | MultiPolygon):
        return g
    parts = [x for x in getattr(g, "geoms", []) if isinstance(x, Polygon | MultiPolygon) and not x.is_empty]
    return unary_union(parts) if parts else Polygon()


def regularize_chain(geom, steps=("v2", "v3", "v3")):
    """Terapkan langkah berurutan. Bila satu langkah gagal, poligon dari langkah sebelumnya dipertahankan."""
    for s in steps:
        if s not in STEPS:
            raise ValueError(f"langkah regularisasi tidak dikenal: {s} (pilih {sorted(STEPS)})")
    for s in steps:
        if geom is None or geom.is_empty:
            return geom
        try:
            out = _polygonal(STEPS[s](make_valid(geom).buffer(0)))
        except Exception as e:  # noqa: BLE001 - satu poligon buruk tidak boleh menggagalkan seluruh raster
            log.debug("regularisasi %s gagal: %s", s, e)
            continue
        if out is not None and not out.is_empty:
            geom = out
    return geom


def regularize_gdf(gdf, steps=("v2", "v3", "v3")):
    """Regularisasi semua geometri GeoDataFrame (CRS geografis diproyeksikan ke UTM dulu)."""
    if not steps or gdf.empty:
        return gdf
    crs = gdf.crs
    work = gdf.to_crs(gdf.estimate_utm_crs()) if crs is not None and crs.is_geographic else gdf.copy()
    work["geometry"] = [regularize_chain(g, steps) for g in work.geometry]
    work = work[~work.geometry.is_empty & work.geometry.notna()]
    if "area_m2" in work.columns:
        work["area_m2"] = work.area
    return work.to_crs(crs) if crs is not None and work.crs != crs else work
