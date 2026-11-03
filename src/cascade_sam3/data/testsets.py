"""Pilih & ekspor area uji (default 3 per kota, satu per kelas kepadatan)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
import shapely
from shapely.geometry import box

from ..config import cities, work
from ..utils.geo import read_window, write_rgb_tif
from .catalog import gsd_for, load_catalog, raster_for

log = logging.getLogger(__name__)
CLASSES = {1: ["sedang"], 2: ["rendah", "tinggi"], 3: ["rendah", "sedang", "tinggi"]}


def _candidates(labels: gpd.GeoDataFrame, cov, size: float) -> list:
    b = labels.groupby("src_file").geometry.agg(lambda s: tuple(s.total_bounds))
    sides = np.array([[x[2] - x[0], x[3] - x[1]] for x in b]).ravel() if len(b) else np.array([0.0])
    if 0.6 * size <= np.median(sides) <= 1.4 * size:
        centers = [((x[0] + x[2]) / 2, (x[1] + x[3]) / 2) for x in b]
    else:
        minx, miny, maxx, maxy = cov.bounds
        step = size / 2
        centers = [(x, y) for x in np.arange(minx + size / 2, maxx - size / 2 + 1e-6, step)
                   for y in np.arange(miny + size / 2, maxy - size / 2 + 1e-6, step)]
    h = size / 2
    return [box(cx - h, cy - h, cx + h, cy + h) for cx, cy in centers]


def _valid_fraction(raster: dict, geom) -> float:
    with rasterio.open(raster["path"]) as ds:
        _, valid = read_window(ds, geom.bounds, (96, 96), raster.get("stretch"))
    return float(valid.mean())


def select_areas(cfg: dict, cat: dict, labels: gpd.GeoDataFrame, cov) -> gpd.GeoDataFrame:
    t = cfg["testsets"]
    size, margin = t["area_size_m"], t["context_margin_m"]
    tree = shapely.STRtree(labels.geometry.values)
    cents = labels.geometry.centroid.values
    rows = []
    for a in _candidates(labels, cov, size):
        if a.intersection(cov).area < 0.98 * a.area:
            continue
        ctx = a.buffer(margin, join_style="mitre")
        r = raster_for(cat, ctx)
        if r is None:
            continue
        idx = tree.query(a)
        n = int(shapely.contains(a, cents[idx]).sum()) if len(idx) else 0
        if n < t["min_buildings"]:
            continue
        if _valid_fraction(r, a) < 0.95:
            continue
        roof = float(shapely.area(shapely.intersection(labels.geometry.values[idx], a)).sum())
        rows.append(dict(geometry=a, n_buildings=n, density=roof / a.area, raster=r["path"]))
    if not rows:
        raise RuntimeError(f"{cat['city']}: tidak ada kandidat area uji yang lolos syarat")
    cand = gpd.GeoDataFrame(rows, crs=cat["crs"])
    n = min(t["areas_per_city"], len(cand))
    rng = np.random.default_rng(cfg["data"]["seed"])
    cand["tiebreak"] = rng.random(len(cand))
    order = np.argsort(cand.density.values, kind="stable")
    cand["group"] = 0
    for g, ix in enumerate(np.array_split(order, n)):
        cand.loc[cand.index[ix], "group"] = g
    names = CLASSES.get(n, [f"q{i}" for i in range(n)])
    picked: list[int] = []
    sep = t["min_separation_m"]
    for g in sorted(range(n), key=lambda g: (abs(g - (n - 1) / 2), g)):
        grp = cand[cand.group == g]
        med = grp.density.median()
        grp = grp.assign(d=(grp.density - med).abs()).sort_values(["d", "tiebreak"])
        for s in (sep, sep / 2, 0):
            ok = [i for i in grp.index
                  if all(not cand.geometry[i].intersects(cand.geometry[j])
                         and cand.geometry[i].distance(cand.geometry[j]) >= s for j in picked)]
            if ok:
                picked.append(ok[0])
                break
    sel = cand.loc[picked].copy()
    sel["density_class"] = [names[g] for g in sel.group]
    sel = sel.sort_values("density").reset_index(drop=True)
    sel["city"] = cat["city"]
    sel["name"] = [f"{cat['city']}_{i + 1:02d}" for i in range(len(sel))]
    return sel[["city", "name", "density", "density_class", "n_buildings", "raster", "geometry"]]


def _manual(cfg: dict, cat: dict) -> gpd.GeoDataFrame | None:
    path = cfg["testsets"].get("manual_areas")
    if not path:
        return None
    m = gpd.read_file(path).to_crs(cat["crs"])
    m = m[m["city"] == cat["city"]].reset_index(drop=True)
    if m.empty:
        return None
    m["density"] = np.nan
    m["density_class"] = "manual"
    m["n_buildings"] = -1
    margin = cfg["testsets"]["context_margin_m"]
    m["raster"] = [(raster_for(cat, g.buffer(margin)) or {}).get("path") for g in m.geometry]
    return m[["city", "name", "density", "density_class", "n_buildings", "raster", "geometry"]]


def export_area(cfg: dict, cat: dict, labels: gpd.GeoDataFrame, row) -> Path:
    margin = cfg["testsets"]["context_margin_m"]
    d = work(cfg, "testsets", row["name"])
    d.mkdir(parents=True, exist_ok=True)
    a = row.geometry
    ctx = a.buffer(margin, join_style="mitre")
    r = next(x for x in cat["rasters"] if x["path"] == row["raster"])
    gsd = gsd_for(cfg, r)
    minx, miny, maxx, maxy = ctx.bounds
    w, h = round((maxx - minx) / gsd), round((maxy - miny) / gsd)
    with rasterio.open(r["path"]) as ds:
        rgb, valid = read_window(ds, ctx.bounds, (h, w), r.get("stretch"))
    write_rgb_tif(d / "image.tif", rgb, valid, minx, maxy, gsd, cat["crs"])
    gt = labels[labels.intersects(ctx)]
    gt.to_file(d / "gt.gpkg", layer="gt", driver="GPKG")
    gpd.GeoDataFrame({"name": [row["name"]], "city": [row["city"]]}, geometry=[a], crs=cat["crs"]).to_file(
        d / "area.gpkg", driver="GPKG")
    _quicklook(d / "quicklook.png", rgb, gt.geometry.values, a, minx, maxy, gsd)
    meta = dict(name=row["name"], city=row["city"], raster=r["path"], source=r["source"], gsd=gsd,
                n_gt=int(gt.geometry.centroid.within(a).sum()), density=float(row["density"]),
                density_class=row["density_class"], bounds=list(a.bounds), crs=cat["crs"])
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return d


def _quicklook(path, rgb, geoms, area, minx, maxy, gsd, max_px=1600):
    s = min(1.0, max_px / max(rgb.shape[:2]))
    img = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)[..., ::-1].copy()

    def px(g):
        c = shapely.get_coordinates(g.exterior)
        return np.round(np.c_[(c[:, 0] - minx) / gsd * s, (maxy - c[:, 1]) / gsd * s]).astype(np.int32)

    cv2.polylines(img, [px(g) for g in geoms], True, (0, 255, 255), 1)
    cv2.polylines(img, [px(area)], True, (255, 0, 255), 2)
    cv2.imwrite(str(path), img)


def make_testsets(cfg: dict, city: str | None = None) -> gpd.GeoDataFrame:
    out = []
    path = work(cfg, "testsets", "test_areas.gpkg")
    path.parent.mkdir(parents=True, exist_ok=True)
    for name in cities(cfg, city):
        cat, labels, cov = load_catalog(cfg, name)
        sel = _manual(cfg, cat)
        if sel is None:
            sel = select_areas(cfg, cat, labels, cov)
        for _, row in sel.iterrows():
            export_area(cfg, cat, labels, row)
            log.info("area uji %s: %s, kepadatan %.3f, %d bangunan", row["name"], row["density_class"],
                     row["density"], row["n_buildings"])
        sel.to_file(path, layer=name, driver="GPKG")
        out.append(sel)
    return out


def load_test_areas(cfg: dict, city: str) -> gpd.GeoDataFrame | None:
    path = work(cfg, "testsets", "test_areas.gpkg")
    if not path.exists():
        return None
    if city not in [n for n, _ in pyogrio.list_layers(path)]:
        return None
    return gpd.read_file(path, layer=city)
