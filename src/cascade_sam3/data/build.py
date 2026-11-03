"""Tile ortho + COCO (format roboflow RF-DETR) dengan split spasial dan eksklusi area uji."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil

import cv2
import numpy as np
import rasterio
import shapely
from shapely.geometry import box

from ..config import cities, work
from ..utils.geo import read_window
from .catalog import gsd_for, load_catalog
from .testsets import load_test_areas

log = logging.getLogger(__name__)
SPLITS = ("train", "valid", "test")
MIN_ANN_PX = 16  # ponytail: serpihan < 16 px dibuang, cukup untuk tile 1008


def block_hash(seed, city, bx, by) -> float:
    return int(hashlib.md5(f"{seed}:{city}:{bx}:{by}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def plan_tiles(cfg: dict, cat: dict, cov, areas) -> list[dict]:
    """Tentukan tile + split tanpa membaca piksel."""
    d = cfg["data"]
    t = cfg["testsets"]
    test_u = shapely.union_all(areas.geometry.values) if areas is not None and len(areas) else shapely.Polygon()
    excl = test_u.buffer(t["exclusion_buffer_m"] + t["context_margin_m"]) if not test_u.is_empty else test_u
    shapely.prepare(cov)
    tiles, seen = [], set()
    for ri, r in enumerate(cat["rasters"]):
        gsd = gsd_for(cfg, r)
        tile_m = d["tile_size"] * gsd
        step = max(1, round(d["tile_size"] * (1 - d["tile_overlap"]))) * gsd
        minx, miny, maxx, maxy = r["bounds"]
        xs = np.arange(minx, maxx - tile_m + 1e-6, step)
        ys = np.arange(maxy, miny + tile_m - 1e-6, -step)
        boxes = np.array([box(x, y - tile_m, x + tile_m, y) for y in ys for x in xs], dtype=object)
        if not len(boxes):
            continue
        frac = shapely.area(shapely.intersection(boxes, cov)) / (tile_m * tile_m)
        for b, f in zip(boxes, frac, strict=True):
            if f < d["min_tile_coverage"]:
                continue
            key = tuple(round(v, 2) for v in b.bounds)
            if key in seen:  # raster bertumpuk: raster pertama menang
                continue
            seen.add(key)
            c = b.centroid
            if not test_u.is_empty and test_u.contains(c):
                split = "test"
            elif not excl.is_empty and excl.intersects(b):
                continue
            else:
                bx, by = int(c.x // d["split_block_m"]), int(c.y // d["split_block_m"])
                h = block_hash(d["seed"], cat["city"], bx, by)
                split = "valid" if h < d["val_fraction"] else "train"
                tiles.append(dict(geom=b, raster=ri, gsd=gsd, split=split, block=(bx, by), h=h))
                continue
            tiles.append(dict(geom=b, raster=ri, gsd=gsd, split=split, block=None, h=None))
    tv = [x for x in tiles if x["split"] in ("train", "valid")]
    if tv and not any(x["split"] == "valid" for x in tv):
        blk = min(tv, key=lambda x: x["h"])["block"]
        for x in tv:
            if x["block"] == blk:
                x["split"] = "valid"
        log.warning("%s: valid kosong, blok %s dipaksa ke valid", cat["city"], blk)
    return tiles


def _annotations(geoms, ids, tile, gsd):
    minx, _, _, maxy = tile.bounds
    anns = []
    for g, bid in zip(geoms, ids, strict=True):
        clipped = g.intersection(tile)
        polys = [p for p in shapely.get_parts(clipped) if p.geom_type == "Polygon" and not p.is_empty]
        area_px = sum(p.area for p in polys) / gsd**2
        if area_px < MIN_ANN_PX:
            continue
        seg = []
        for p in polys:
            c = shapely.get_coordinates(p.exterior)[:-1]
            seg.append(np.c_[(c[:, 0] - minx) / gsd, (maxy - c[:, 1]) / gsd].round(2).ravel().tolist())
        x0, y0, x1, y1 = shapely.MultiPolygon(polys).bounds
        bbox = [(x0 - minx) / gsd, (maxy - y1) / gsd, (x1 - x0) / gsd, (y1 - y0) / gsd]
        anns.append(dict(segmentation=seg, bbox=[round(v, 2) for v in bbox], area=round(area_px, 2),
                         iscrowd=0, category_id=1, building_id=int(bid), truncated=bool(clipped.area < 0.99 * g.area)))
    return anns


def build_dataset(cfg: dict, city: str | None = None) -> dict:
    d = cfg["data"]
    root = work(cfg, "dataset")
    coco_dir = root / "coco"
    if coco_dir.exists():
        shutil.rmtree(coco_dir)
    for s in SPLITS:
        (coco_dir / s).mkdir(parents=True)
    (root / "overlays").mkdir(parents=True, exist_ok=True)
    coco = {s: dict(info={}, licenses=[], images=[], annotations=[],
                    categories=[{"id": 1, "name": "building", "supercategory": "none"}]) for s in SPLITS}
    rows, stats = [], {}
    img_id = ann_id = 0
    ext = "." + d["image_format"].lower().lstrip(".")
    for name in cities(cfg, city):
        cat, labels, cov = load_catalog(cfg, name)
        areas = load_test_areas(cfg, name)
        if areas is None:
            log.warning("%s: test_areas belum ada; jalankan make-testsets dulu", name)
        tiles = plan_tiles(cfg, cat, cov, areas)
        geoms = labels.geometry.values
        ids = labels.building_id.values
        tree = shapely.STRtree(geoms)
        rng = np.random.default_rng(d["seed"])
        keep = []
        for t in tiles:
            t["bids"] = tree.query(t["geom"], predicate="intersects")
            if len(t["bids"]) or t["split"] == "test" or rng.random() < d["keep_empty_tile_ratio"]:
                keep.append(t)
        tv = [t for t in keep if t["split"] != "test"]
        if len(tv) > d["max_tiles_per_city"]:
            tv = [tv[i] for i in sorted(rng.choice(len(tv), d["max_tiles_per_city"], replace=False))]
        keep = tv + [t for t in keep if t["split"] == "test"]
        st = {s: dict(tiles=0, buildings=0) for s in SPLITS}
        dsets = {}
        written = []
        try:
            for i, t in enumerate(keep):
                r = cat["rasters"][t["raster"]]
                ds = dsets.get(t["raster"]) or dsets.setdefault(t["raster"], rasterio.open(r["path"]))
                rgb, valid = read_window(ds, t["geom"].bounds, (d["tile_size"], d["tile_size"]), r.get("stretch"))
                if 1 - valid.mean() > d["max_nodata_fraction"]:
                    continue
                anns = _annotations(geoms[t["bids"]], ids[t["bids"]], t["geom"], t["gsd"])
                fname = f"{name}_{t['raster']:03d}_{i:06d}{ext}"
                params = [cv2.IMWRITE_JPEG_QUALITY, d["jpeg_quality"]] if ext in (".jpg", ".jpeg") else []
                cv2.imwrite(str(coco_dir / t["split"] / fname), rgb[..., ::-1], params)
                img_id += 1
                coco[t["split"]]["images"].append(dict(id=img_id, file_name=fname, width=d["tile_size"],
                                                       height=d["tile_size"]))
                for a in anns:
                    ann_id += 1
                    coco[t["split"]]["annotations"].append(dict(id=ann_id, image_id=img_id, **a))
                st[t["split"]]["tiles"] += 1
                st[t["split"]]["buildings"] += len(anns)
                rows.append(dict(image_id=img_id, city=name, split=t["split"], file=fname, raster=r["path"],
                                 minx=t["geom"].bounds[0], miny=t["geom"].bounds[1], maxx=t["geom"].bounds[2],
                                 maxy=t["geom"].bounds[3], gsd=t["gsd"], n_buildings=len(anns)))
                if anns:
                    written.append((t["split"], fname, anns))
        finally:
            for ds in dsets.values():
                ds.close()
        _overlays(coco_dir, root / "overlays", name, written, rng)
        stats[name] = st
        log.info("%s: %s", name, st)
    for s in SPLITS:
        (coco_dir / s / "_annotations.coco.json").write_text(json.dumps(coco[s]), encoding="utf-8")
    with open(root / "tiles.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["image_id"])
        w.writeheader()
        w.writerows(rows)
    (root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def _overlays(coco_dir, out_dir, city, written, rng, n=12):
    if not written:
        return
    for k, i in enumerate(rng.choice(len(written), min(n, len(written)), replace=False)):
        split, fname, anns = written[i]
        img = cv2.imread(str(coco_dir / split / fname))
        for a in anns:
            for seg in a["segmentation"]:
                cv2.polylines(img, [np.round(np.array(seg).reshape(-1, 2)).astype(np.int32)], True, (0, 255, 255), 1)
        cv2.imwrite(str(out_dir / f"{city}_{k:02d}.jpg"), img)
