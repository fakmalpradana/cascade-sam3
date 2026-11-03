"""Inferensi sliding window pada raster -> GPKG layer `buildings`."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from tqdm import tqdm

from ..config import work
from ..post.polygon import mask_to_polygon, regularize, stitch
from ..utils.geo import percentile_stretch, read_window

log = logging.getLogger(__name__)
COLUMNS = ["score", "source", "refined", "partial", "area_m2"]


def window_origins(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    return sorted(set(list(range(0, length - tile, step)) + [length - tile]))


def predict_raster(raster, out, predictor, cfg: dict, gsd: float | None = None) -> gpd.GeoDataFrame:
    ic = cfg["infer"]
    tile, ov = ic["tile_size"], ic["overlap_px"]
    t0 = time.time()
    recs = []
    n_tiles = 0
    with rasterio.open(raster) as ds:
        gsd = gsd or cfg["data"]["target_gsd_m"] or ds.res[0]
        minx, miny, maxx, maxy = ds.bounds
        W, H = round((maxx - minx) / gsd), round((maxy - miny) / gsd)
        stretch = percentile_stretch(ds)
        crs = ds.crs
        origins = [(ox, oy) for oy in window_origins(H, tile, ov) for ox in window_origins(W, tile, ov)]
        for ox, oy in tqdm(origins, desc=Path(raster).name, leave=False):
            b = (minx + ox * gsd, maxy - (oy + tile) * gsd, minx + (ox + tile) * gsd, maxy - oy * gsd)
            rgb, valid = read_window(ds, b, (tile, tile), stretch)
            if valid.mean() < 0.01:
                continue
            n_tiles += 1
            inst = predictor(rgb, valid, gsd)
            tr = from_origin(b[0], b[3], gsd, gsd)
            internal = (ox > 0, oy > 0, ox + tile < W, oy + tile < H)  # kiri, atas, kanan, bawah
            for i in range(len(inst)):
                m = inst.masks[i]
                touch = (internal[0] and m[:, 0].any()) or (internal[1] and m[0].any()) or \
                        (internal[2] and m[:, -1].any()) or (internal[3] and m[-1].any())
                poly = mask_to_polygon(m, tr)
                if poly is None or poly.is_empty:
                    continue
                recs.append(dict(geometry=poly if touch else regularize(poly, ic), score=float(inst.scores[i]),
                                 source=inst.sources[i], refined=bool(inst.refined[i]), partial=bool(touch)))
    recs = [r for r in stitch(recs, ic) if r["geometry"] is not None and not r["geometry"].is_empty]
    gdf = gpd.GeoDataFrame(recs, geometry="geometry", crs=crs) if recs else \
        gpd.GeoDataFrame({c: [] for c in COLUMNS}, geometry=[], crs=crs)
    gdf["area_m2"] = gdf.area
    gdf = gdf[gdf.area_m2 >= ic["min_area_m2"]][COLUMNS + ["geometry"]].reset_index(drop=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    gdf.to_file(out, layer="buildings", driver="GPKG")
    dt = time.time() - t0
    out.with_suffix(".json").write_text(json.dumps(dict(raster=str(raster), seconds=dt, tiles=n_tiles,
                                                        buildings=len(gdf), gsd=gsd), indent=2), encoding="utf-8")
    log.info("%s: %d bangunan, %d tile, %.1f s", Path(raster).name, len(gdf), n_tiles, dt)
    return gdf


def test_area_dirs(cfg: dict, names=None) -> list[Path]:
    dirs = sorted(p.parent for p in work(cfg, "testsets").glob("*/meta.json"))
    if names:
        dirs = [d for d in dirs if d.name in set(names)]
    return dirs


def predict_areas(cfg: dict, run: str, predictor, areas=None) -> Path:
    out_dir = work(cfg, "predictions", run)
    dirs = test_area_dirs(cfg, areas)
    if not dirs:
        raise FileNotFoundError("tidak ada area uji; jalankan make-testsets dulu")
    for d in dirs:
        predict_raster(d / "image.tif", out_dir / f"{d.name}.gpkg", predictor, cfg, gsd=None)
    return out_dir


def predict_cli(cfg: dict, args) -> None:
    from ..models.cascade import build_predictor

    mode = args.mode or cfg["infer"]["mode"]
    predictor = build_predictor(cfg, mode, args.detector, args.sam3, args.zero_shot_sam3)
    if args.raster:
        if not args.out:
            raise SystemExit("--raster butuh --out")
        predict_raster(args.raster, args.out, predictor, cfg)
        return
    name = args.name or mode
    out = predict_areas(cfg, name, predictor, args.areas)
    log.info("prediksi tersimpan di %s", out)
