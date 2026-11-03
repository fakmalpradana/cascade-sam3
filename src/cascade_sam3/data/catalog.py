"""Katalog per kota: raster terbaca, CRS kerja, label harmonis, coverage."""

from __future__ import annotations

import json
import logging
from collections import Counter

import geopandas as gpd
import rasterio
import shapely
from shapely.geometry import box

from ..config import cities, work
from ..utils.geo import percentile_stretch
from . import ecw, sources

log = logging.getLogger(__name__)


def _open_rasters(cfg: dict, city: str, ccfg: dict):
    found, skipped = [], []
    for src in sources.raster_files(ccfg, cfg):
        path = src
        if src.suffix.lower() == ".ecw":
            conv = ecw.converted_path(cfg, city, src)
            if conv.exists():
                path = conv
        try:
            with rasterio.open(path) as ds:
                if ds.crs is None:
                    raise ValueError("tanpa CRS")
                found.append(dict(path=str(path), source=str(src), crs=ds.crs.to_string(),
                                  bounds=list(ds.bounds), res=float(ds.res[0]), width=ds.width,
                                  height=ds.height, count=ds.count, dtype=ds.dtypes[0],
                                  stretch=percentile_stretch(ds)))
        except Exception as e:  # noqa: BLE001
            reason = "ECW belum dikonversi (jalankan convert-ecw)" if src.suffix.lower() == ".ecw" else str(e)
            skipped.append(dict(path=str(src), reason=reason))
    return found, skipped


def build_catalog(cfg: dict, city: str | None = None) -> dict:
    out = {}
    for name, ccfg in cities(cfg, city).items():
        rasters, skipped = _open_rasters(cfg, name, ccfg)
        if not rasters:
            raise RuntimeError(f"{name}: tidak ada raster terbaca; skipped={skipped}")
        crs = Counter(r["crs"] for r in rasters).most_common(1)[0][0]
        for r in [r for r in rasters if r["crs"] != crs]:
            skipped.append(dict(path=r["path"], reason=f"CRS {r['crs']} != CRS kerja {crs}"))
        rasters = [r for r in rasters if r["crs"] == crs]
        for s in skipped:
            log.warning("%s: raster dilewati %s (%s)", name, s["path"], s["reason"])
        labels = sources.load_labels(ccfg, cfg, crs=crs)
        footprint = shapely.union_all([box(*r["bounds"]) for r in rasters])
        cov = sources.coverage(labels, ccfg, footprint)
        d = work(cfg, "catalog", name)
        d.mkdir(parents=True, exist_ok=True)
        labels.to_file(d / "labels.gpkg", layer="labels", driver="GPKG")
        gpd.GeoDataFrame({"city": [name]}, geometry=[cov], crs=crs).to_file(d / "coverage.gpkg", driver="GPKG")
        cat = dict(city=name, crs=crs, rasters=rasters, skipped=skipped, n_labels=len(labels),
                   coverage_km2=cov.area / 1e6)
        (d / "catalog.json").write_text(json.dumps(cat, indent=2), encoding="utf-8")
        log.info("%s: %d raster, %d label, coverage %.2f km2", name, len(rasters), len(labels), cov.area / 1e6)
        out[name] = cat
    return out


def load_catalog(cfg: dict, city: str):
    """-> (catalog dict, labels gdf, coverage geom)."""
    d = work(cfg, "catalog", city)
    if not (d / "catalog.json").exists():
        raise FileNotFoundError(f"katalog {city} belum ada; jalankan `cascade-sam3 catalog`")
    cat = json.loads((d / "catalog.json").read_text(encoding="utf-8"))
    labels = gpd.read_file(d / "labels.gpkg")
    cov = gpd.read_file(d / "coverage.gpkg").geometry.iloc[0]
    return cat, labels, cov


def raster_for(cat: dict, geom) -> dict | None:
    """Raster pertama yang memuat geom secara penuh."""
    for r in cat["rasters"]:
        if box(*r["bounds"]).contains(geom):
            return r
    return None


def gsd_for(cfg: dict, raster: dict) -> float:
    return cfg["data"]["target_gsd_m"] or raster["res"]
