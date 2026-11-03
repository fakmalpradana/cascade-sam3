"""Cari file raster/vektor dan muat label bangunan yang harmonis."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from shapely.geometry import box

log = logging.getLogger(__name__)


def as_list(x) -> list:
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def find_files(paths, exts, include_gdb: bool = False) -> list[Path]:
    """Rekursif; folder .gdb diperlakukan sebagai satu sumber. Sidecar SHP otomatis terabaikan (ekstensi lain)."""
    exts = {e.lower() for e in exts}
    out: list[Path] = []
    for root in map(Path, as_list(paths)):
        if root.is_file() or (include_gdb and root.suffix.lower() == ".gdb"):
            out.append(root)
            continue
        if not root.exists():
            log.warning("path tidak ada: %s", root)
            continue
        for f in sorted(root.rglob("*")):
            if any(p.lower().endswith(".gdb") for p in f.relative_to(root).parts[:-1]):
                continue
            if f.is_dir():
                if include_gdb and f.suffix.lower() == ".gdb":
                    out.append(f)
            elif f.suffix.lower() in exts:
                out.append(f)
    return out


def vector_files(city_cfg: dict, cfg: dict) -> list[Path]:
    return find_files(city_cfg["labels"], cfg["data"]["vector_exts"], include_gdb=True)


def raster_files(city_cfg: dict, cfg: dict) -> list[Path]:
    return find_files(city_cfg["ortho"], cfg["data"]["raster_exts"])


def to_polygons(geoms) -> np.ndarray:
    """force_2d, garis tertutup -> poligon, make_valid, pecah multipart, hanya Polygon."""
    g = shapely.force_2d(np.asarray(geoms, dtype=object))
    g = g[~shapely.is_missing(g)]
    tid = shapely.get_type_id(g)
    is_line = np.isin(tid, [1, 2, 5])  # LineString, LinearRing, MultiLineString
    if is_line.any():
        g[is_line] = [shapely.polygonize(shapely.get_parts(x)) for x in g[is_line]]
    g = shapely.make_valid(g)
    parts = shapely.get_parts(g)
    while (shapely.get_type_id(parts) == 7).any():  # GeometryCollection bersarang
        parts = shapely.get_parts(parts)
    parts = parts[shapely.get_type_id(parts) == 3]
    return parts[~shapely.is_empty(parts)]


def _apply_filter(df: pd.DataFrame, flt: dict | None) -> pd.DataFrame:
    if not flt:
        return df
    col = flt["column"]
    if col not in df.columns:
        log.warning("kolom filter %r tidak ada; layer dilewati", col)
        return df.iloc[:0]
    vals = {str(v).strip().upper() for v in as_list(flt["values"])}
    return df[df[col].astype(str).str.strip().str.upper().isin(vals)]


def dedup(gdf: gpd.GeoDataFrame, iou_thr: float = 0.9) -> gpd.GeoDataFrame:
    """Buang duplikat di sambungan grid (IoU >= iou_thr)."""
    g = gdf.geometry.values
    i, j = shapely.STRtree(g).query(g, predicate="intersects")
    keep = i < j
    i, j = i[keep], j[keep]
    if not len(i):
        return gdf
    inter = shapely.area(shapely.intersection(g[i], g[j]))
    a = shapely.area(g)
    iou = inter / np.maximum(a[i] + a[j] - inter, 1e-9)
    drop = np.unique(j[iou >= iou_thr])
    if len(drop):
        log.info("dedup: %d poligon duplikat dibuang", len(drop))
    return gdf.drop(gdf.index[drop])


def load_labels(city_cfg: dict, cfg: dict, crs=None, min_area: float | None = None,
                do_dedup: bool = True) -> gpd.GeoDataFrame:
    """Gabungan semua label kota dalam satu CRS. Kolom: src_file, building_id, geometry."""
    frames = []
    for f in vector_files(city_cfg, cfg):
        try:
            layers = [str(n) for n, _ in pyogrio.list_layers(f)]
        except Exception as e:  # noqa: BLE001
            log.warning("gagal baca %s: %s", f, e)
            continue
        for layer in layers:
            try:
                df = pyogrio.read_dataframe(f, layer=layer)
            except Exception as e:  # noqa: BLE001
                log.warning("gagal baca %s:%s: %s", f, layer, e)
                continue
            if not isinstance(df, gpd.GeoDataFrame) or df.empty:
                continue
            df = _apply_filter(df, city_cfg.get("label_filter"))
            if df.crs is None:
                log.warning("%s:%s tanpa CRS; diasumsikan CRS kerja", f, layer)
            polys = to_polygons(df.geometry.values)
            if not len(polys):
                continue
            frames.append(gpd.GeoDataFrame({"src_file": [str(f)] * len(polys)}, geometry=polys, crs=df.crs or crs))
    if not frames:
        return gpd.GeoDataFrame({"src_file": [], "building_id": []}, geometry=[], crs=crs)
    crs = crs or frames[0].crs
    labels = pd.concat([fr.to_crs(crs) if fr.crs and fr.crs != crs else fr.set_crs(crs, allow_override=True)
                        for fr in frames], ignore_index=True)
    labels = gpd.GeoDataFrame(labels, geometry="geometry", crs=crs)
    if min_area is None:
        min_area = cfg["data"]["min_building_area_m2"]
    labels = labels[labels.area >= min_area]
    if do_dedup and len(labels):
        labels = dedup(labels)
    labels = labels.reset_index(drop=True)
    labels["building_id"] = np.arange(len(labels))
    return labels[["src_file", "building_id", "geometry"]]


def coverage(labels: gpd.GeoDataFrame, city_cfg: dict, footprint) -> shapely.Geometry:
    """Area di mana 'tidak ada label' berarti 'bukan bangunan', diiris footprint raster."""
    mode = city_cfg.get("coverage_mode", "buffer")
    if not len(labels):
        return shapely.Polygon()
    if mode == "file_bbox":
        geom = shapely.union_all([box(*g.total_bounds) for _, g in labels.groupby("src_file")])
    elif mode == "buffer":
        b = city_cfg.get("coverage_buffer_m", 30)
        geom = shapely.union_all(shapely.buffer(labels.geometry.values, b, quad_segs=2)).buffer(-10)
    elif mode == "polygon":
        geom = gpd.read_file(city_cfg["coverage_polygon"]).to_crs(labels.crs).union_all()
    else:
        raise ValueError(f"coverage_mode tidak dikenal: {mode}")
    return geom.intersection(footprint)
