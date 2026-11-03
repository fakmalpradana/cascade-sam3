"""Audit data: reports/data_report.{md,json}."""

from __future__ import annotations

import json
import logging

import numpy as np
import pyogrio
import rasterio
from shapely.geometry import box

from ..config import cities, work
from . import ecw, sources

log = logging.getLogger(__name__)


def _raster_info(cfg: dict, path) -> dict:
    d = dict(name=path.name, path=str(path), size_gb=round(path.stat().st_size / 1e9, 3))
    try:
        with rasterio.open(path) as ds:
            d.update(readable=True, driver=ds.driver, crs=ds.crs.to_string() if ds.crs else None,
                     gsd=ds.res[0], width=ds.width, height=ds.height, bands=ds.count, dtype=ds.dtypes[0],
                     nodata=ds.nodata, bounds=list(ds.bounds))
    except Exception as e:  # noqa: BLE001
        d.update(readable=False, error=str(e)[:200])
        try:
            info = ecw.gdalinfo_json(cfg, path)
            gt = info.get("geoTransform", [0, 0])
            d.update(driver=info.get("driverShortName"), crs=ecw.info_crs(info),
                     gsd=abs(gt[1]), width=info["size"][0], height=info["size"][1],
                     bands=len(info.get("bands", [])), nodata=(info.get("bands") or [{}])[0].get("noDataValue"),
                     bounds=list(ecw.info_bounds(info)), via="gdalinfo QGIS")
        except Exception as e2:  # noqa: BLE001
            d["gdalinfo_error"] = str(e2)[:200]
    return d


def _vector_info(path) -> list[dict]:
    out = []
    for layer, gtype in pyogrio.list_layers(path):
        info = pyogrio.read_info(path, layer=layer)
        d = dict(file=str(path), layer=str(layer), geometry_type=gtype, crs=info.get("crs"),
                 features=int(info.get("features", -1)), fields=list(info["fields"]))
        text = [f for f, t in zip(info["fields"], info["dtypes"], strict=False) if t == "object"]
        if text:
            df = pyogrio.read_dataframe(path, layer=layer, columns=text, read_geometry=False)
            d["unique_values"] = {c: sorted(map(str, df[c].dropna().unique()))[:25]
                                  for c in text if df[c].nunique() <= 25}
        out.append(d)
    return out


def _work_crs(rasters: list[dict]) -> str | None:
    """CRS untuk statistik label: raster pertama ber-CRS, termasuk yang hanya terbaca via gdalinfo (ECW)."""
    return next((r["crs"] for r in rasters if r.get("crs")), None)


def inspect(cfg: dict, city: str | None = None) -> dict:
    report = {}
    for name, ccfg in cities(cfg, city).items():
        rasters = [_raster_info(cfg, p) for p in sources.raster_files(ccfg, cfg)]
        vectors = []
        for f in sources.vector_files(ccfg, cfg):
            try:
                vectors += _vector_info(f)
            except Exception as e:  # noqa: BLE001
                vectors.append(dict(file=str(f), error=str(e)[:200]))
        crs = _work_crs(rasters)
        labels = sources.load_labels(ccfg, cfg, crs=crs, min_area=0.0, do_dedup=False)
        lab = dict(count=len(labels))
        if len(labels):
            a = labels.area.values
            lb = labels.total_bounds
            first = labels[labels.src_file == labels.src_file.iloc[0]].total_bounds
            lab.update(area_p5=float(np.percentile(a, 5)), area_p50=float(np.percentile(a, 50)),
                       area_p95=float(np.percentile(a, 95)), n_below_4m2=int((a < 4).sum()),
                       bbox=list(map(float, lb)), crs=str(labels.crs),
                       first_file_extent_m=[float(first[2] - first[0]), float(first[3] - first[1])])
            lbox = box(*lb)
            for r in rasters:
                if r.get("bounds"):
                    rb = box(*r["bounds"])
                    r["label_overlap_fraction"] = round(rb.intersection(lbox).area / max(rb.area, 1e-9), 4)
        report[name] = dict(rasters=rasters, vectors=vectors, labels=lab)
    d = work(cfg, "reports")
    d.mkdir(parents=True, exist_ok=True)
    (d / "data_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (d / "data_report.md").write_text(_to_md(report), encoding="utf-8")
    log.info("laporan: %s", d / "data_report.md")
    return report


def _to_md(report: dict) -> str:
    L = ["# Laporan data\n"]
    for name, r in report.items():
        L += [f"## {name}\n", "### Raster\n",
              "| nama | driver | CRS | GSD | px | band | GB | rasterio | overlap label |",
              "|---|---|---|---|---|---|---|---|---|"]
        for x in r["rasters"]:
            L.append(f"| {x['name']} | {x.get('driver')} | {str(x.get('crs'))[:40]} | {x.get('gsd')} | "
                     f"{x.get('width')}x{x.get('height')} | {x.get('bands')} | {x['size_gb']} | "
                     f"{x.get('readable')} | {x.get('label_overlap_fraction')} |")
        L += ["\n### Vektor\n"]
        for v in r["vectors"]:
            L.append(f"- `{v['file']}` layer `{v.get('layer')}`: {v.get('geometry_type')}, "
                     f"{v.get('features')} fitur, CRS {str(v.get('crs'))[:40]}, kolom {v.get('fields')}")
            for c, vals in (v.get("unique_values") or {}).items():
                L.append(f"  - `{c}`: {vals}")
            if "error" in v:
                L.append(f"  - ERROR: {v['error']}")
        L += ["\n### Label\n", "```", json.dumps(r["labels"], indent=2), "```\n"]
    return "\n".join(L)
