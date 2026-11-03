"""Validasi prediksi vs GT pada area uji: reports/validation_<run>.{md,json,csv} + errors_<run>/."""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from rasterio.transform import from_origin

from ..config import work
from ..infer.predict import test_area_dirs
from . import metrics as M

log = logging.getLogger(__name__)


def _prep(gdf: gpd.GeoDataFrame, area, min_area: float) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    gdf = gdf[gdf.intersects(area)].copy()
    gdf["geometry"] = gdf.geometry.intersection(area)
    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.geom_type == "Polygon"]
    return gdf[gdf.area >= min_area].reset_index(drop=True)


def _read_pred(path: Path, crs) -> gpd.GeoDataFrame:
    if not path.exists():
        log.warning("prediksi tidak ada: %s (dihitung kosong)", path)
        return gpd.GeoDataFrame({"score": []}, geometry=[], crs=crs)
    g = gpd.read_file(path)
    g = g.to_crs(crs) if g.crs and g.crs != crs else g.set_crs(crs, allow_override=True)
    if "score" not in g.columns:
        g["score"] = 1.0
    return g


def evaluate_area(cfg: dict, area_dir: Path, pred_path: Path, err_dir: Path | None = None) -> dict:
    ec = cfg["eval"]
    meta = json.loads((area_dir / "meta.json").read_text(encoding="utf-8"))
    area_gdf = gpd.read_file(area_dir / "area.gpkg")
    crs, area = area_gdf.crs, area_gdf.geometry.iloc[0]
    gt = _prep(gpd.read_file(area_dir / "gt.gpkg").to_crs(crs), area, ec["min_area_m2"])
    pred = _prep(_read_pred(pred_path, crs), area, ec["min_area_m2"])
    gtg, prg = gt.geometry.values, pred.geometry.values
    scores = pred["score"].astype(float).values if len(pred) else np.zeros(0)

    with rasterio.open(area_dir / "image.tif") as ds:
        gsd = ec["pixel_gsd_m"] or ds.res[0]
    minx, miny, maxx, maxy = area.bounds
    shape = (math.ceil((maxy - miny) / gsd), math.ceil((maxx - minx) / gsd))
    pix = M.pixel_confusion(gtg, prg, area, from_origin(minx, maxy, gsd, gsd), shape)

    m = M.iou_matrix(gtg, prg)
    obj = {str(t): M.object_counts(m, t) for t in ec["iou_thresholds"]}
    res = dict(area=area_dir.name, city=meta["city"], density_class=meta.get("density_class"), pixel=pix,
               objects=obj, ap50=M.average_precision(m, scores, 0.5), ap75=M.average_precision(m, scores, 0.75),
               ap50_95=float(np.nanmean([M.average_precision(m, scores, t) for t in np.arange(0.5, 0.96, 0.05)])),
               boundary=M.boundary_f1(gtg, prg, ec["boundary_tolerance_m"]),
               area_gt=float(shapely.area(gtg).sum()), area_pred=float(shapely.area(prg).sum()),
               n_gt=len(gt), n_pred=len(pred),
               sweep={str(s): M.object_counts(m[:, scores >= s], ec["primary_iou"]) for s in ec["score_sweep"]})
    if err_dir is not None:
        r, c = M.hungarian(m, ec["primary_iou"])
        err_dir.mkdir(parents=True, exist_ok=True)
        out = err_dir / f"errors_{area_dir.name}.gpkg"
        if out.exists():
            out.unlink()
        g_iou, p_iou = np.zeros(len(gt)), np.zeros(len(pred))
        g_iou[r], p_iou[c] = m[r, c], m[r, c]
        gt_e = gt[["geometry"]].assign(status=np.where(np.isin(np.arange(len(gt)), r), "TP", "FN"), iou=g_iou)
        pr_e = pred[["geometry"]].assign(status=np.where(np.isin(np.arange(len(pred)), c), "TP", "FP"), iou=p_iou,
                                         score=scores)
        if len(gt_e):
            gt_e.to_file(out, layer="gt", driver="GPKG")
        if len(pr_e):
            pr_e.to_file(out, layer="pred", driver="GPKG")
    return res


def _summarize(rows: list[dict], cfg: dict) -> dict:
    ec = cfg["eval"]
    pix = {k: sum(r["pixel"][k] for r in rows) for k in ("tp", "fp", "fn", "tn")}
    out = dict(n_areas=len(rows), pixel=M.pixel_scores(**pix))
    for t in map(str, ec["iou_thresholds"]):
        c = {k: sum(r["objects"][t][k] for r in rows) for k in ("tp", "fp", "fn", "iou_sum")}
        out[f"obj@{t}"] = dict(M.object_scores(c["tp"], c["fp"], c["fn"]),
                               mean_iou=c["iou_sum"] / c["tp"] if c["tp"] else float("nan"))
    for k in ("ap50", "ap75", "ap50_95"):
        out[k] = float(np.nanmean([r[k] for r in rows])) if rows else float("nan")
    out["boundary_f1"] = float(np.nanmean([r["boundary"]["f1"] for r in rows])) if rows else float("nan")
    ag, ap = sum(r["area_gt"] for r in rows), sum(r["area_pred"] for r in rows)
    ng, npred = sum(r["n_gt"] for r in rows), sum(r["n_pred"] for r in rows)
    out.update(area_gt_m2=ag, area_pred_m2=ap, area_rel_err=(ap - ag) / ag if ag else float("nan"),
               n_gt=ng, n_pred=npred, count_diff=npred - ng)
    sweep = {}
    for s in map(str, ec["score_sweep"]):
        c = {k: sum(r["sweep"][s][k] for r in rows) for k in ("tp", "fp", "fn")}
        sweep[s] = M.object_scores(c["tp"], c["fp"], c["fn"])["f1"]
    out["sweep_f1"] = sweep
    valid = {k: v for k, v in sweep.items() if not math.isnan(v)}
    out["recommended_threshold"] = float(max(valid, key=valid.get)) if valid else None
    return out


def validate(cfg: dict, name: str, pred_dir: str | Path | None = None, areas=None) -> dict:
    pred_dir = Path(pred_dir) if pred_dir else work(cfg, "predictions", name)
    rep = work(cfg, "reports")
    err_dir = rep / f"errors_{name}"
    dirs = test_area_dirs(cfg, areas)
    if not dirs:
        raise FileNotFoundError("tidak ada area uji; jalankan make-testsets dulu")
    rows = [evaluate_area(cfg, d, pred_dir / f"{d.name}.gpkg", err_dir) for d in dirs]
    summary = dict(run=name, pred_dir=str(pred_dir), areas=rows,
                   cities={c: _summarize([r for r in rows if r["city"] == c], cfg)
                           for c in sorted({r["city"] for r in rows})},
                   overall=_summarize(rows, cfg))
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"validation_{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(rep / f"validation_{name}.csv", summary, cfg)
    (rep / f"validation_{name}.md").write_text(_to_md(summary, cfg), encoding="utf-8")
    o = summary["overall"]
    p = str(cfg["eval"]["primary_iou"])
    log.info("%s: F1 objek@%s %.4f | IoU piksel %.4f | boundary F1 %.4f | AP50 %.4f", name, p,
             o[f"obj@{p}"]["f1"], o["pixel"]["iou"], o["boundary_f1"], o["ap50"])
    return summary


def _flat(scope: str, s: dict, cfg: dict) -> dict:
    p = str(cfg["eval"]["primary_iou"])
    return dict(scope=scope, pixel_f1=s["pixel"]["f1"], pixel_iou=s["pixel"]["iou"], kappa=s["pixel"]["kappa"],
                obj_precision=s[f"obj@{p}"]["precision"], obj_recall=s[f"obj@{p}"]["recall"],
                obj_f1=s[f"obj@{p}"]["f1"], mean_iou=s[f"obj@{p}"]["mean_iou"], ap50=s["ap50"], ap75=s["ap75"],
                ap50_95=s["ap50_95"], boundary_f1=s["boundary_f1"], n_gt=s["n_gt"], n_pred=s["n_pred"],
                area_rel_err=s["area_rel_err"])


def _rows(summary: dict, cfg: dict) -> list[dict]:
    rows = [_flat(r["area"], _summarize([r], cfg), cfg) for r in summary["areas"]]
    rows += [_flat(c, s, cfg) for c, s in summary["cities"].items()]
    return rows + [_flat("overall", summary["overall"], cfg)]


def _write_csv(path: Path, summary: dict, cfg: dict) -> None:
    rows = _rows(summary, cfg)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _to_md(summary: dict, cfg: dict) -> str:
    rows = _rows(summary, cfg)
    cols = list(rows[0])
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    L = [f"# Validasi `{summary['run']}`\n", f"Prediksi: `{summary['pred_dir']}`\n",
         f"Objek pada IoU {cfg['eval']['primary_iou']} (Hungarian).\n",
         "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    L += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    o = summary["overall"]
    L += ["\n## Sweep skor (F1 objek, keseluruhan)\n", "| ambang | F1 |", "|---|---|"]
    L += [f"| {k} | {fmt(v)} |" for k, v in o["sweep_f1"].items()]
    L.append(f"\nRekomendasi ambang: **{o['recommended_threshold']}** "
             "(jangan dipakai untuk tuning; setel ambang pada split valid).\n")
    return "\n".join(L)
