"""ECW -> GeoTIFF memakai GDAL milik QGIS/OSGeo4W (wheel rasterio tidak punya driver ECW)."""

from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from shapely.geometry import box

from ..config import cities, work
from . import sources

log = logging.getLogger(__name__)

IS_WIN = os.name == "nt"


def find_gdal_bin(cfg: dict) -> Path | None:
    if cfg["data"].get("gdal_bin"):
        return Path(cfg["data"]["gdal_bin"])
    if IS_WIN:
        for pat in (r"C:\Program Files\QGIS*\bin", r"C:\OSGeo4W*\bin"):
            for d in sorted(glob.glob(pat), reverse=True):
                if (Path(d) / "gdal_translate.exe").exists():
                    return Path(d)
    exe = shutil.which("gdal_translate")
    return Path(exe).parent if exe else None


def gdal_run(cfg: dict, tool: str, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    bindir = find_gdal_bin(cfg)
    if bindir is None:
        raise RuntimeError("GDAL QGIS tidak ditemukan; pasang QGIS LTR atau set data.gdal_bin")
    env_bat = bindir / "o4w_env.bat"
    if IS_WIN and env_bat.exists():
        # String mentah + /s: list ["cmd","/c",...] membuat Python meng-escape " jadi \" yang tak dipahami cmd.exe.
        # /q: echo off, karena o4w_env.bat mencetak tiap perintahnya ke stdout (merusak gdalinfo -json).
        inner = f'call "{env_bat}" && {subprocess.list2cmdline([tool, *map(str, args)])}'
        proc = subprocess.run(f'cmd /s /q /c "{inner}"', capture_output=True, text=True)
    else:
        proc = subprocess.run([str(bindir / tool), *map(str, args)], capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{tool} gagal ({proc.returncode}): {proc.stderr.strip()[-2000:]}")
    return proc


def has_ecw_driver(cfg: dict) -> bool:
    return "ECW" in gdal_run(cfg, "gdalinfo", ["--formats"]).stdout


def gdalinfo_json(cfg: dict, path) -> dict:
    return json.loads(gdal_run(cfg, "gdalinfo", ["-json", str(path)]).stdout)


def info_bounds(info: dict) -> tuple[float, float, float, float]:
    cc = info["cornerCoordinates"]
    xs = [cc[k][0] for k in ("upperLeft", "lowerRight", "upperRight", "lowerLeft")]
    ys = [cc[k][1] for k in ("upperLeft", "lowerRight", "upperRight", "lowerLeft")]
    return min(xs), min(ys), max(xs), max(ys)


def converted_path(cfg: dict, city: str, src: Path) -> Path:
    return work(cfg, "ortho_converted", city, src.stem + ".tif")


def convert(cfg: dict, src: Path, dst: Path, n_bands: int, projwin=None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.tif")
    args = ["-of", "GTiff", "-b", "1", "-b", "2", "-b", "3"]
    if n_bands >= 4:
        args += ["-mask", "4"]
    for co in ("TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512", "COMPRESS=JPEG", "PHOTOMETRIC=YCBCR",
               "JPEG_QUALITY=92", "BIGTIFF=YES"):
        args += ["-co", co]
    if projwin is not None:
        minx, miny, maxx, maxy = projwin
        args += ["-projwin", minx, maxy, maxx, miny]
    args += ["--config", "GDAL_TIFF_INTERNAL_MASK", "YES", "--config", "GDAL_CACHEMAX", "2048", src, tmp]
    log.info("gdal_translate %s -> %s", src, dst)
    gdal_run(cfg, "gdal_translate", args)
    gdal_run(cfg, "gdaladdo", ["-r", "average", "--config", "COMPRESS_OVERVIEW", "JPEG", tmp,
                               "2", "4", "8", "16", "32", "64"])
    os.replace(tmp, dst)


def convert_city(cfg: dict, city: str | None = None, full: bool = False, force: bool = False) -> list[Path]:
    """Konversi hanya area berlabel (+buffer) kecuali --full."""
    done = []
    buf = cfg["data"]["convert_clip_buffer_m"]
    for name, ccfg in cities(cfg, city).items():
        ecws = [p for p in sources.raster_files(ccfg, cfg) if p.suffix.lower() == ".ecw"]
        if not ecws:
            log.info("%s: tidak ada ECW", name)
            continue
        labels = None
        for src in ecws:
            dst = converted_path(cfg, name, src)
            if dst.exists() and not force:
                log.info("lewati (sudah ada): %s", dst)
                done.append(dst)
                continue
            info = gdalinfo_json(cfg, src)
            rb = info_bounds(info)
            projwin = None
            if not full:
                wkt = info.get("coordinateSystem", {}).get("wkt")
                if labels is None or labels.crs.to_wkt() != wkt:
                    labels = sources.load_labels(ccfg, cfg, crs=wkt, do_dedup=False)
                # ponytail: satu persegi bbox label per raster; bila label berpulau jauh, area kosong ikut terkonversi
                hit = labels[labels.intersects(box(*rb))]
                if hit.empty:
                    log.info("lewati (tanpa label): %s", src)
                    continue
                lb = box(*hit.total_bounds).buffer(buf, join_style="mitre").intersection(box(*rb))
                projwin = lb.bounds
            convert(cfg, src, dst, len(info.get("bands", [])), projwin)
            done.append(dst)
    return done
