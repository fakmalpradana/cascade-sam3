"""Helper raster: baca jendela RGB + mask valid, tulis GeoTIFF RGB, stretch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.windows import from_bounds


def percentile_stretch(ds, lo=1, hi=99) -> list[list[float]] | None:
    """Persentil per band dari overview kecil; None bila uint8."""
    if ds.dtypes[0] == "uint8":
        return None
    scale = max(1, max(ds.width, ds.height) // 1024)
    data = ds.read(_bands(ds), out_shape=(3, ds.height // scale, ds.width // scale),
                   resampling=Resampling.average, masked=True)
    return [[float(np.percentile(b.compressed(), lo)), float(np.percentile(b.compressed(), hi))] for b in data]


def _bands(ds):
    return [1, 2, 3] if ds.count >= 3 else [1, 1, 1]


def read_window(ds, bounds, size, stretch=None):
    """Baca bounds (minx,miny,maxx,maxy) ke RGB uint8 (h,w,3) + mask valid (h,w)."""
    h, w = size
    win = from_bounds(*bounds, transform=ds.transform)
    inside = (win.col_off >= -1e-6 and win.row_off >= -1e-6
              and win.col_off + win.width <= ds.width + 1e-6 and win.row_off + win.height <= ds.height + 1e-6)
    kw = dict(window=win, boundless=not inside)
    data = ds.read(_bands(ds), out_shape=(3, h, w), resampling=Resampling.bilinear, fill_value=0, **kw)
    valid = ds.read_masks(1, out_shape=(h, w), resampling=Resampling.nearest, **kw) > 0
    if stretch:
        out = np.empty((3, h, w), np.uint8)
        for i, (lo, hi) in enumerate(stretch):
            out[i] = np.clip((data[i].astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
        data = out
    else:
        data = data.astype(np.uint8)
    valid &= data.any(0)  # collar hitam ECW tanpa nodata
    return np.ascontiguousarray(data.transpose(1, 2, 0)), valid


def write_rgb_tif(path: str | Path, rgb: np.ndarray, valid: np.ndarray, minx: float, maxy: float, gsd: float, crs):
    """GeoTIFF RGB JPEG/YCbCr tiled + internal mask."""
    h, w = rgb.shape[:2]
    profile = dict(driver="GTiff", width=w, height=h, count=3, dtype="uint8", crs=crs,
                   transform=from_origin(minx, maxy, gsd, gsd), tiled=True, blockxsize=256, blockysize=256,
                   compress="JPEG", photometric="YCBCR", jpeg_quality=92)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(path, "w", **profile) as dst:
        dst.write(rgb.transpose(2, 0, 1))
        dst.write_mask(valid.astype(np.uint8) * 255)
