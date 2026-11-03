"""Kota sintetis: GeoTIFF RGB EPSG:32748 3000x3000 px @0,1 m, atap terang berotasi, label 9 SHP grid 100 m."""

from __future__ import annotations

import cv2
import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely import affinity
from shapely.geometry import box

from cascade_sam3.config import load_config
from cascade_sam3.models.instances import Instances

X0, Y0, GSD, N = 700000.0, 9300000.0, 0.1, 3000
CRS = "EPSG:32748"


def roof_polygons(seed=0):
    rng = np.random.default_rng(seed)
    roofs, roads = [], []
    for cx in np.arange(10, 300, 20):
        for cy in np.arange(10, 300, 20):
            if rng.random() > 0.25 + 0.7 * cx / 300:  # kepadatan naik ke timur
                roads.append(box(X0 + cx - 6, Y0 - cy - 1, X0 + cx + 6, Y0 - cy + 1))
                continue
            w, h = rng.uniform(6, 12), rng.uniform(5, 8)
            jx, jy = rng.uniform(-1, 1, 2)
            r = box(-w / 2, -h / 2, w / 2, h / 2)
            r = affinity.rotate(r, rng.uniform(0, 90))
            roofs.append(affinity.translate(r, X0 + cx + jx, Y0 - cy + jy))
    return roofs, roads


def paint(roofs, n=N, x0=X0, y0=Y0, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 80, (n, n, 3), dtype=np.uint8)
    for r in roofs:
        c = np.asarray(r.exterior.coords)
        px = np.round(np.c_[(c[:, 0] - x0) / GSD, (y0 - c[:, 1]) / GSD]).astype(np.int32)
        cv2.fillPoly(img, [px], tuple(int(v) for v in rng.integers(200, 240, 3)))
    return img


def write_tif(path, img, x0=X0, y0=Y0):
    h, w = img.shape[:2]
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint8", crs=CRS,
                       transform=from_origin(x0, y0, GSD, GSD), tiled=True, blockxsize=256, blockysize=256) as ds:
        ds.write(img.transpose(2, 0, 1))


@pytest.fixture(scope="session")
def synthetic_city(tmp_path_factory):
    root = tmp_path_factory.mktemp("city")
    roofs, roads = roof_polygons()
    write_tif(root / "ortho.tif", paint(roofs))
    lab = root / "labels"
    lab.mkdir()
    gdf = gpd.GeoDataFrame({"KETERANGAN": ["BANGUNAN"] * len(roofs) + ["JALAN"] * len(roads)},
                           geometry=roofs + roads, crs=CRS)
    c = gdf.geometry.centroid
    gi, gj = ((c.x - X0) // 100).astype(int), ((Y0 - c.y) // 100).astype(int)
    for i in range(3):
        for j in range(3):
            sub = gdf[(gi == i) & (gj == j)]
            sub.to_file(lab / f"grid_{i}{j}.shp")
    return dict(root=root, ortho=root / "ortho.tif", labels=lab, roofs=roofs)


def make_cfg(city, work_dir, **extra):
    over = dict(
        work_dir=str(work_dir),
        cities={"syn": dict(ortho=str(city["ortho"]), labels=str(city["labels"]), coverage_mode="file_bbox",
                            label_filter={"column": "KETERANGAN", "values": ["BANGUNAN"]})},
        data=dict(tile_size=256, max_tiles_per_city=2000, keep_empty_tile_ratio=1.0, split_block_m=50,
                  val_fraction=0.3, image_format="jpg"),
        testsets=dict(area_size_m=60, min_buildings=3, min_separation_m=80, exclusion_buffer_m=5, context_margin_m=4),
        infer=dict(tile_size=256, overlap_px=64),
    )
    for k, v in extra.items():
        over.setdefault(k, {}).update(v)
    return load_config(overrides=over, use_local=False)


class BrightPredictor:
    """Prediktor palsu: threshold kecerahan + connected components."""

    def __call__(self, rgb, valid, gsd):
        m = ((rgb.mean(2) > 150) & valid).astype(np.uint8)
        n, lab = cv2.connectedComponents(m, connectivity=4)
        h, w = m.shape
        if n <= 1:
            return Instances.empty(h, w)
        masks = np.stack([lab == k for k in range(1, n)])
        keep = masks.sum((1, 2)) >= 20
        return Instances(masks[keep], np.full(keep.sum(), 0.9, np.float32), ["det"] * int(keep.sum()))
