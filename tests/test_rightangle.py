import math

import geopandas as gpd
import numpy as np
import pytest
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box

from cascade_sam3.infer.predict import regularize_predictions
from cascade_sam3.post.rightangle import regularize_chain, regularize_gdf, regularize_v2, regularize_v3

from .conftest import make_cfg


def jagged_rect(w=12.0, h=8.0, step=0.4, amp=0.15, seed=0):
    """Persegi panjang dengan tepi bergerigi seperti hasil poligonisasi mask piksel."""
    rng = np.random.default_rng(seed)
    pts = []
    for x in np.arange(0, w, step):
        pts.append((x, rng.uniform(-amp, amp)))
    for y in np.arange(0, h, step):
        pts.append((w + rng.uniform(-amp, amp), y))
    for x in np.arange(w, 0, -step):
        pts.append((x, h + rng.uniform(-amp, amp)))
    for y in np.arange(h, 0, -step):
        pts.append((rng.uniform(-amp, amp), y))
    return Polygon(pts)


def corner_angles(poly):
    c = np.asarray(poly.exterior.coords)[:-1]
    out = []
    for i in range(len(c)):
        a, b, d = c[i - 1], c[i], c[(i + 1) % len(c)]
        v1, v2 = a - b, d - b
        out.append(math.degrees(math.acos(np.clip(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1))))
    return out


@pytest.mark.parametrize("angle", [0, 23, 67])
def test_chain_turns_jagged_footprint_into_right_angles(angle):
    src = rotate(jagged_rect(), angle, origin="centroid")
    out = regularize_chain(src, ("v2", "v3", "v3"))
    assert out.is_valid and out.geom_type == "Polygon"
    assert len(out.exterior.coords) - 1 < len(src.exterior.coords) - 1
    assert all(abs(a - 90) < 1.0 for a in corner_angles(out))
    assert out.intersection(src).area / out.union(src).area > 0.95
    assert abs(out.area - src.area) / src.area < 0.05


def test_single_steps_keep_clean_rectangle():
    r = rotate(box(0, 0, 10, 6), 30, origin="centroid")
    for f in (regularize_v2, regularize_v3):
        out = f(r)
        assert out.intersection(r).area / out.union(r).area > 0.99


def test_chain_rejects_unknown_step_and_passes_empty():
    with pytest.raises(ValueError):
        regularize_chain(box(0, 0, 1, 1), ("v9",))
    assert regularize_chain(Polygon(), ("v2",)).is_empty


def test_gdf_in_geographic_crs_is_processed_in_metres():
    geom = translate(rotate(jagged_rect(), 15, origin="centroid"), 706900, 9320100)  # DKI, UTM 48S
    g = gpd.GeoDataFrame(geometry=[geom], crs=32748).to_crs(4326)
    out = regularize_gdf(g, ["v2", "v3", "v3"])
    assert out.crs.to_epsg() == 4326 and len(out) == 1
    assert all(abs(a - 90) < 1.0 for a in corner_angles(out.to_crs(32748).geometry.iloc[0]))


def test_regularize_command_writes_new_run_and_keeps_source(tmp_path):
    cfg = make_cfg(dict(ortho="", labels=""), tmp_path / "work")
    src = tmp_path / "work" / "predictions" / "cascade"
    src.mkdir(parents=True)
    geoms = [jagged_rect(seed=s).buffer(0) for s in range(3)]
    g = gpd.GeoDataFrame(dict(score=[0.9, 0.8, 0.7], area_m2=[x.area for x in geoms]), geometry=geoms, crs=32748)
    g.to_file(src / "a.gpkg", layer="buildings", driver="GPKG")
    out = regularize_predictions(cfg, "cascade", "cascade_reg", ["v2", "v3", "v3"])
    r = gpd.read_file(out / "a.gpkg", layer="buildings")
    assert len(r) == 3 and list(r.score) == [0.9, 0.8, 0.7]
    assert np.allclose(r.area_m2, r.area)
    assert (out / "a.json").exists()
    assert gpd.read_file(src / "a.gpkg").geometry.iloc[0].equals(geoms[0])
    with pytest.raises(SystemExit):
        regularize_predictions(cfg, "cascade", "cascade", ["v2"])
