import geopandas as gpd
import pytest
from shapely.geometry import box

from cascade_sam3.infer.predict import predict_raster

from .conftest import GSD, BrightPredictor, make_cfg, paint, write_tif


@pytest.mark.parametrize("bld", [
    (5, 5, 45, 15),     # 2-3 tile horizontal
    (10, 10, 40, 40),   # 4 tile (2x2)
])
def test_building_across_tiles_is_one_polygon(tmp_path, bld):
    x0, y0 = 500000.0, 9000000.0
    b = box(x0 + bld[0], y0 - bld[3], x0 + bld[2], y0 - bld[1])
    write_tif(tmp_path / "r.tif", paint([b], n=500, x0=x0, y0=y0), x0=x0, y0=y0)
    cfg = make_cfg(dict(ortho="", labels=""), tmp_path / "work")
    gdf = predict_raster(tmp_path / "r.tif", tmp_path / "out.gpkg", BrightPredictor(), cfg)
    assert len(gdf) == 1
    assert gdf.geometry.iloc[0].intersection(b).area / b.union(gdf.geometry.iloc[0]).area > 0.95
    assert len(gpd.read_file(tmp_path / "out.gpkg", layer="buildings")) == 1
    assert GSD == 0.1
