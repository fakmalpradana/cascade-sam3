import json

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from cascade_sam3.data.build import build_dataset
from cascade_sam3.data.catalog import build_catalog
from cascade_sam3.data.testsets import make_testsets
from cascade_sam3.eval.validate import validate
from cascade_sam3.infer.predict import predict_areas

from .conftest import BrightPredictor, make_cfg


def test_pipeline_end_to_end(synthetic_city, tmp_path):
    cfg = make_cfg(synthetic_city, tmp_path / "work")
    w = tmp_path / "work"
    cat = build_catalog(cfg)["syn"]
    assert cat["n_labels"] == len(synthetic_city["roofs"])  # label_filter membuang JALAN

    make_testsets(cfg)
    areas = gpd.read_file(w / "testsets" / "test_areas.gpkg", layer="syn")
    assert len(areas) == 3
    assert set(areas.density_class) == {"rendah", "sedang", "tinggi"}
    for n in areas.name:
        for f in ("image.tif", "gt.gpkg", "area.gpkg", "quicklook.png", "meta.json"):
            assert (w / "testsets" / n / f).exists()

    stats = build_dataset(cfg)["syn"]
    assert stats["valid"]["tiles"] > 0 and stats["train"]["tiles"] > 0
    for s in ("train", "valid", "test"):
        coco = json.loads((w / "dataset" / "coco" / s / "_annotations.coco.json").read_text())
        assert coco["categories"] == [{"id": 1, "name": "building", "supercategory": "none"}]
        ids = {im["id"] for im in coco["images"]}
        for a in coco["annotations"]:
            assert a["image_id"] in ids and len(a["bbox"]) == 4 and a["segmentation"]
            assert all(len(seg) >= 6 for seg in a["segmentation"])

    # tidak ada tile train/valid yang beririsan zona eksklusi
    t = cfg["testsets"]
    excl = areas.union_all().buffer(t["exclusion_buffer_m"] + t["context_margin_m"])
    tiles = pd.read_csv(w / "dataset" / "tiles.csv")
    tv = tiles[tiles.split.isin(["train", "valid"])]
    assert len(tv)
    assert not any(box(r.minx, r.miny, r.maxx, r.maxy).intersects(excl) for r in tv.itertuples())

    predict_areas(cfg, "fake", BrightPredictor())
    s = validate(cfg, "fake")
    assert s["overall"]["obj@0.5"]["f1"] > 0.9
    assert s["overall"]["pixel"]["f1"] > 0.9
    assert (w / "reports" / "validation_fake.md").exists()
    assert (w / "reports" / "errors_fake").is_dir()
