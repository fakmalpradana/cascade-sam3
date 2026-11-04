import json

from cascade_sam3.data.subset import make_train_subset, sample_images


def _coco(names):
    return {"images": [{"id": i, "file_name": f} for i, f in enumerate(names)],
            "annotations": [{"id": i, "image_id": i} for i in range(len(names))], "categories": []}


def test_sample_images_proportional_per_city():
    ims = [{"id": i, "file_name": f"dki_{i}.jpg"} for i in range(300)] + \
          [{"id": 300 + i, "file_name": f"bontang_{i}.jpg"} for i in range(100)]
    got = sample_images(ims, 200, seed=0)
    assert len(got) == 200
    assert sum(g["file_name"].startswith("bontang") for g in got) == 50
    assert [g["id"] for g in got] == sorted(g["id"] for g in got)
    assert sample_images(ims, 1000, seed=0) == ims


def test_make_train_subset_keeps_valid_and_points_to_originals(tmp_path):
    coco = tmp_path / "dataset" / "coco"
    for split, n in (("train", 10), ("valid", 4)):
        (coco / split).mkdir(parents=True)
        (coco / split / "_annotations.coco.json").write_text(
            json.dumps(_coco([f"dki_{split}_{i}.jpg" for i in range(n)])), encoding="utf-8")
        for i in range(n):
            (coco / split / f"dki_{split}_{i}.jpg").write_bytes(b"x")
    out = make_train_subset(coco, 3)
    tr = json.loads((out / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    va = json.loads((out / "valid" / "_annotations.coco.json").read_text(encoding="utf-8"))
    assert len(tr["images"]) == 3 and len(tr["annotations"]) == 3
    assert len(va["images"]) == 4
    for split, d in (("train", tr), ("valid", va)):
        for im in d["images"]:
            assert (out / split / im["file_name"]).resolve().exists()
