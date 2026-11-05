import json

import cv2
import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("torchvision")


def _toy_coco(root, n=2, size=256):
    rng = np.random.default_rng(0)
    for split in ("train", "valid"):
        d = root / "dataset" / "coco" / split
        d.mkdir(parents=True)
        images, anns = [], []
        for i in range(n):
            img = np.full((size, size, 3), 60, np.uint8)
            for k in range(3):
                x, y = rng.integers(10, size - 60, 2)
                w, h = rng.integers(20, 50, 2)
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (220, 220, 220), -1)
                seg = [float(v) for v in (x, y, x + w, y, x + w, y + h, x, y + h)]
                anns.append(dict(id=len(anns) + 1, image_id=i + 1, category_id=1, segmentation=[seg],
                                 bbox=[float(x), float(y), float(w), float(h)], area=float(w * h), iscrowd=0))
            cv2.imwrite(str(d / f"{i}.jpg"), img)
            images.append(dict(id=i + 1, file_name=f"{i}.jpg", width=size, height=size))
        (d / "_annotations.coco.json").write_text(json.dumps(dict(images=images, annotations=anns,
                                                                  categories=[{"id": 1, "name": "building"}])))


@pytest.mark.parametrize("mode", ["decoder", "lora"])
def test_sam3_train_tiny(tmp_path, mode):
    from transformers import Sam3ImageProcessor, Sam3TrackerModel, Sam3TrackerProcessor

    from cascade_sam3.config import load_config
    from cascade_sam3.models.sam3 import load_decoder_weights, tiny_tracker_config
    from cascade_sam3.train.train_sam3 import train_sam3

    _toy_coco(tmp_path)
    cfg = load_config(overrides=dict(work_dir=str(tmp_path), sam3=dict(num_workers=0, train_mode=mode)),
                      use_local=False)
    model = Sam3TrackerModel(tiny_tracker_config())
    proc = Sam3TrackerProcessor(image_processor=Sam3ImageProcessor())
    best = train_sam3(cfg, epochs=1, model=model, processor=proc, device="cpu")
    assert best.exists() and (tmp_path / "runs" / "sam3" / "sam3_decoder_last.pt").exists()
    assert (tmp_path / "runs" / "sam3" / "sam3_decoder_ep1.pt").exists()
    assert not list((tmp_path / "runs" / "sam3").glob("*.tmp"))
    assert (tmp_path / "runs" / "sam3" / "train_log.csv").read_text().count("\n") == 3  # header + zero-shot + 1

    fresh = Sam3TrackerModel(tiny_tracker_config())
    ck = load_decoder_weights(fresh, best)
    assert ck["mode"] == mode and ck["epoch"] == 0
    sd = fresh.mask_decoder.state_dict()
    assert all((sd[k] == v).all() for k, v in ck["state_dict"].items())
