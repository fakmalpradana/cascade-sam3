import numpy as np

from cascade_sam3.config import load_config
from cascade_sam3.models.fusion import fuse, mask_nms, refine, resolve_overlaps
from cascade_sam3.models.instances import Instances

IC = load_config(use_local=False)["infer"]


def rect(y0, y1, x0, x1, h=64, w=64):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_fuse_match_and_add():
    det = Instances(np.stack([rect(0, 20, 0, 20)]), [0.5])
    con = Instances(np.stack([rect(1, 20, 0, 20), rect(40, 60, 40, 60), rect(40, 50, 0, 10)]), [0.9, 0.7, 0.5],
                    ["concept"] * 3)
    out = fuse(det, con, IC)
    assert len(out) == 2  # det cocok + 1 konsep baru (skor 0.5 < 0.6 ditolak)
    assert out.sources == ["det+concept", "concept"]
    assert np.isclose(out.scores[0], 0.5 + 0.5 * 0.8 * 0.9)
    assert np.isclose(out.scores[1], 0.7 * 0.8)


def test_refine_rejected_on_low_iou_pred():
    inst = Instances(np.stack([rect(10, 30, 10, 30), rect(40, 60, 40, 60)]), [0.9, 0.9])
    new = np.stack([rect(10, 31, 10, 31), rect(40, 61, 40, 61)])

    def refiner(rgb, boxes):
        return new, np.array([0.9, 0.5])

    out = refine(inst, np.zeros((64, 64, 3), np.uint8), refiner, IC)
    assert out.refined.tolist() == [True, False]
    assert out.masks[0].sum() == 21 * 21 and out.masks[1].sum() == 20 * 20


def test_refine_rejected_on_inconsistent_mask():
    inst = Instances(np.stack([rect(10, 30, 10, 30)]), [0.9])
    out = refine(inst, np.zeros((64, 64, 3), np.uint8), lambda r, b: (np.stack([rect(40, 60, 40, 60)]), [0.99]), IC)
    assert not out.refined[0]


def test_nms_iou_and_containment():
    inst = Instances(np.stack([rect(0, 40, 0, 40), rect(0, 40, 1, 40), rect(5, 15, 5, 15), rect(50, 60, 50, 60)]),
                     [0.9, 0.8, 0.95, 0.7])
    out = mask_nms(inst, 0.5, 0.8)
    # kotak kecil (skor tertinggi) menekan kotak besar via containment
    assert np.allclose(sorted(out.scores), [0.7, 0.95])


def test_resolve_overlaps():
    inst = Instances(np.stack([rect(0, 20, 0, 20), rect(10, 30, 10, 30)]), [0.5, 0.9])
    out = resolve_overlaps(inst)
    assert not (out.masks[0] & out.masks[1]).any()
    hi = out.masks[np.argmax(out.scores)]
    assert hi.sum() == 400
