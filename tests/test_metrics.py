import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

from cascade_sam3.eval import metrics as M


def test_pixel_known_case():
    area = box(0, 0, 10, 10)
    c = M.pixel_confusion([box(0, 0, 5, 10)], [box(0, 0, 10, 5)], area, from_origin(0, 10, 1, 1), (10, 10))
    assert c == dict(tp=25, fp=25, fn=25, tn=25)
    s = M.pixel_scores(**c)
    assert s["precision"] == s["recall"] == s["f1"] == 0.5
    assert s["iou"] == pytest.approx(1 / 3)
    assert s["kappa"] == pytest.approx(0.0)


def test_hungarian_duplicate_is_fp():
    gt = [box(0, 0, 10, 10), box(20, 0, 30, 10)]
    pred = [box(0, 0, 10, 10), box(0.5, 0, 10.5, 10), box(20, 0, 30, 10)]
    c = M.object_counts(M.iou_matrix(gt, pred), 0.5)
    assert (c["tp"], c["fp"], c["fn"]) == (2, 1, 0)


def test_ap_order():
    gt = [box(0, 0, 10, 10), box(20, 0, 30, 10)]
    pred = [box(0, 0, 10, 10), box(20, 0, 30, 10), box(50, 50, 60, 60)]
    m = M.iou_matrix(gt, pred)
    assert M.average_precision(m, [0.9, 0.8, 0.1], 0.5) == pytest.approx(1.0)
    assert M.average_precision(m, [0.1, 0.2, 0.9], 0.5) < 1.0


def test_boundary_f1():
    g = [box(0, 0, 10, 10)]
    assert M.boundary_f1(g, g, 0.5)["f1"] == pytest.approx(1.0)
    assert M.boundary_f1(g, [box(100, 100, 110, 110)], 0.5)["f1"] == 0.0
    assert M.boundary_f1(g, [], 0.5)["f1"] == 0.0
