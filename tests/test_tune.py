import numpy as np

from cascade_sam3.eval.tune import cache_tile, predict_cached, score, sweep
from cascade_sam3.models.fusion import refine, smooth_masks
from cascade_sam3.models.instances import Instances

IC = dict(det_threshold=0.2, use_concept=True, use_refine=True, concept_score_weight=0.8, match_iou=0.3,
          concept_add_threshold=0.6, refine_min_iou_pred=0.7, refine_consistency_iou=0.5, refine_smooth_px=0,
          nms_iou=0.5, nms_containment=0.8)


def box(y0, y1, x0, x1, size=64):
    m = np.zeros((size, size), bool)
    m[y0:y1, x0:x1] = True
    return m


def tile():
    gt = np.stack([box(4, 24, 4, 24), box(40, 60, 40, 60)])
    det = Instances(np.stack([box(4, 24, 4, 26), box(40, 60, 4, 24)]), [0.9, 0.3])  # kedua = FP
    con = Instances(np.stack([box(40, 60, 40, 60)]), [0.8], ["concept"])
    new = np.stack([box(4, 24, 4, 24), box(40, 60, 4, 24), box(40, 60, 40, 60)])  # refine menepatkan yang pertama
    return cache_tile(gt, det, con, new, [0.95, 0.9, 0.9])


def test_score_detector_mode_uses_only_det_threshold():
    t = tile()
    both = score([t], {**IC, "det_threshold": 0.2}, "detector", 0)
    high = score([t], {**IC, "det_threshold": 0.5}, "detector", 0)
    assert high["obj_precision"] > both["obj_precision"]
    assert high["pixel_iou"] > both["pixel_iou"]


def test_refinement_and_concept_change_cascade_result():
    t = tile()
    inst = predict_cached(t, {**IC, "det_threshold": 0.5}, "cascade", 0)
    assert len(inst) == 2 and "concept" in inst.sources and inst.refined.any()
    assert score([t], {**IC, "det_threshold": 0.5}, "cascade", 0)["pixel_iou"] == 1.0


def test_sweep_picks_best_values_and_reports_table():
    res = sweep([tile()], IC, "cascade", 0, grid={"det_threshold": [0.2, 0.5], "concept_add_threshold": [0.6, 1.01]})
    assert res["params"] == {"det_threshold": 0.5, "concept_add_threshold": 0.6}
    assert res["best"]["pixel_iou"] == 1.0 and res["start"]["pixel_iou"] < 1.0
    assert len(res["table"]) == 4


def test_sweep_detector_mode_only_tunes_threshold():
    assert list(sweep([tile()], IC, "detector", 0)["params"]) == ["det_threshold"]


def test_smooth_masks_removes_spur_and_zero_is_noop():
    m = box(10, 40, 10, 40)
    m[20:22, 40:48] = True  # gerigi
    out = smooth_masks(m[None], 7)[0]
    assert not out[20, 46] and out[25, 25]
    assert smooth_masks(m[None], 0) is not None and (smooth_masks(m[None], 0)[0] == m).all()


def test_refine_applies_smoothing_only_when_set():
    m = box(10, 40, 10, 40)
    spur = m.copy()
    spur[20:22, 40:48] = True
    inst = Instances(m[None], [0.9])

    def refiner(rgb, boxes):
        return spur[None], np.array([0.9])

    rgb = np.zeros((64, 64, 3), np.uint8)
    ic = {"refine_min_iou_pred": 0.5, "refine_consistency_iou": 0.5}
    assert refine(inst, rgb, refiner, ic).masks[0, 20, 46]
    assert not refine(inst, rgb, refiner, {**ic, "refine_smooth_px": 7}).masks[0, 20, 46]
