import pytest

from cascade_sam3.train.train_detector import resume_checkpoint


def test_resume_picks_latest_full_checkpoint(tmp_path):
    # rfdetr 1.11.2: state lengkap di last.ckpt / checkpoint_<epoch>.ckpt; *.pth hanya bobot
    for f in ("checkpoint_best_total.pth", "last_ema.pth", "checkpoint_4.ckpt", "checkpoint_14.ckpt",
              "checkpoint_9.ckpt"):
        (tmp_path / f).write_bytes(b"")
    assert resume_checkpoint(tmp_path).name == "checkpoint_14.ckpt"
    (tmp_path / "last.ckpt").write_bytes(b"")
    assert resume_checkpoint(tmp_path).name == "last.ckpt"


def test_resume_without_checkpoint_raises(tmp_path):
    (tmp_path / "checkpoint_best_total.pth").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        resume_checkpoint(tmp_path)


def _train_with_fake_model(tmp_path, monkeypatch, **detector):
    pytest.importorskip("torch")
    from cascade_sam3.config import load_config
    from cascade_sam3.train import rfdetr_patches, train_detector

    (tmp_path / "dataset" / "coco" / "train").mkdir(parents=True)
    (tmp_path / "dataset" / "coco" / "train" / "_annotations.coco.json").write_text("{}")
    calls = {}

    class Fake:
        def __init__(self, **kw):
            calls["init"] = kw

        def train(self, **kw):
            calls["train"] = kw

    monkeypatch.setattr(train_detector, "detector_class", lambda variant: Fake)
    monkeypatch.setattr(rfdetr_patches, "apply", lambda: True)
    cfg = load_config(overrides=dict(work_dir=str(tmp_path), detector=detector), use_local=False)
    train_detector.train_detector(cfg, name="x", epochs=1)
    return calls


def test_default_uses_pretrained_weights(tmp_path, monkeypatch):
    assert _train_with_fake_model(tmp_path, monkeypatch)["init"] == {}


def test_init_weights_builds_model_at_target_resolution(tmp_path, monkeypatch):
    calls = _train_with_fake_model(tmp_path, monkeypatch, init_weights="w.pth", resolution=672)
    assert calls["init"] == dict(pretrain_weights="w.pth", resolution=672)
    assert calls["train"]["resolution"] == 672
