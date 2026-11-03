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
