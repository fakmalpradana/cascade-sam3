import pytest

from cascade_sam3.cli import build_parser
from cascade_sam3.config import load_config

CMDS = ["check-env", "show-config", "inspect", "convert-ecw", "catalog", "make-testsets", "build-dataset",
        "prepare", "train-detector", "train-sam3", "predict", "validate"]


@pytest.mark.parametrize("cmd", CMDS)
def test_parser_subcommands(cmd):
    assert build_parser().parse_args([cmd]).cmd == cmd


def test_predict_args():
    a = build_parser().parse_args(["--set", "infer.use_concept=false", "predict", "--mode", "detector",
                                   "--areas", "dki_01", "dki_02", "--zero-shot-sam3"])
    assert a.mode == "detector" and a.areas == ["dki_01", "dki_02"] and a.zero_shot_sam3


def test_set_override():
    cfg = load_config(sets=["infer.use_concept=false", "detector.resolution=432", "sam3.train_mode=lora"],
                      use_local=False)
    assert cfg["infer"]["use_concept"] is False
    assert cfg["detector"]["resolution"] == 432
    assert cfg["sam3"]["train_mode"] == "lora"
