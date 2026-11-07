"""cascade-sam3 [--config F] [--set k=v ...] <cmd>"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config

log = logging.getLogger("cascade_sam3")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cascade-sam3", description=__doc__)
    p.add_argument("--config", help="YAML lokal (default configs/local.yaml)")
    p.add_argument("--set", action="append", default=[], metavar="k=v", help="override, mis. infer.use_concept=false")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("check-env")
    s.add_argument("--smoke", action="store_true")
    s = sub.add_parser("show-config")
    s.add_argument("--key", help="dotted key, mis. work_dir")
    for name in ("inspect", "catalog", "make-testsets", "build-dataset", "prepare"):
        sub.add_parser(name).add_argument("--city")
    s = sub.add_parser("convert-ecw")
    s.add_argument("--city")
    s.add_argument("--full", action="store_true", help="konversi seluruh raster, bukan hanya area berlabel")
    s.add_argument("--force", action="store_true")
    for name in ("train-detector", "train-sam3"):
        s = sub.add_parser(name)
        s.add_argument("--name")
        s.add_argument("--resume", action="store_true")
        s.add_argument("--epochs", type=int)
    s = sub.add_parser("predict")
    s.add_argument("--mode", choices=["cascade", "detector", "concept"])
    s.add_argument("--detector", help="checkpoint RF-DETR (default terbaik di runs/detector)")
    s.add_argument("--sam3", help="checkpoint decoder (default runs/sam3/sam3_decoder_best.pt)")
    s.add_argument("--zero-shot-sam3", action="store_true")
    s.add_argument("--name")
    s.add_argument("--areas", nargs="+")
    s.add_argument("--raster")
    s.add_argument("--out")
    s = sub.add_parser("regularize", help="post-process sudut siku pada prediksi yang sudah ada")
    s.add_argument("--src", required=True, help="nama run sumber di predictions/")
    s.add_argument("--name", required=True, help="nama run hasil")
    s.add_argument("--steps", nargs="+", choices=["v2", "v3"], help="default infer.rightangle_chain")
    s = sub.add_parser("tune-infer", help="setel ambang inferensi pada split valid")
    s.add_argument("--mode", choices=["cascade", "detector"], default="cascade")
    s.add_argument("--detector")
    s.add_argument("--sam3")
    s.add_argument("--n-tiles", type=int, default=300)
    s.add_argument("--name")
    s = sub.add_parser("validate")
    s.add_argument("--name")
    s.add_argument("--pred-dir")
    s.add_argument("--areas", nargs="+")
    return p


def run(args, cfg: dict) -> int:
    c = args.cmd
    if c == "check-env":
        from .env_check import check_env

        return check_env(cfg, args.smoke)
    if c == "show-config":
        import yaml

        v = cfg
        for k in (args.key.split(".") if args.key else []):
            v = v[k]
        print(v if not isinstance(v, (dict, list)) else yaml.safe_dump(v, allow_unicode=True, sort_keys=False))
        return 0
    if c == "inspect":
        from .data.inspect_data import inspect

        inspect(cfg, args.city)
    elif c == "convert-ecw":
        from .data.ecw import convert_city

        convert_city(cfg, args.city, args.full, args.force)
    elif c in ("catalog", "make-testsets", "build-dataset", "prepare"):
        from .data.build import build_dataset
        from .data.catalog import build_catalog
        from .data.testsets import make_testsets

        if c in ("catalog", "prepare"):
            build_catalog(cfg, args.city)
        if c in ("make-testsets", "prepare"):
            make_testsets(cfg, args.city)
        if c in ("build-dataset", "prepare"):
            build_dataset(cfg, args.city)
    elif c == "train-detector":
        from .train.train_detector import train_detector

        train_detector(cfg, args.name, args.resume, args.epochs)
    elif c == "train-sam3":
        from .train.train_sam3 import train_sam3

        train_sam3(cfg, args.name, args.resume, args.epochs)
    elif c == "predict":
        from .infer.predict import predict_cli

        predict_cli(cfg, args)
    elif c == "regularize":
        from .infer.predict import regularize_predictions

        log.info("hasil di %s", regularize_predictions(cfg, args.src, args.name, args.steps))
    elif c == "tune-infer":
        from .eval.tune import tune_infer

        print(tune_infer(cfg, args.mode, args.detector, args.sam3, args.n_tiles, args.name))
    elif c == "validate":
        from pathlib import Path

        from .eval.validate import validate

        name = args.name or (Path(args.pred_dir).name if args.pred_dir else cfg["infer"]["mode"])
        validate(cfg, name, args.pred_dir, args.areas)
    return 0


def utf8_stdio() -> None:
    """Paksa stdout/stderr UTF-8. Di Windows output yang dialihkan ke file memakai cp1252; tabel `rich`
    dari rfdetr (karakter seperti '│') lalu memicu UnicodeEncodeError di akhir validasi dan training gagal."""
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")


def main(argv=None) -> int:
    utf8_stdio()
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.cmd not in ("show-config",):
        from .utils.log import setup_logging

        setup_logging(cfg["work_dir"], args.cmd)
    try:
        return run(args, cfg)
    except Exception:
        log.exception("%s gagal", args.cmd)
        return 1


if __name__ == "__main__":
    sys.exit(main())
