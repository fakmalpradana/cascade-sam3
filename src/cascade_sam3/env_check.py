"""check-env: torch/CUDA, paket, GDAL QGIS + ECW, akses HF, path data, disk; --smoke menjalankan SAM 3 di GPU."""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

from .config import cities

MIN_VRAM_GB = 11
MIN_DISK_GB = 150


def _check(results, name, fn):
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"{type(e).__name__}: {e}"
    results.append((name, ok, detail))
    print(f"[{'OK' if ok else 'FAIL':4}] {name}: {detail}", flush=True)


def _torch():
    import torch

    if not torch.cuda.is_available():
        return False, f"torch {torch.__version__}, CUDA tidak tersedia"
    p = torch.cuda.get_device_properties(0)
    gb = p.total_memory / 1e9
    return gb >= MIN_VRAM_GB and torch.cuda.is_bf16_supported(), \
        f"torch {torch.__version__}, {p.name}, {gb:.1f} GB, bf16={torch.cuda.is_bf16_supported()}"


def _packages():
    vers, missing = [], []
    for m in ("rfdetr", "transformers", "peft", "supervision", "rasterio", "geopandas", "pyogrio", "cv2", "scipy"):
        try:
            vers.append(f"{m} {getattr(importlib.import_module(m), '__version__', '?')}")
        except Exception:  # noqa: BLE001
            missing.append(m)
    return not missing, ", ".join(vers) + (f" | HILANG: {missing}" if missing else "")


def _gdal(cfg):
    from .data import ecw

    b = ecw.find_gdal_bin(cfg)
    if b is None:
        return False, "GDAL QGIS tidak ditemukan (pasang QGIS LTR atau set data.gdal_bin)"
    return ecw.has_ecw_driver(cfg), f"{b} (driver ECW {'ada' if ecw.has_ecw_driver(cfg) else 'TIDAK ada'})"


def _hf(cfg):
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(cfg["sam3"]["model_id"], "config.json")
    return True, f"akses {cfg['sam3']['model_id']} OK ({Path(p).parent.name})"


def _paths(cfg):
    bad, n = [], 0
    for name, c in cities(cfg).items():
        for k in ("ortho", "labels"):
            for p in c[k] if isinstance(c[k], list) else [c[k]]:
                n += 1
                if not Path(p).exists():
                    bad.append(f"{name}.{k}={p}")
    if not n:
        return False, "tidak ada kota di config (salin configs/local.example.yaml ke configs/local.yaml)"
    return not bad, "semua path ada" if not bad else f"tidak ada: {bad}"


def _disk(cfg):
    wd = Path(cfg["work_dir"])
    anchor = next((p for p in [wd, *wd.parents] if p.exists()), Path.cwd())
    free = shutil.disk_usage(anchor).free / 1e9
    return free >= MIN_DISK_GB, f"{anchor}: {free:.0f} GB kosong (min {MIN_DISK_GB})"


def _smoke(cfg):
    import numpy as np
    import torch

    from .models.sam3 import BoxRefiner, load_tracker

    torch.cuda.reset_peak_memory_stats()
    model, proc = load_tracker(cfg["sam3"]["model_id"], cfg["sam3"]["amp_dtype"], "cuda")
    rgb = np.random.default_rng(0).integers(0, 255, (1008, 1008, 3), dtype=np.uint8)
    masks, iou = BoxRefiner(model, proc)(rgb, np.array([[100, 100, 400, 300], [500, 500, 900, 800]], np.float32))
    gb = torch.cuda.max_memory_allocated() / 1e9
    return masks.shape == (2, 1008, 1008), f"SAM 3 box prompt OK, iou_pred={np.round(iou, 3)}, VRAM puncak {gb:.2f} GB"


def check_env(cfg: dict, smoke: bool = False) -> int:
    r: list = []
    _check(r, "python", lambda: (sys.version_info[:2] >= (3, 11), sys.version.split()[0]))
    _check(r, "torch+CUDA", _torch)
    _check(r, "paket", _packages)
    _check(r, "GDAL QGIS + ECW", lambda: _gdal(cfg))
    _check(r, "Hugging Face", lambda: _hf(cfg))
    _check(r, "path data", lambda: _paths(cfg))
    _check(r, "disk", lambda: _disk(cfg))
    if smoke:
        _check(r, "smoke SAM 3", lambda: _smoke(cfg))
    fails = [n for n, ok, _ in r if not ok]
    print("SEMUA OK" if not fails else f"GAGAL: {fails}")
    return 1 if fails else 0
