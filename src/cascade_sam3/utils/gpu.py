"""Batas VRAM per proses."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def limit_vram(fraction: float | None) -> bool:
    """Batasi cache allocator PyTorch ke `fraction` dari VRAM.

    Di Windows (WDDM) VRAM penuh tidak memicu OOM tetapi meluber ke RAM sistem ("sysmem fallback"),
    sehingga training merangkak. Dengan batas ini allocator membuang cache-nya sebelum melewati batas.
    """
    if not fraction:
        return False
    if not 0 < fraction <= 1:
        raise ValueError(f"max_vram_fraction {fraction} harus di (0, 1]")
    import torch

    if not torch.cuda.is_available():
        return False
    torch.cuda.set_per_process_memory_fraction(fraction)
    log.info("VRAM dibatasi %.0f%% (%.1f GB)", fraction * 100,
             fraction * torch.cuda.get_device_properties(0).total_memory / 2**30)
    return True
