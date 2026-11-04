"""Tambalan memori untuk rfdetr 1.11.x saat training segmentasi.

`criterion._sample_target_masks_at_points` mengambil semua mask GT yang di-match sekaligus lalu `.float()`.
Dengan group DETR (13 grup query) tile padat (≈170 bangunan) menjadi ≈2.200 mask float ≈ 2,4 GB per
panggilan → OOM di GPU 12 GB. Tambalan ini memanggil fungsi asli per potongan indeks (hasil identik, baris
per baris independen) sehingga puncak memori dibatasi `chunk` mask.
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)

_CHUNK = 256


def chunked_sampler(orig, chunk: int = _CHUNK):
    def sample(targets, indices, point_coords):
        if sum(t.numel() for _, t in indices) <= chunk:
            return orig(targets, indices, point_coords)
        out, offset = [], 0
        for target, (src, tgt) in zip(targets, indices):
            n = tgt.numel()
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                out.append(orig([target], [(src[s:e], tgt[s:e])], point_coords[offset + s: offset + e]))
            offset += n
        return torch.cat(out, dim=0)

    sample._cascade_chunked = True
    return sample


def apply() -> bool:
    from rfdetr.models import criterion

    fn = getattr(criterion, "_sample_target_masks_at_points", None)
    if fn is None:
        log.warning("rfdetr: _sample_target_masks_at_points tidak ada; tambalan memori dilewati")
        return False
    if not getattr(fn, "_cascade_chunked", False):
        criterion._sample_target_masks_at_points = chunked_sampler(fn)
    return True
