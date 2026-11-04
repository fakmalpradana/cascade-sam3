import pytest
import torch

from cascade_sam3.train import rfdetr_patches

pytest.importorskip("rfdetr")  # CI tanpa extra `train`


def test_chunked_sampler_matches_original():
    from rfdetr.models import criterion

    orig = criterion._sample_target_masks_at_points
    g = torch.Generator().manual_seed(0)
    targets, indices, n_tot = [], [], 0
    for n_gt, n_match in ((7, 40), (3, 25)):
        targets.append({"masks": torch.rand(n_gt, 32, 32, generator=g) > 0.5})
        tgt = torch.randint(0, n_gt, (n_match,), generator=g)
        indices.append((torch.arange(n_match), tgt))
        n_tot += n_match
    coords = torch.rand(n_tot, 50, 2, generator=g)
    ref = orig(targets, indices, coords)
    got = rfdetr_patches.chunked_sampler(orig, chunk=8)(targets, indices, coords)
    assert torch.equal(ref, got)


def test_apply_is_idempotent():
    from rfdetr.models import criterion

    before = criterion._sample_target_masks_at_points
    try:
        assert rfdetr_patches.apply()
        once = criterion._sample_target_masks_at_points
        assert rfdetr_patches.apply()
        assert criterion._sample_target_masks_at_points is once
        assert once._cascade_chunked
    finally:
        criterion._sample_target_masks_at_points = before
