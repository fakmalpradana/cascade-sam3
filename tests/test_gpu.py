import pytest
import torch

from cascade_sam3.utils.gpu import limit_vram


def test_limit_vram_off_by_default(monkeypatch):
    called = []
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f: called.append(f))
    assert limit_vram(None) is False
    assert called == []


def test_limit_vram_sets_fraction(monkeypatch):
    called = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f: called.append(f))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: type("P", (), {"total_memory": 12 * 2**30}))
    assert limit_vram(0.7) is True
    assert called == [0.7]


def test_limit_vram_rejects_bad_fraction():
    with pytest.raises(ValueError):
        limit_vram(1.5)
