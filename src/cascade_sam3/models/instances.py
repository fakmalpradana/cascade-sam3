from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Instances:
    """Instance per tile: masks (N,H,W) bool, scores (N,), sources list[str], refined (N,) bool."""

    masks: np.ndarray
    scores: np.ndarray
    sources: list = field(default_factory=list)
    refined: np.ndarray | None = None

    def __post_init__(self):
        self.masks = np.asarray(self.masks, dtype=bool)
        self.scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        if not self.sources:
            self.sources = ["det"] * len(self.scores)
        if self.refined is None:
            self.refined = np.zeros(len(self.scores), bool)

    @classmethod
    def empty(cls, h: int, w: int) -> Instances:
        return cls(np.zeros((0, h, w), bool), np.zeros(0, np.float32), [])

    def __len__(self) -> int:
        return len(self.scores)

    def select(self, idx) -> Instances:
        idx = np.asarray(idx, dtype=int).reshape(-1)
        return Instances(self.masks[idx], self.scores[idx], [self.sources[i] for i in idx], self.refined[idx])

    def boxes(self) -> np.ndarray:
        """xyxy (eksklusif di x2,y2) dari mask."""
        out = np.zeros((len(self), 4), np.float32)
        for i, m in enumerate(self.masks):
            rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
            if len(rows):
                out[i] = cols[0], rows[0], cols[-1] + 1, rows[-1] + 1
        return out

    @staticmethod
    def cat(a: Instances, b: Instances) -> Instances:
        return Instances(np.concatenate([a.masks, b.masks]), np.concatenate([a.scores, b.scores]),
                         a.sources + b.sources, np.concatenate([a.refined, b.refined]))
