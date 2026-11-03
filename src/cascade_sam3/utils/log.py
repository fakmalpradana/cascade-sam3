from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def setup_logging(work_dir: str | Path, cmd: str) -> Path:
    logs = Path(work_dir) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"{cmd}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")],
        force=True,
    )
    return path
