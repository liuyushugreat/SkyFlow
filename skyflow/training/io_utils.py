"""Small I/O helpers shared by the trainers."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_with_retry(obj, path: str | Path, attempts: int = 5, delay_s: float = 3.0) -> None:
    """torch.save that survives transient file locks (sync clients / AV
    scanners on Windows raise WinError 32 for a few seconds)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    last = None
    for i in range(attempts):
        try:
            torch.save(obj, tmp)
            tmp.replace(path)
            return
        except (OSError, PermissionError) as e:  # noqa: PERF203
            last = e
            logger.warning(f"save {path.name} failed ({e}); retry {i + 1}/{attempts} in {delay_s:.0f}s")
            time.sleep(delay_s)
    raise last
