"""
src/utils/logger.py
====================
Thin TensorBoard + console logger for LatentIntercept training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except ImportError:  # pragma: no cover
    _TB = False


class TrainingLogger:
    """Write scalar metrics to TensorBoard and/or stdout."""

    def __init__(self, log_dir: Union[str, Path]) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self._log_dir)) if _TB else None
        if not _TB:
            print("[Logger] TensorBoard not available — stdout only.")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
