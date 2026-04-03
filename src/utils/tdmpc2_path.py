"""
src/utils/tdmpc2_path.py
========================
Adds the third_party/tdmpc2 checkout to sys.path so that:
  - `from tdmpc2 import TDMPC2` resolves (the package dir)
  - `from common import ...`    resolves (tdmpc2's internal imports)

Call setup_tdmpc2_path() once at the top of any script that imports tdmpc2.
"""

from __future__ import annotations

import sys
from pathlib import Path


def setup_tdmpc2_path() -> None:
    """Prepend third_party/tdmpc2 paths onto sys.path if not already present."""
    root = Path(__file__).resolve().parents[2]          # project root
    tp   = root / "third_party" / "tdmpc2"
    pkg  = tp / "tdmpc2"                                # contains TDMPC2 class

    for p in (tp, pkg):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
