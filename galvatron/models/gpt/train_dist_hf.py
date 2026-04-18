"""Backward-compatible entry point for HuggingFace streaming training.

The implementation lives in ``train_dist.py``. This module re-executes it as
``__main__`` so existing launchers that invoke ``train_dist_hf.py`` keep working.

Preferred: call ``train_dist.py`` and set ``runtime.data.data_source=hf_streaming``
(``scripts/train_dist_hf.yaml`` already sets this).
"""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    train_dist = Path(__file__).resolve().parent / "train_dist.py"
    sys.argv[0] = str(train_dist)
    runpy.run_path(str(train_dist), run_name="__main__")
