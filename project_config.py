"""Shared project paths and runtime defaults."""

from pathlib import Path


SB3_DEVICE = "cpu"

DEFAULT_CHECKPOINT_FREQ = 1_000_000
DEFAULT_MAX_STEPS = 1000

CURRENT_BASELINE_MODEL = Path(
    "runs/default_baseline/checkpoints/default_baseline.zip"
)
