"""Shared project paths and runtime defaults."""

from pathlib import Path


SB3_DEVICE = "cpu"

ORIGINAL_STAND_MODEL = Path("runs/stand/final_model.zip")
FIXED_BASELINE_DIR = Path("runs/stand_base_v2")
FIXED_BASELINE_MODEL = FIXED_BASELINE_DIR / "final_model.zip"

DEFAULT_CHECKPOINT_FREQ = 25_000

FRICTION_A_DIR = Path("runs/stand_friction_a_07_09")
FRICTION_AB_DIR = Path("runs/stand_friction_ab_065_095")
FRICTION_B_DIR = Path("runs/stand_friction_b_06_10")
FRICTION_C_DIR = Path("runs/stand_friction_c_05_11")

FRICTION_A_MODEL = FRICTION_A_DIR / "final_model.zip"
FRICTION_B_CAPABLE_MODEL = FRICTION_AB_DIR / "final_model.zip"
CURRENT_BASELINE_MODEL = FRICTION_B_CAPABLE_MODEL

DEFAULT_VIEWER_FRICTION_RANGE = (0.6, 1.0)
