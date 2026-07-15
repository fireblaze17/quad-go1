"""Shared project paths and runtime defaults."""

from pathlib import Path


SB3_DEVICE = "cpu"

ORIGINAL_STAND_MODEL = Path("runs/stand/final_model.zip")
FIXED_BASELINE_DIR = Path("runs/stand_base_v2")
FIXED_BASELINE_MODEL = FIXED_BASELINE_DIR / "final_model.zip"

DEFAULT_CHECKPOINT_FREQ = 25_000
DEFAULT_ACTION_FILTER_TAU = 0.05

FRICTION_A_DIR = Path("runs/stand_friction_a_07_09")
FRICTION_AB_DIR = Path("runs/stand_friction_ab_065_095")
FILTERED_FIXED_DIR = Path("runs/stand_action_filter_tau005_from_jitter5k_5k")
FRICTION_050_120_DIR = Path("runs/stand_friction_random_060_090_tau005_from_filtered2k")

FRICTION_A_MODEL = FRICTION_A_DIR / "final_model.zip"
FRICTION_B_CAPABLE_MODEL = FRICTION_AB_DIR / "final_model.zip"
FILTERED_FIXED_MODEL = FILTERED_FIXED_DIR / "checkpoints" / "stand_policy_2000_steps.zip"
FRICTION_050_120_MODEL = FRICTION_050_120_DIR / "checkpoints" / "stand_policy_10000_steps.zip"
CURRENT_BASELINE_MODEL = FRICTION_050_120_MODEL

DEFAULT_VIEWER_FRICTION_RANGE = (0.5, 1.2)
