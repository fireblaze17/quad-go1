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
V3_RELATIVE_FIXED08_DIR = Path("runs/stand_v3_relative_obs_fixed08_500k")
V3P1_RELATIVE_FIXED08_DIR = Path("runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k")

FRICTION_A_MODEL = FRICTION_A_DIR / "final_model.zip"
FRICTION_B_CAPABLE_MODEL = FRICTION_AB_DIR / "final_model.zip"
FILTERED_FIXED_MODEL = FILTERED_FIXED_DIR / "checkpoints" / "stand_policy_2000_steps.zip"
FRICTION_050_120_MODEL = FRICTION_050_120_DIR / "checkpoints" / "stand_policy_10000_steps.zip"
LAST_ACCEPTED_V2_MODEL = FRICTION_050_120_MODEL
V3_RELATIVE_FIXED08_ATTEMPT_MODEL = V3_RELATIVE_FIXED08_DIR / "checkpoints" / "stand_policy_25000_steps.zip"
V3P1_RELATIVE_FIXED08_MODEL = V3P1_RELATIVE_FIXED08_DIR / "checkpoints" / "stand_policy_5000_steps.zip"
V3P1_FRICTION_ROBUST_MODEL = V3P1_RELATIVE_FIXED08_MODEL

# Active code now uses the v3.1 65D relative-state observation. This checkpoint
# is accepted for fixed standing, coordinate invariance, and effective friction
# 0.5-1.2. V2 remains the last accepted reset-noise robust result, but its
# checkpoints are shape-incompatible with this worktree.
CURRENT_BASELINE_MODEL = V3P1_FRICTION_ROBUST_MODEL

DEFAULT_VIEWER_FRICTION_RANGE = (0.8, 0.8)
