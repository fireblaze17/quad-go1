"""Shared project paths and runtime defaults."""

from pathlib import Path


SB3_DEVICE = "cpu"

DEFAULT_CHECKPOINT_FREQ = 25_000
DEFAULT_MAX_STEPS = 1000

# Historical references only. They are useful for experiment history, but their
# observation/action assumptions are not the active training target.
LAST_ACCEPTED_V2_MODEL = Path(
    "runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip"
)
V3P1_HISTORICAL_OBS65_MODEL = Path(
    "runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip"
)

V3P1_IMPLICIT_LIMITED_DRIVE_DIR = Path("runs/stand_v4_implicit_limited_drive_reward_aligned_1m")
V3P1_IMPLICIT_LIMITED_DRIVE_MODEL = V3P1_IMPLICIT_LIMITED_DRIVE_DIR / "best_model.zip"

# Active code uses the v3.1 45D relative-state observation with Chrono
# driveline-based implicit limited drives, 50 Hz control, and no action filter.
# Older position-motor, 65D, and raw torque-PD checkpoints remain historical
# references and need retraining here.
CURRENT_BASELINE_MODEL = V3P1_IMPLICIT_LIMITED_DRIVE_MODEL

DEFAULT_VIEWER_FRICTION_RANGE = (0.8, 0.8)
