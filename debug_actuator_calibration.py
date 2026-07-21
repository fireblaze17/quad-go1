"""Diagnostics for the Chrono Go1 actuator path.

This script does not train or tune a policy. It checks the mechanical/control
assumptions that must be true before limited-actuator standing training is
meaningful. The active path is the driveline/clutch implicit limited drive, but
some historical raw torque-PD checks remain available for debugging.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass

import numpy as np
import pychrono.core as chrono
import pychrono.parsers as parsers

import go1_env as env_mod
from go1_env import (
    Go1Env,
    _FOOT_BODY_NAMES,
    _HOME_JOINT_ANGLES,
    _HIP_ACTION_INDICES,
    _HIP_ACTION_SCALE_MULTIPLIER,
    _JOINT_AXES,
    _JOINT_AXIS_SIGN,
    _JOINT_EFFORT_LIMIT,
    _JOINT_HIGH,
    _JOINT_LOW,
    _JOINT_NAMES,
    _PHYSICS_SUBSTEPS,
    _PHYSICS_TIME_STEP,
    _RESET_NOISE_COMPONENTS,
    _RESET_NOISE_LEVELS,
    _TIME_STEP,
    _ACTION_SCALE,
)


GAIN_SWEEP = (
    (20.0, 0.5),
    (40.0, 1.0),
    (60.0, 1.5),
    (80.0, 2.0),
    (100.0, 2.5),
)


@dataclass
class Args:
    mode: str
    terrain: str
    friction_min: float
    friction_max: float
    max_steps: int
    reset_noise_level: str
    reset_noise_components: str
    impulse_nm: float
    impulse_steps: int


def _make_env(args: Args, *, max_steps: int | None = None) -> Go1Env:
    return Go1Env(
        max_steps=max_steps or args.max_steps,
        terrain=args.terrain,
        friction_range=(args.friction_min, args.friction_max),
        reset_noise_level=args.reset_noise_level,
        reset_noise_components=args.reset_noise_components,
    )


def _print_header(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def _joint_type(name: str) -> str:
    if "_hip_" in name:
        return "hip"
    if "_thigh_" in name:
        return "thigh"
    if "_calf_" in name:
        return "calf"
    return "unknown"


def _foot_loads(env: Go1Env) -> dict[str, float]:
    return {
        name.split("_")[0]: abs(float(foot.GetContactForce().y))
        for name, foot in zip(_FOOT_BODY_NAMES, env._feet)
    }


def _set_all_torques(env: Go1Env, torques: np.ndarray | None = None) -> None:
    torques = np.zeros(12, dtype=np.float32) if torques is None else torques
    for function, torque in zip(env._motor_funcs, torques):
        function.SetConstant(float(torque))


def _lift_robot_bodies(env: Go1Env, lift: float = 0.25) -> None:
    for body in env._system.GetBodies():
        if body.IsFixed():
            continue
        pos = body.GetPos()
        body.SetPos(chrono.ChVector3d(float(pos.x), float(pos.y) + lift, float(pos.z)))


def _signed_body_projected_joint_vel(env: Go1Env, index: int) -> float:
    body1, body2 = env._joint_body_pairs[index]
    return env._joint_vel(
        body1,
        body2,
        int(_JOINT_AXES[index]),
        float(_JOINT_AXIS_SIGN[index]),
    )


def _motor_angle_dt(motor, index: int) -> float | None:
    try:
        torque_motor = chrono.CastToChLinkMotorRotationTorque(motor)
        return float(torque_motor.GetMotorAngleDt())
    except Exception:
        try:
            rotation_motor = chrono.CastToChLinkMotorRotation(motor)
            return float(rotation_motor.GetMotorAngleDt())
        except Exception:
            return None


@contextlib.contextmanager
def _temporary_gains(kp: float, kd: float):
    old_kp = env_mod._PD_KP
    old_kd = env_mod._PD_KD
    env_mod._PD_KP = float(kp)
    env_mod._PD_KD = float(kd)
    try:
        yield
    finally:
        env_mod._PD_KP = old_kp
        env_mod._PD_KD = old_kd


def check_joint_order(args: Args) -> None:
    _print_header("Joint Order Consistency")
    env = _make_env(args, max_steps=1)
    try:
        env.reset()
        motor_names = [motor.GetName() for motor in env._motors]
        print(f"expected_order={list(_JOINT_NAMES)}")
        print(f"parser_order  ={motor_names}")
        print(f"hip_action_indices={_HIP_ACTION_INDICES.astype(int).tolist()}")
        print(
            "idx name              type   axis sign  low      high     effort   home     hip_scaled"
        )
        for i, name in enumerate(_JOINT_NAMES):
            hip_scaled = "yes" if int(i) in set(_HIP_ACTION_INDICES.astype(int).tolist()) else "no"
            print(
                f"{i:02d}  {name:16s} {_joint_type(name):6s} "
                f"{int(_JOINT_AXES[i]):4d} {float(_JOINT_AXIS_SIGN[i]):+4.0f} "
                f"{float(_JOINT_LOW[i]):+8.3f} {float(_JOINT_HIGH[i]):+8.3f} "
                f"{float(_JOINT_EFFORT_LIMIT[i]):8.3f} {float(_HOME_JOINT_ANGLES[i]):+8.3f} "
                f"{hip_scaled}"
            )
        print(f"PASS_joint_order={motor_names == list(_JOINT_NAMES)}")
    finally:
        env.close()


def check_reset(args: Args) -> None:
    _print_header("Reset/Home Check")
    env = _make_env(args, max_steps=1)
    try:
        env.reset()
        q = env._last_joint_pos
        err = q - _HOME_JOINT_ANGLES
        print(f"observation_shape={env.observation_space.shape}")
        print(f"max_abs_reset_q_error={float(np.max(np.abs(err))):.9f}")
        print("idx name              target    read      error")
        for i, name in enumerate(_JOINT_NAMES):
            print(
                f"{i:02d}  {name:16s} "
                f"{float(_HOME_JOINT_ANGLES[i]):+8.4f} "
                f"{float(q[i]):+8.4f} {float(err[i]):+9.5f}"
            )
        print(f"base_relative_height={env._base_relative_height():.6f}")
        print(f"min_foot_clearance={env._min_foot_clearance():.6f}")
        print(f"initial_foot_loads={_foot_loads(env)}")
    finally:
        env.close()


def check_motor_type(args: Args) -> None:
    _print_header("Motor Type / Torque Function Check")
    env = _make_env(args, max_steps=1)
    try:
        env.reset()
        all_pass = True
        print("idx name              base_type           cast_type                         torque_after_SetTorque  torque_after_SetMotor")
        for i, (name, motor) in enumerate(zip(_JOINT_NAMES, env._motors)):
            cast_ok = True
            try:
                torque_motor = chrono.CastToChLinkMotorRotationTorque(motor)
                torque_motor.SetTorqueFunction(chrono.ChFunctionConst(1.0))
                torque_val_after_torque = float(torque_motor.GetTorqueFunction().GetVal(0.0))
                motor.SetMotorFunction(chrono.ChFunctionConst(2.0))
                torque_val_after_motor = float(torque_motor.GetTorqueFunction().GetVal(0.0))
            except Exception:
                cast_ok = False
                torque_motor = None
                torque_val_after_torque = np.nan
                torque_val_after_motor = np.nan
            all_pass = (
                all_pass
                and cast_ok
                and abs(torque_val_after_torque - 1.0) < 1e-9
                and abs(torque_val_after_motor - 2.0) < 1e-9
            )
            print(
                f"{i:02d}  {name:16s} {type(motor).__name__:18s} "
                f"{type(torque_motor).__name__ if torque_motor is not None else 'CAST_FAILED':32s} "
                f"{torque_val_after_torque:21.3f} {torque_val_after_motor:22.3f}"
            )
        print(f"PASS_motor_type={all_pass}")
    finally:
        env.close()


def _impulse_once(args: Args, index: int, torque: float) -> tuple[float, float | None, float]:
    env = _make_env(args, max_steps=1)
    try:
        env.reset()
        _lift_robot_bodies(env)
        env._trunk.SetFixed(True)
        _set_all_torques(env)
        q0 = env._read_joint_angles()
        motor_angle_dt0 = _motor_angle_dt(env._motors[index], index)
        torques = np.zeros(12, dtype=np.float32)
        torques[index] = float(torque)
        _set_all_torques(env, torques)
        for _ in range(args.impulse_steps):
            env._system.DoStepDynamics(_PHYSICS_TIME_STEP)
        q1 = env._read_joint_angles()
        body_projected_qd = _signed_body_projected_joint_vel(env, index)
        motor_angle_dt1 = _motor_angle_dt(env._motors[index], index)
        fd_qd = float((q1[index] - q0[index]) / (args.impulse_steps * _PHYSICS_TIME_STEP))
        motor_qd = motor_angle_dt1 if motor_angle_dt1 is not None else motor_angle_dt0
        return float(q1[index] - q0[index]), motor_qd, float(body_projected_qd if np.isfinite(body_projected_qd) else np.nan), fd_qd
    finally:
        env.close()


def check_impulse(args: Args) -> None:
    _print_header("Torque Direction Impulse Check")
    all_pass = True
    print("idx name              +dq        -dq        pass")
    for i, name in enumerate(_JOINT_NAMES):
        pos_dq, _, _, _ = _impulse_once(args, i, abs(args.impulse_nm))
        neg_dq, _, _, _ = _impulse_once(args, i, -abs(args.impulse_nm))
        passed = pos_dq > 0.0 and neg_dq < 0.0
        all_pass = all_pass and passed
        print(f"{i:02d}  {name:16s} {pos_dq:+10.6f} {neg_dq:+10.6f} {passed}")
    print(f"PASS_torque_direction={all_pass}")


def check_velocity(args: Args) -> None:
    _print_header("Velocity / Damping Direction Check")
    all_pass = True
    print("idx name              fd_qd      motor_qd   bodyproj_qd damping_opposes")
    for i, name in enumerate(_JOINT_NAMES):
        dq, motor_qd, body_qd, fd_qd = _impulse_once(args, i, abs(args.impulse_nm))
        # Positive motion should produce negative damping torque for Kd > 0.
        damping_opposes = (-env_mod._PD_KD * fd_qd) < 0.0 if fd_qd > 0.0 else False
        all_pass = all_pass and damping_opposes and dq > 0.0
        motor_text = f"{motor_qd:+10.6f}" if motor_qd is not None else "     None "
        print(
            f"{i:02d}  {name:16s} {fd_qd:+10.6f} {motor_text} "
            f"{body_qd:+12.6f} {damping_opposes}"
        )
    print(f"PASS_damping_direction={all_pass}")


def check_reset_handoff(args: Args) -> None:
    _print_header("Reset Handoff Check")
    env = _make_env(args, max_steps=1)
    try:
        env._sample_reset_noise()
        assembly_system, _, assembly_parser = env._build_imported_system(
            parsers.ChParserURDF.ActuationType_POSITION
        )
        assembly_motors = [assembly_parser.GetChMotor(name) for name in _JOINT_NAMES]
        reset_targets = np.clip(env._reset_joint_targets, _JOINT_LOW, _JOINT_HIGH)
        for motor, target in zip(assembly_motors, reset_targets):
            motor.SetMotorFunction(chrono.ChFunctionConst(float(target)))
        assembly_trunk = assembly_parser.GetChBody("trunk")
        assembly_feet = [assembly_parser.GetChBody(name) for name in _FOOT_BODY_NAMES]
        assembly_trunk.SetFixed(True)
        assembly_system.DoAssembly(1)
        assembly_trunk.SetFixed(False)
        env._apply_reset_contact_safety_lift(system=assembly_system, feet=assembly_feet)

        runtime_system, _, runtime_parser = env._build_imported_system(
            parsers.ChParserURDF.ActuationType_FORCE
        )
        env._copy_body_states(assembly_system, runtime_system)
        env._cache_robot_handles(runtime_system, None, runtime_parser)
        runtime_q = env._read_joint_angles()

        env._system = assembly_system
        env._motors = assembly_motors
        assembly_q = env._read_joint_angles()

        q_delta = runtime_q - assembly_q
        print(f"max_abs_assembly_error={float(np.max(np.abs(assembly_q - reset_targets))):.9f}")
        print(f"max_abs_runtime_handoff_delta={float(np.max(np.abs(q_delta))):.9f}")
        print(f"assembly_min_foot_clearance={env._min_foot_clearance(assembly_feet):.6f}")
        runtime_trunk = runtime_parser.GetChBody('trunk')
        print(
            "runtime_base_pos="
            f"({float(runtime_trunk.GetPos().x):+.6f}, "
            f"{float(runtime_trunk.GetPos().y):+.6f}, "
            f"{float(runtime_trunk.GetPos().z):+.6f})"
        )
        print("idx name              assembly  runtime   delta")
        for i, name in enumerate(_JOINT_NAMES):
            print(
                f"{i:02d}  {name:16s} "
                f"{float(assembly_q[i]):+8.4f} {float(runtime_q[i]):+8.4f} {float(q_delta[i]):+9.5f}"
            )
        print(f"PASS_reset_handoff={float(np.max(np.abs(q_delta))) < 1e-5}")
    finally:
        env.close()


def run_zero_action(args: Args, *, kp: float | None = None, kd: float | None = None) -> dict:
    context = _temporary_gains(kp, kd) if kp is not None and kd is not None else contextlib.nullcontext()
    with context:
        env = _make_env(args, max_steps=args.max_steps)
        try:
            env.reset()
            max_tilt = 0.0
            min_height = float("inf")
            max_joint_error = 0.0
            max_joint_vel = 0.0
            torque_means = []
            torque_maxes = []
            sat_fracs = []
            final_terms = {}
            reason = None
            steps = 0
            for step in range(args.max_steps):
                _, _, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
                terms = info["reward_terms"]
                final_terms = terms
                max_tilt = max(
                    max_tilt,
                    abs(float(terms.get("trunk_x_up", 0.0))),
                    abs(float(terms.get("trunk_z_up", 0.0))),
                )
                min_height = min(min_height, float(terms.get("base_relative_height", 0.0)))
                max_joint_error = max(max_joint_error, float(terms.get("pose_error", 0.0)))
                max_joint_vel = max(max_joint_vel, float(terms.get("max_abs_joint_vel", 0.0)))
                torque_means.append(float(terms.get("mean_abs_motor_torque", 0.0)))
                torque_maxes.append(float(terms.get("max_abs_motor_torque", 0.0)))
                sat_fracs.append(float(terms.get("fraction_torque_saturated", 0.0)))
                steps = step + 1
                if terminated or truncated:
                    reason = info.get("termination_reason") or ("truncated" if truncated else "terminated")
                    break
            result = {
                "kp": env_mod._PD_KP,
                "kd": env_mod._PD_KD,
                "steps": steps,
                "reason": reason or "completed",
                "max_tilt_xz": max_tilt,
                "min_relative_height": min_height,
                "max_joint_error": max_joint_error,
                "max_joint_vel": max_joint_vel,
                "mean_abs_torque": float(np.mean(torque_means)) if torque_means else 0.0,
                "max_abs_torque": float(np.max(torque_maxes)) if torque_maxes else 0.0,
                "mean_saturation_fraction": float(np.mean(sat_fracs)) if sat_fracs else 0.0,
                "max_saturation_fraction": float(np.max(sat_fracs)) if sat_fracs else 0.0,
                "foot_loads": _foot_loads(env),
                "final_height": float(final_terms.get("base_relative_height", 0.0)),
                "final_upright": float(final_terms.get("upright_score", 0.0)),
            }
            return result
        finally:
            env.close()


def check_zero_action(args: Args) -> None:
    _print_header("Zero-Action Stability Check")
    result = run_zero_action(args)
    for key, value in result.items():
        print(f"{key}={value}")


def check_gain_sweep(args: Args) -> None:
    _print_header("Gain Sweep")
    print(
        "kp     kd     steps reason      max_tilt  min_h    max_qerr  max_qd   "
        "mean_tau max_tau  mean_sat max_sat"
    )
    best = None
    for kp, kd in GAIN_SWEEP:
        result = run_zero_action(args, kp=kp, kd=kd)
        if best is None or result["steps"] > best["steps"]:
            best = result
        print(
            f"{kp:5.1f} {kd:6.2f} {result['steps']:5d} {result['reason']:10s} "
            f"{result['max_tilt_xz']:8.3f} {result['min_relative_height']:7.3f} "
            f"{result['max_joint_error']:9.3f} {result['max_joint_vel']:8.3f} "
            f"{result['mean_abs_torque']:8.3f} {result['max_abs_torque']:8.3f} "
            f"{result['mean_saturation_fraction']:8.3f} {result['max_saturation_fraction']:7.3f}"
        )
    print(f"best_by_survival=Kp{best['kp']:.1f}_Kd{best['kd']:.2f}_steps{best['steps']}")


def check_implicit_drive(args: Args) -> None:
    _print_header("Implicit Limited Drive Check")
    env = _make_env(args, max_steps=100)
    try:
        env.reset()
        print(f"actuator_model={env_mod._ACTUATOR_MODEL}")
        print(f"drive_links={len(env._drive_links)}")
        print(f"drive_clutches={len(env._drive_clutches)}")
        print(f"drive_motors={len(env._drive_motors)}")
        all_created = (
            len(env._drive_links) == 12
            and len(env._drive_clutches) == 12
            and len(env._drive_motors) == 12
        )
        limits_match = True
        print("idx name              clutch_limit expected  initial_torque")
        for i, (name, clutch, drive, expected) in enumerate(
            zip(_JOINT_NAMES, env._drive_clutches, env._drive_links, _JOINT_EFFORT_LIMIT)
        ):
            limit = float(clutch.GetTorqueLimit())
            torque = float(drive.GetMotorTorque())
            limits_match = limits_match and abs(limit - float(expected)) < 1e-6
            print(f"{i:02d}  {name:16s} {limit:12.4f} {float(expected):8.4f} {torque:+14.6f}")

        direction_pass = True
        print("\nidx name              q_before  q_after   delta    pass")
        for i, name in enumerate(_JOINT_NAMES):
            env_i = _make_env(args, max_steps=10)
            try:
                env_i.reset()
                _lift_robot_bodies(env_i)
                env_i._trunk.SetFixed(True)
                q0 = env_i._last_joint_pos.copy()
                action = np.zeros(12, dtype=np.float32)
                action[i] = 0.5
                for _ in range(5):
                    env_i.step(action)
                q1 = env_i._last_joint_pos.copy()
                delta = float(q1[i] - q0[i])
                passed = delta > 0.0
                direction_pass = direction_pass and passed
                print(f"{i:02d}  {name:16s} {float(q0[i]):+8.4f} {float(q1[i]):+8.4f} {delta:+9.6f} {passed}")
            finally:
                env_i.close()

        zero = run_zero_action(args)
        print("\nzero_action:")
        for key, value in zero.items():
            print(f"  {key}={value}")
        print(f"PASS_implicit_drive_created={all_created}")
        print(f"PASS_clutch_limits={limits_match}")
        print(f"PASS_positive_target_direction={direction_pass}")
        print(f"PASS_implicit_drive={all_created and limits_match and direction_pass}")
    finally:
        env.close()


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "all",
            "reset",
            "motor-type",
            "impulse",
            "velocity",
            "handoff",
            "zero-action",
            "gain-sweep",
            "implicit-drive",
        ),
        default="all",
    )
    parser.add_argument("--terrain", choices=("flat", "scm"), default="flat")
    parser.add_argument("--friction-min", type=float, default=0.8)
    parser.add_argument("--friction-max", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--reset-noise-level", choices=_RESET_NOISE_LEVELS, default="clean")
    parser.add_argument("--reset-noise-components", choices=_RESET_NOISE_COMPONENTS, default="combined")
    parser.add_argument("--impulse-nm", type=float, default=1.0)
    parser.add_argument("--impulse-steps", type=int, default=5)
    return Args(**vars(parser.parse_args()))


def main() -> None:
    args = parse_args()
    print("active_assumptions:")
    print(f"  observation_dimension=45")
    print(f"  actuator_model={env_mod._ACTUATOR_MODEL}")
    print(f"  pd_kp={env_mod._PD_KP}")
    print(f"  pd_kd={env_mod._PD_KD}")
    print(f"  action_scale={_ACTION_SCALE}")
    print(f"  hip_action_scale_multiplier={_HIP_ACTION_SCALE_MULTIPLIER}")
    print(f"  time_step={_TIME_STEP}")
    print(f"  physics_time_step={_PHYSICS_TIME_STEP}")
    print(f"  physics_substeps={_PHYSICS_SUBSTEPS}")

    modes = (
        ["reset", "implicit-drive", "zero-action", "gain-sweep"]
        if args.mode == "all"
        else [args.mode]
    )
    for mode in modes:
        if mode == "reset":
            check_joint_order(args)
            check_reset(args)
        elif mode == "motor-type":
            check_motor_type(args)
        elif mode == "impulse":
            check_impulse(args)
        elif mode == "velocity":
            check_velocity(args)
        elif mode == "handoff":
            check_reset_handoff(args)
        elif mode == "zero-action":
            check_zero_action(args)
        elif mode == "gain-sweep":
            check_gain_sweep(args)
        elif mode == "implicit-drive":
            check_implicit_drive(args)


if __name__ == "__main__":
    main()
