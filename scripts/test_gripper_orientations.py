#!/usr/bin/env -S uv run
"""
Test gripper preset orientations and grasp planner seed alignment on the real xArm.

Two modes of orientations are shown:

  [preset]  The raw quaternion from gripper_orientations.py — what the planner
            uses when no antipodal sampling is needed.

  [sampler] The orientation GraspPlanner._build_quaternion produces at angle=0
            for the same seed — verifies the sampler's starting frame matches
            the preset (i.e. the flip fix is working).

For each, the gripper moves in-place to that orientation at the current TCP
position so you can visually confirm they are identical.

Usage:
    python scripts/test_gripper_orientations.py [--robot-ip 192.168.1.224] [--speed 30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.spatial.transform import Rotation

from ptp.kinematics.gripper_orientations import PRESET_DESCRIPTIONS, PRESET_QUATERNIONS


# ---------------------------------------------------------------------------
# Build the sampler's angle=0 orientation for each seed (mirrors GraspPlanner)
# ---------------------------------------------------------------------------

def _sampler_quat_at_angle0(seed_name: str) -> list[float]:
    """Return the quaternion GraspPlanner produces at angle=0 for this seed."""
    seed_quat = np.array(PRESET_QUATERNIONS[seed_name])
    seed_rot  = Rotation.from_quat(seed_quat)
    approach  = seed_rot.apply(np.array([0.0, 0.0, 1.0]))
    approach  = approach / np.linalg.norm(approach)
    seed_jaw  = seed_rot.apply(np.array([1.0, 0.0, 0.0]))

    # _perp_frame_from_seed
    u = seed_jaw - np.dot(seed_jaw, approach) * approach
    norm_u = np.linalg.norm(u)
    if norm_u < 1e-6:
        hint = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(approach, hint)) > 0.9:
            hint = np.array([1.0, 0.0, 0.0])
        u = np.cross(approach, hint); u /= np.linalg.norm(u)
    else:
        u /= norm_u

    grasp_axis = u  # angle=0

    # _build_quaternion: X=grasp_axis, Z=approach
    x_axis = grasp_axis
    z_axis = approach
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    return Rotation.from_matrix(R).as_quat().tolist()


def _quat_angle_deg(q1: list[float], q2: list[float]) -> float:
    """Angular distance in degrees between two quaternions."""
    r1 = Rotation.from_quat(q1)
    r2 = Rotation.from_quat(q2)
    delta = (r1.inv() * r2).magnitude()
    return float(np.degrees(delta))


# ---------------------------------------------------------------------------
# All test orientations: preset + sampler angle=0 for each seed
# ---------------------------------------------------------------------------

def _build_orientation_list() -> list[dict]:
    entries = []
    for seed_name in PRESET_QUATERNIONS:
        preset_quat = PRESET_QUATERNIONS[seed_name]
        sampler_quat = _sampler_quat_at_angle0(seed_name)
        angle_diff = _quat_angle_deg(preset_quat, sampler_quat)

        entries.append({
            "label": f"{seed_name} [preset]",
            "quat_xyzw": preset_quat,
            "description": PRESET_DESCRIPTIONS[seed_name],
            "note": "",
        })
        entries.append({
            "label": f"{seed_name} [sampler angle=0]",
            "quat_xyzw": sampler_quat,
            "description": f"GraspPlanner angle=0 for '{seed_name}' seed",
            "note": f"  diff vs preset: {angle_diff:.2f}°  {'✓ MATCH' if angle_diff < 2.0 else '✗ MISMATCH — flip bug present'}",
        })
    return entries


# ---------------------------------------------------------------------------
# Robot helpers
# ---------------------------------------------------------------------------

def _quat_to_rpy_deg(quat_xyzw: list[float]) -> tuple[float, float, float]:
    r, p, y = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
    return r, p, y


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test xArm gripper preset and sampler orientations")
    p.add_argument("--robot-ip", default="192.168.1.224", help="xArm IP address")
    p.add_argument("--speed", type=float, default=30.0, help="Cartesian speed in mm/s")
    p.add_argument("--acc", type=float, default=200.0, help="Cartesian acceleration in mm/s²")
    return p.parse_args()


def _connect(robot_ip: str):
    try:
        from xarm.wrapper import XArmAPI
    except ImportError:
        print("ERROR: xarm SDK not found. Install with: pip install xarm-python-sdk")
        sys.exit(1)
    print(f"Connecting to xArm at {robot_ip}…")
    arm = XArmAPI(robot_ip, is_radian=True)
    arm.connect()
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    print("Connected.\n")
    return arm


def _get_tcp(arm) -> tuple[list[float], list[float]]:
    code, pose = arm.get_position(is_radian=True)
    if code != 0 or pose is None:
        print(f"ERROR: get_position returned code {code}")
        sys.exit(1)
    x_m, y_m, z_m = pose[0] / 1000.0, pose[1] / 1000.0, pose[2] / 1000.0
    roll, pitch, yaw = pose[3], pose[4], pose[5]
    quat_xyzw = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat().tolist()
    return [x_m, y_m, z_m], quat_xyzw


def _move_to_orientation(arm, position_m: list[float], quat_xyzw: list[float],
                          speed: float, acc: float) -> bool:
    x_mm, y_mm, z_mm = [v * 1000.0 for v in position_m]
    roll, pitch, yaw = Rotation.from_quat(quat_xyzw).as_euler("xyz")
    arm.set_mode(0)
    arm.set_state(0)
    code = arm.set_position(
        x=x_mm, y=y_mm, z=z_mm,
        roll=roll, pitch=pitch, yaw=yaw,
        speed=speed, mvacc=acc,
        is_radian=True, wait=True,
    )
    return code == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    arm = _connect(args.robot_ip)

    orientations = _build_orientation_list()

    # Print alignment summary before starting
    print("Grasp sampler alignment check (angle=0 vs preset):")
    for seed_name in PRESET_QUATERNIONS:
        q_preset  = PRESET_QUATERNIONS[seed_name]
        q_sampler = _sampler_quat_at_angle0(seed_name)
        diff = _quat_angle_deg(q_preset, q_sampler)
        status = "✓ match" if diff < 2.0 else "✗ MISMATCH"
        print(f"  {seed_name}: preset vs sampler angle=0 → {diff:.3f}°  {status}")
    print()

    while True:
        pos_m, cur_quat = _get_tcp(arm)
        cur_rpy = _quat_to_rpy_deg(cur_quat)

        print("─" * 60)
        print(f"Current TCP : x={pos_m[0]*1000:.1f}mm  y={pos_m[1]*1000:.1f}mm  z={pos_m[2]*1000:.1f}mm")
        print(f"Current RPY : roll={cur_rpy[0]:.1f}°  pitch={cur_rpy[1]:.1f}°  yaw={cur_rpy[2]:.1f}°")
        print()
        print("Orientations to test:")
        for idx, entry in enumerate(orientations):
            rpy = _quat_to_rpy_deg(entry["quat_xyzw"])
            print(f"  [{idx}] {entry['label']}")
            print(f"        {entry['description']}")
            print(f"        RPY: roll={rpy[0]:.1f}°  pitch={rpy[1]:.1f}°  yaw={rpy[2]:.1f}°{entry['note']}")
        print("  [q] quit")
        print()

        choice = input("Select orientation (number): ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("Bye.")
            break

        if not choice.isdigit() or not (0 <= int(choice) < len(orientations)):
            print(f"  Unknown choice: {choice!r}\n")
            continue

        entry = orientations[int(choice)]
        target_quat = entry["quat_xyzw"]
        target_rpy  = _quat_to_rpy_deg(target_quat)

        print(f"\nMoving to '{entry['label']}':")
        print(f"  Position (fixed): x={pos_m[0]*1000:.1f}mm  y={pos_m[1]*1000:.1f}mm  z={pos_m[2]*1000:.1f}mm")
        print(f"  Target RPY:       roll={target_rpy[0]:.1f}°  pitch={target_rpy[1]:.1f}°  yaw={target_rpy[2]:.1f}°")
        print(f"  Speed: {args.speed:.0f} mm/s")

        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Skipped.\n")
            continue

        ok = _move_to_orientation(arm, pos_m, target_quat, args.speed, args.acc)
        if ok:
            print(f"Done — gripper is now in '{entry['label']}' orientation.\n")
        else:
            print("ERROR: move command returned a non-zero error code.\n")

    arm.disconnect()


if __name__ == "__main__":
    main()
