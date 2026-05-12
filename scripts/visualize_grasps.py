#!/usr/bin/env -S uv run
"""
Grasp sampling visualizer.

Captures the current scene, builds collision meshes from depth, then walks
through every antipodal grasp candidate for a target object one-by-one in the
PyBullet GUI:

  - Green frame + jaw lines  → collision-free
  - Red frame + jaw lines    → in collision (with culprit detail)

Press Enter to advance to the next candidate.  Summary printed at the end.

Usage:
    python scripts/visualize_grasps.py --object-id cup_1 [--seed top_down] [--robot-ip ...]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

from ptp.kinematics.gripper_orientations import PRESET_DESCRIPTIONS, PRESET_QUATERNIONS

_GRIPPER_WIDTH_M    = 0.085
_FINGER_THICKNESS_M = 0.012
_N_ROTATIONS        = 36


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize antipodal grasp candidates in PyBullet")
    p.add_argument("--robot-ip",   default="192.168.1.224")
    p.add_argument("--object-id",  default=None,
                   help="Object ID to grasp (prompted if omitted)")
    p.add_argument("--seed",       default="top_down",
                   choices=list(PRESET_QUATERNIONS.keys()),
                   help="Seed orientation for grasp sampling")
    p.add_argument("--n-rotations", type=int, default=_N_ROTATIONS,
                   help="Number of in-plane rotation candidates")
    p.add_argument("--gsam2-model", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--perception-pool",
                   default=str(_REPO_ROOT / "outputs" / "perception_pool"))
    p.add_argument("--no-robot", action="store_true",
                   help="Skip real robot connection (GUI only, no joint sync)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Scene setup (mirrors run_ptp.py)
# ---------------------------------------------------------------------------

def _capture_frame():
    from ptp.camera import REALSENSE_AVAILABLE, CameraIntrinsics
    if REALSENSE_AVAILABLE:
        from ptp.camera.realsense_camera import RealSenseCamera
        with RealSenseCamera(width=640, height=480, fps=30) as cam:
            color, depth = cam.get_aligned_frames()
            intrinsics = cam.get_camera_intrinsics()
    else:
        from ptp.camera.realsense_camera import _load_latest_snapshot
        color, depth, intrinsics = _load_latest_snapshot(Path(args_global.perception_pool))
    return color, depth, intrinsics


def _detect_objects(color, depth, intrinsics, gsam2_model, robot_state=None):
    import asyncio
    from ptp.perception.gsam2.gsam2_tracker import GSAM2ObjectTracker
    tracker = GSAM2ObjectTracker(
        grounding_model_id=gsam2_model,
        compute_clearances=False,
        compute_contacts=False,
        compute_occlusion=False,
    )
    asyncio.get_event_loop().run_until_complete(
        tracker.detect_objects(color, depth, intrinsics, robot_state=robot_state)
    )
    masks = dict(tracker._last_masks)
    return tracker.registry, masks, tracker


def _build_collider(planner, depth, intrinsics, masks, robot_state):
    from ptp.kinematics.depth_environment_collider import DepthEnvironmentCollider
    collider = DepthEnvironmentCollider(planner)
    snap_joints = (robot_state or {}).get("joints")
    if snap_joints is not None:
        planner.set_current_joint_state(snap_joints)
    collider.update_from_depth(
        depth_m=depth.astype("float32"),
        intrinsics=intrinsics,
        masks=masks,
    )
    planner.attach_collider(collider)
    return collider


# ---------------------------------------------------------------------------
# PyBullet debug drawing helpers
# ---------------------------------------------------------------------------

_debug_ids: List[int] = []


def _clear_debug(client: int) -> None:
    import pybullet as p
    for did in _debug_ids:
        try:
            p.removeUserDebugItem(did, physicsClientId=client)
        except Exception:
            pass
    _debug_ids.clear()


def _draw_line(client: int, a, b, color, width: float = 2.0) -> None:
    import pybullet as p
    did = p.addUserDebugLine(
        list(a), list(b), lineColorRGB=color, lineWidth=width,
        physicsClientId=client,
    )
    _debug_ids.append(did)


def _draw_text(client: int, pos, text: str, color) -> None:
    import pybullet as p
    did = p.addUserDebugText(
        text, list(pos), textColorRGB=color, textSize=1.2,
        physicsClientId=client,
    )
    _debug_ids.append(did)


def _draw_grasp(client: int, position: np.ndarray, orientation_xyzw: np.ndarray,
                in_collision: bool, label: str = "") -> None:
    """Draw the gripper TCP frame and jaw spread at the given pose."""
    rot   = Rotation.from_quat(orientation_xyzw)
    x_ax  = rot.apply([1, 0, 0])   # grasp axis (jaw direction)
    z_ax  = rot.apply([0, 0, 1])   # approach axis
    y_ax  = rot.apply([0, 1, 0])

    pos   = np.asarray(position)
    half  = _GRIPPER_WIDTH_M / 2.0
    ax_len = 0.06
    c_ok  = [0.0, 1.0, 0.0]
    c_bad = [1.0, 0.2, 0.0]
    c     = c_bad if in_collision else c_ok
    c_dim = [v * 0.5 for v in c]

    # Approach axis (Z)
    _draw_line(client, pos, pos + ax_len * z_ax, c, width=3.0)
    # Grasp axis (X) — jaw spread
    jaw_a = pos + half * x_ax
    jaw_b = pos - half * x_ax
    _draw_line(client, jaw_b, jaw_a, c, width=3.0)
    # Finger tips (small perpendicular stubs)
    stub = 0.02
    _draw_line(client, jaw_a, jaw_a + stub * z_ax, c_dim, width=1.5)
    _draw_line(client, jaw_b, jaw_b + stub * z_ax, c_dim, width=1.5)
    # Y axis
    _draw_line(client, pos, pos + ax_len * 0.5 * y_ax, [0.2, 0.4, 1.0], width=1.5)

    if label:
        _draw_text(client, pos + 0.04 * z_ax + 0.01 * y_ax, label, c)


# ---------------------------------------------------------------------------
# Grasp iteration (mirrors GraspPlanner.plan but yields every candidate)
# ---------------------------------------------------------------------------

def _iter_grasps(
    contact_position: np.ndarray,
    object_points: Optional[np.ndarray],
    seed_orientation: str,
    n_rotations: int,
    planner,
    ignore_labels: Optional[set],
):
    """Yield (position, quat_xyzw, angle_rad, score, is_clear, collision_detail) per candidate."""
    from ptp.grasp.grasp_planner import GraspPlanner

    seed_quat = np.array(PRESET_QUATERNIONS[seed_orientation])
    seed_rot  = Rotation.from_quat(seed_quat)
    approach  = seed_rot.apply([0.0, 0.0, 1.0])
    approach /= np.linalg.norm(approach) + 1e-9

    hint = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(approach, hint)) > 0.9:
        hint = np.array([1.0, 0.0, 0.0])
    u = np.cross(approach, hint); u /= np.linalg.norm(u) + 1e-9
    v = np.cross(approach, u);   v /= np.linalg.norm(v) + 1e-9

    half_width   = _GRIPPER_WIDTH_M / 2.0
    floor_clr_z  = _FINGER_THICKNESS_M

    env_bodies, floor_body = planner._build_collision_bodies(ignore_labels=ignore_labels)
    n_links = __import__("pybullet").getNumJoints(planner._robot_id,
                                                   physicsClientId=planner._physics_client)

    for angle in np.linspace(0.0, math.pi, n_rotations, endpoint=False):
        grasp_axis = np.cos(angle) * u + np.sin(angle) * v
        grasp_axis /= np.linalg.norm(grasp_axis) + 1e-9

        # Antipodal center
        if object_points is not None and len(object_points) >= 4:
            rel  = object_points - contact_position
            proj = rel @ grasp_axis
            pos_mask = (proj >= 0) & (proj <= half_width)
            neg_mask = (proj <= 0) & (proj >= -half_width)
            if np.any(pos_mask) and np.any(neg_mask):
                midpoint = (float(np.max(proj[pos_mask])) + float(np.min(proj[neg_mask]))) / 2.0
                jaw_spread = float(np.max(proj[pos_mask])) - float(np.min(proj[neg_mask]))
                grasp_center = contact_position + midpoint * grasp_axis
            else:
                grasp_center, jaw_spread = contact_position.copy(), 0.0
        else:
            grasp_center, jaw_spread = contact_position.copy(), 0.0

        # Floor clearance
        jaw_a = grasp_center + half_width * grasp_axis
        jaw_b = grasp_center - half_width * grasp_axis
        if min(jaw_a[2], jaw_b[2]) < floor_clr_z:
            continue

        # Build quaternion
        x_axis = grasp_axis
        z_axis = approach
        y_axis = np.cross(z_axis, x_axis)
        ny = np.linalg.norm(y_axis)
        if ny < 1e-9:
            continue
        y_axis /= ny
        R    = np.stack([x_axis, y_axis, z_axis], axis=1)
        quat = Rotation.from_matrix(R).as_quat()

        # Score
        midpt  = float((grasp_center - contact_position) @ grasp_axis)
        score  = -abs(midpt) + 0.3 * (jaw_spread / _GRIPPER_WIDTH_M)

        # Apply goal joints to sim to check collision and get joint state
        goal_joints = None
        try:
            success, traj, _ = planner.move_to_pose(
                target_position=grasp_center.tolist(),
                target_orientation=quat.tolist(),
                execute=False,
                ignore_labels=ignore_labels,
            )
            if success and traj is not None:
                goal_joints = traj[-1]
        except Exception:
            pass

        if goal_joints is not None:
            is_clear      = planner._is_state_valid(goal_joints, env_bodies, floor_body, 0.005, n_links)
            collision_str = "" if is_clear else planner._describe_collision(goal_joints)
            # Leave sim at goal joints for visualisation
            for j, ji in enumerate(planner._movable_joints):
                if j < len(goal_joints):
                    __import__("pybullet").resetJointState(
                        planner._robot_id, ji, float(goal_joints[j]),
                        physicsClientId=planner._physics_client,
                    )
        else:
            is_clear, collision_str = False, "IK failed"

        yield grasp_center, quat, float(angle), float(score), is_clear, collision_str, goal_joints


# ---------------------------------------------------------------------------
# Point cloud from depth + mask
# ---------------------------------------------------------------------------

def _object_point_cloud(
    depth: np.ndarray,
    intrinsics,
    mask: np.ndarray,
    robot_state: Optional[Dict],
    max_pts: int = 2000,
) -> Optional[np.ndarray]:
    """Back-project masked depth pixels to 3-D world-frame points."""
    from ptp.perception.utils.coordinates import compute_3d_position
    h, w = depth.shape
    ys, xs = np.where(mask.astype(bool))
    if len(ys) == 0:
        return None
    step = max(1, len(ys) // max_pts)
    pts  = []
    cam_tf = (robot_state or {}).get("camera")
    for y, x in zip(ys[::step], xs[::step]):
        ny = int(y / h * 1000)
        nx = int(x / w * 1000)
        p3 = compute_3d_position([ny, nx], depth, intrinsics)
        if p3 is None:
            continue
        if cam_tf is not None:
            try:
                pos = np.array(cam_tf["position"], dtype=float)
                rot = Rotation.from_quat(cam_tf["quaternion_xyzw"])
                p3 = rot.apply(p3) + pos
            except Exception:
                pass
        pts.append(p3)
    return np.array(pts) if pts else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

args_global: argparse.Namespace  # set in main, used by _capture_frame


def main() -> None:
    global args_global
    args = _parse_args()
    args_global = args

    # ── 1. Robot + planner ──────────────────────────────────────────────────
    robot_state: Optional[Dict] = None
    robot = None
    if args.no_robot:
        from ptp.kinematics.xarm_pybullet_planned_primitives import XArmPybulletPlannedPrimitives
        primitives = XArmPybulletPlannedPrimitives(robot=None, use_gui=True)
        planner    = primitives._planner
        print("No-robot mode: PyBullet GUI open, joint state not synced to real arm.")
    else:
        from ptp.kinematics.xarm_robot_adapter import XArmRobotAdapter
        from ptp.kinematics.xarm_pybullet_planned_primitives import XArmPybulletPlannedPrimitives
        print(f"Connecting to xArm at {args.robot_ip}…")
        robot      = XArmRobotAdapter(args.robot_ip)
        primitives = XArmPybulletPlannedPrimitives(robot=robot, use_gui=True)
        planner    = primitives._planner
        joints     = robot.get_robot_joint_state()
        if joints is not None:
            planner.set_current_joint_state(joints)
            robot_state = {"joints": joints.tolist()}
            cam_pos, cam_rot = planner.get_camera_transform()
            if cam_pos is not None and cam_rot is not None:
                robot_state["camera"] = {
                    "position": cam_pos.tolist(),
                    "quaternion_xyzw": cam_rot.as_quat().tolist(),
                }
        print("Robot connected.\n")

    client = planner._physics_client

    # ── 2. Capture frame + perception ───────────────────────────────────────
    print("Capturing frame…")
    color, depth, intrinsics = _capture_frame()
    print("Running GSAM2 object detection…")
    registry, masks, _ = _detect_objects(color, depth, intrinsics, args.gsam2_model,
                                          robot_state=robot_state)

    objects = registry.get_all_objects() if hasattr(registry, "get_all_objects") else []

    if not objects:
        print("ERROR: no objects detected. Check GSAM2 model / prompt text.")
        sys.exit(1)

    # ── 3. Build collision meshes ────────────────────────────────────────────
    print("\nBuilding collision meshes…")
    try:
        _build_collider(planner, depth, intrinsics, masks, robot_state)
        print("Collision meshes ready.")
    except Exception as exc:
        print(f"WARNING: could not build collision meshes: {exc}")

    # ── 4. Choose target object ──────────────────────────────────────────────
    print(f"\nDetected {len(objects)} object(s):")
    obj_list = list(objects)
    for idx, obj in enumerate(obj_list):
        oid   = getattr(obj, "object_id", "?")
        otype = getattr(obj, "object_type", "?")
        pos3d = getattr(obj, "position_3d", None)
        pos3d_str = (f"  3D=({pos3d[0]:.3f}, {pos3d[1]:.3f}, {pos3d[2]:.3f})"
                     if pos3d is not None and len(pos3d) >= 3 else "")
        print(f"  [{idx}]  {oid}  ({otype}){pos3d_str}")

    object_id = args.object_id
    if object_id is not None:
        # Validate the CLI-provided ID
        if not any(getattr(o, "object_id", None) == object_id for o in obj_list):
            print(f"WARNING: --object-id '{object_id}' not found in detections.")
    else:
        print()
        choice = input("Select object by index or ID: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(obj_list):
                object_id = getattr(obj_list[idx], "object_id", None)
            else:
                print(f"Index {idx} out of range.")
                sys.exit(1)
        else:
            object_id = choice

    obj_info = next((o for o in obj_list if getattr(o, "object_id", None) == object_id), None)
    if obj_info is None:
        print(f"Object '{object_id}' not found in registry. Continuing without point cloud.")

    # Contact position: use 3D position from registry if available.
    contact_pos_raw = getattr(obj_info, "position_3d", None) if obj_info else None
    if contact_pos_raw is None:
        contact_pos_raw = getattr(obj_info, "position_2d", None)
        print("WARNING: no 3D position — using 2D centroid fallback (expect inaccuracy).")
    if contact_pos_raw is None:
        print("ERROR: no position available for target object.")
        sys.exit(1)
    contact_position = np.asarray(contact_pos_raw, dtype=float)[:3]

    # Point cloud for antipodal refinement.
    mask = masks.get(object_id)
    obj_pts = _object_point_cloud(depth, intrinsics, mask, robot_state) if mask is not None else None
    if obj_pts is not None:
        print(f"Point cloud: {len(obj_pts)} pts for '{object_id}'")
    else:
        print("No mask / point cloud — orientation sampling only (no antipodal refinement).")

    seed = args.seed
    print(f"\nSeed orientation: {seed}  ({PRESET_DESCRIPTIONS[seed]})")
    print(f"Sampling {args.n_rotations} rotation candidates…\n")
    print("Press Enter to advance.  Ctrl-C to quit.\n")

    # ── 5. Walk through candidates ───────────────────────────────────────────
    results: List[Dict] = []
    ignore_labels = {object_id} if object_id else None

    for i, (pos, quat, angle_rad, score, is_clear, coll_str, goal_joints) in enumerate(
        _iter_grasps(contact_position, obj_pts, seed, args.n_rotations,
                     planner, ignore_labels)
    ):
        _clear_debug(client)
        status = "OK" if is_clear else "COLLISION"
        label  = f"[{i}] {status} {math.degrees(angle_rad):.0f}°"
        _draw_grasp(client, pos, quat, in_collision=not is_clear, label=label)

        rpy = np.degrees(Rotation.from_quat(quat).as_euler("xyz"))
        print(f"{'─'*60}")
        print(f"Candidate [{i}]  angle={math.degrees(angle_rad):.1f}°  score={score:.4f}")
        print(f"  Position : x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}")
        print(f"  RPY (deg): roll={rpy[0]:.1f}  pitch={rpy[1]:.1f}  yaw={rpy[2]:.1f}")
        print(f"  Status   : {status}")
        if not is_clear:
            print(f"  Collision: {coll_str}")
        if goal_joints is None:
            print(f"  IK       : failed — cannot execute")

        results.append({"idx": i, "angle_deg": math.degrees(angle_rad), "score": score,
                         "clear": is_clear, "collision": coll_str,
                         "pos": pos.tolist(), "quat_xyzw": quat.tolist()})

        # Offer to send joints to the real robot.
        if not args.no_robot and goal_joints is not None and robot is not None:
            try:
                ans = input("  [Enter] next  [m] move robot  [Ctrl-C] quit\n").strip().lower()
            except KeyboardInterrupt:
                print("\nStopped.")
                break
            if ans == "m":
                joints_deg = np.degrees(goal_joints[:7]).tolist()
                print(f"  Moving robot to joints (deg): {[f'{v:.1f}' for v in joints_deg]}")
                ok = robot.set_robot_joint_angles(goal_joints[:7].tolist())
                if ok:
                    print("  Robot moved.")
                else:
                    print("  ERROR: robot move returned failure code.")
        else:
            try:
                input("  [Enter] next  [Ctrl-C] quit\n")
            except KeyboardInterrupt:
                print("\nStopped.")
                break

    # ── 6. Summary ───────────────────────────────────────────────────────────
    _clear_debug(client)
    n_clear    = sum(r["clear"] for r in results)
    n_total    = len(results)
    best_clear = max((r for r in results if r["clear"]), key=lambda r: r["score"], default=None)
    best_any   = max(results, key=lambda r: r["score"], default=None) if results else None

    print(f"\n{'═'*60}")
    print(f"Summary for '{object_id}'  seed={seed}")
    print(f"  Candidates evaluated : {n_total}")
    print(f"  Collision-free       : {n_clear}")
    print(f"  In collision         : {n_total - n_clear}")
    if best_clear:
        print(f"  Best clear candidate : [{best_clear['idx']}]  "
              f"angle={best_clear['angle_deg']:.1f}°  score={best_clear['score']:.4f}")
    else:
        print("  Best clear candidate : none found")
    if best_any and (best_clear is None or best_any["idx"] != best_clear["idx"]):
        print(f"  Best overall (may collide): [{best_any['idx']}]  "
              f"angle={best_any['angle_deg']:.1f}°  score={best_any['score']:.4f}")

    input("\nPress Enter to exit…\n")


if __name__ == "__main__":
    main()
