"""
Real xArm primitives planned by the PyBullet xArm model.

This module mirrors `XArmPybulletPrimitives`, but sends successful PyBullet
joint trajectories to a real xArm-compatible interface for execution.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ptp.kinematics.gripper_orientations import preset_quat
from ptp.kinematics.xarm_pybullet_interface import XArmPybulletInterface
from ptp.utils.logging_utils import get_structured_logger

_HOME_JOINTS_DEG = [-8.1, -75.3, -24.9, 88.0, -7.6, 116.2, -34.9]
_HOME_JOINTS = np.deg2rad(_HOME_JOINTS_DEG).tolist()
_DEFAULT_POSITION_TOLERANCE_M = 0.07
_DEFAULT_ORIENTATION_TOLERANCE_RAD = 0.6
_DEFAULT_SAFE_SPEED_FACTOR = 0.5
_DEFAULT_SAFE_JOINT_SPEED = 0.5
_DEFAULT_SAFE_JOINT_ACCEL = 0.25


def _normalise_vec(v: Optional[Any]) -> Optional[np.ndarray]:
    if v is None:
        return None
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    return arr / n if n > 1e-6 else None


class XArmPybulletPlannedPrimitives:
    """Primitive API that plans in PyBullet and executes on a real xArm.

    Args:
        robot: Real xArm robot interface. It should expose
            `get_robot_joint_state`, `set_robot_joint_angles`, `open_gripper`,
            and `close_gripper`. `CuRoboMotionPlanner` satisfies this surface.
        planner: PyBullet xArm interface used for IK, FK, and frame transforms.
        registry: Optional object registry for resolving point labels.
        logger: Optional structured logger.

    Example:
        >>> real = CuRoboMotionPlanner(robot_ip="192.168.1.XXX")
        >>> primitives = XArmPybulletPlannedPrimitives(robot=real)
        >>> primitives.move_gripper_to_pose(target_position=[0.35, 0.0, 0.30])
    """

    def __init__(
        self,
        robot: Any,
        planner: Optional[XArmPybulletInterface] = None,
        registry: Optional[Any] = None,
        logger: Optional[Any] = None,
        use_gui: bool = False,
    ) -> None:
        self._robot = robot
        self._planner = planner or XArmPybulletInterface(use_gui=use_gui)
        self._registry = registry
        self._logger = logger or get_structured_logger("XArmPybulletPlannedPrimitives")
        self._gripper_open = True
        self._last_execution_error: Optional[str] = None
        self._use_gui = use_gui

        # Read home position from the robot at startup so retract_gripper
        # returns to wherever the arm actually started, not a hardcoded pose.
        live = self._get_real_joint_state()
        if live is not None:
            self._home_joints: List[float] = live.tolist()
            self._logger.info("Home joints read from robot: %s",
                              [f"{v:.3f}" for v in self._home_joints])
        else:
            self._home_joints = _HOME_JOINTS
            self._logger.warning(
                "Could not read robot joint state at init — using hardcoded home: %s",
                [f"{v:.3f}" for v in self._home_joints],
            )

    def camera_pose_from_joints(self, joints: Optional[List[float]]):
        """Return `(position, Rotation)` camera pose from PyBullet FK.

        Args:
            joints: Optional joint angles in radians. If omitted, the current
                real robot joints are queried first.

        Example:
            >>> primitives.camera_pose_from_joints([0.0] * 7)
        """
        if joints is None:
            joints = self._get_real_joint_state()
        if joints is not None:
            self._planner.set_current_joint_state(joints)
        return self._planner.get_camera_transform()

    def get_robot_tcp_pose(self):
        """Return the PyBullet TCP pose after syncing from the real xArm."""
        if not self._sync_planner_to_real_robot():
            return None
        return self._planner.get_robot_tcp_pose()

    def get_camera_transform(self):
        """Return the PyBullet camera transform after syncing real joints."""
        if not self._sync_planner_to_real_robot():
            return None, None
        return self._planner.get_camera_transform()

    def scan_workspace(
        self,
        camera: Any,
        tracker: Any,
        collider: Any,
        speed: float = 0.3,
        min_points_per_label: int = 500,
        max_passes: int = 4,
    ) -> None:
        """Scan the workspace until every detected label reaches a point density threshold.

        Cycles through hardcoded top-down poses, running GSAM2 at each (with the
        existing prompt — no re-tagging) and accumulating depth into collider.
        Stops when every label has at least min_points_per_label voxel-downsampled
        points, or after max_passes full cycles through all poses.
        The robot returns to its starting joint config when done.

        Args:
            camera: RealSenseCamera (must support get_aligned_frames / get_camera_intrinsics).
            tracker: GSAM2ObjectTracker with a populated prompt (labels already known).
            collider: DepthEnvironmentCollider with start_accumulation() already called.
            speed: Joint speed for scan moves (rad/s).
            min_points_per_label: Point count target per label after voxel downsampling.
                Scanning continues until all labels exceed this or max_passes is reached.
            max_passes: Maximum number of full cycles through all scan poses.
        """
        import asyncio

        # Top-down joint configs covering left / centre / right of the work area.
        # All keep the wrist pointing straight down at a safe height (~35 cm TCP).
        # Joints in degrees: [j1, j2, j3, j4, j5, j6, j7]
        _SCAN_POSES_DEG = [
            [-25.0, -60.0, -30.0,  85.0,  0.0, 100.0, -35.0],  # left
            [ -8.0, -65.0, -25.0,  88.0,  0.0, 105.0, -35.0],  # centre
            [ 15.0, -60.0, -30.0,  85.0,  0.0, 100.0, -35.0],  # right
        ]
        scan_poses_rad = [np.deg2rad(p).tolist() for p in _SCAN_POSES_DEG]
        n_poses = len(scan_poses_rad)

        start_joints = self._get_real_joint_state()
        intrinsics = camera.get_camera_intrinsics()

        total_visits = 0
        done = False
        for pass_idx in range(max_passes):
            if done:
                break

            counts = collider.point_counts()
            unsatisfied = [
                lbl for lbl in (list(counts.keys()) or ["__background__"])
                if counts.get(lbl, 0) < min_points_per_label
            ]
            if not unsatisfied and pass_idx > 0:
                self._logger.info(
                    "scan_workspace: all labels reached density target after %d visits — stopping",
                    total_visits,
                )
                break

            self._logger.info(
                "scan_workspace: pass %d/%d — unsatisfied labels: %s",
                pass_idx + 1, max_passes, unsatisfied,
            )

            for pose_idx, pose_rad in enumerate(scan_poses_rad):
                total_visits += 1
                self._logger.info(
                    "scan_workspace: visit %d — pose %d/%d (pass %d)",
                    total_visits, pose_idx + 1, n_poses, pass_idx + 1,
                )

                if not self._set_real_joint_angles(pose_rad, wait=True, speed=speed):
                    self._logger.warning(
                        "scan_workspace: failed to reach pose %d — skipping", pose_idx + 1
                    )
                    continue

                self._planner.set_current_joint_state(pose_rad)

                color, depth = camera.get_aligned_frames()

                try:
                    asyncio.get_event_loop().run_until_complete(
                        tracker.detect_objects(
                            color_frame=color,
                            depth_frame=depth,
                            camera_intrinsics=intrinsics,
                        )
                    )
                except Exception as exc:
                    self._logger.warning(
                        "scan_workspace: detection failed at visit %d: %s", total_visits, exc
                    )

                masks = dict(getattr(tracker, "_last_masks", {}))

                cam_pos, cam_rot = self._planner.get_camera_transform()
                if cam_pos is None:
                    self._logger.warning(
                        "scan_workspace: no camera transform at visit %d", total_visits
                    )
                    continue

                T = np.eye(4)
                T[:3, :3] = cam_rot.as_matrix()
                T[:3, 3] = cam_pos

                collider.accumulate_from_depth(
                    depth_m=depth,
                    intrinsics=intrinsics,
                    masks=masks,
                    T_base_cam=T,
                )

                counts = collider.point_counts()
                self._logger.info(
                    "scan_workspace: visit %d done — point counts: %s",
                    total_visits,
                    {k: v for k, v in counts.items()},
                )

                if all(counts.get(lbl, 0) >= min_points_per_label for lbl in counts):
                    self._logger.info(
                        "scan_workspace: density target reached after visit %d — stopping",
                        total_visits,
                    )
                    done = True
                    break

        final_counts = collider.point_counts()
        for lbl, count in final_counts.items():
            status = "OK" if count >= min_points_per_label else "SPARSE"
            self._logger.info(
                "scan_workspace: [%s] %s — %d pts (target: %d)", lbl, status, count, min_points_per_label
            )

        # Return to start pose
        if start_joints is not None:
            self._logger.info("scan_workspace: returning to start pose")
            self._set_real_joint_angles(start_joints.tolist(), wait=True, speed=speed)
            self._planner.set_current_joint_state(start_joints)

    def get_robot_state(self):
        """Return a robot state dict with live joints and camera transform.

        Syncs real xArm joint state into PyBullet first so the camera
        transform reflects the current hardware pose.
        """
        self._sync_planner_to_real_robot()
        return self._planner.get_robot_state()

    def convert_cam_pose_to_base(
        self,
        position: Any,
        orientation: Any,
        do_translation: bool = True,
        debug: bool = False,
    ):
        """Convert a camera-frame pose into the xArm base frame using PyBullet.

        Args:
            position: Camera-frame position.
            orientation: Camera-frame orientation as xyzw quaternion or matrix.
            do_translation: Whether to include camera translation.
            debug: Forwarded to the PyBullet transform helper.

        Example:
            >>> primitives.convert_cam_pose_to_base([0, 0, 1], [0, 0, 0, 1])
        """
        if not self._sync_planner_to_real_robot():
            raise RuntimeError("cannot read real robot joint state")
        return self._planner.convert_cam_pose_to_base(
            position=position,
            orientation=orientation,
            do_translation=do_translation,
            debug=debug,
        )

    def move_gripper_to_pose(
        self,
        target_position: Optional[List[float]] = None,
        target_orientation: Optional[List[float]] = None,
        preset_orientation: str = "top_down",
        is_place: bool = False,
        point_label: Optional[str] = None,
        is_top_down_grasp: bool = True,
        is_side_grasp: bool = False,
        planning_dt: float = 0.02,
        max_joint_step: float = 0.05,
        speed_factor: float = _DEFAULT_SAFE_SPEED_FACTOR,
        execute: bool = True,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Plan a PyBullet IK trajectory and optionally execute it on xArm.

        Args:
            target_position: Desired TCP position in the xArm base frame.
            target_orientation: Optional xyzw TCP orientation. If omitted,
                `preset_orientation` selects a top-down or side orientation.
            speed_factor: Execution speed multiplier for waypoint timing.
            execute: If false, only returns planning metadata.

        Example:
            >>> primitives.move_gripper_to_pose([0.35, 0.0, 0.30], execute=False)
        """
        del is_top_down_grasp
        # Resolve target object ID for collision masking — the object being
        # approached should not be treated as an obstacle during planning.
        target_object_id: Optional[str] = _kwargs.get("object_id") or point_label
        ignore_labels = {target_object_id} if target_object_id else None

        if target_position is None and point_label is not None:
            target_position = self._resolve_point_label(point_label)
        if target_position is None:
            return {"success": False, "reason": "cannot determine target_position"}
        if not self._sync_planner_to_real_robot():
            return {"success": False, "reason": "cannot read real robot joint state"}

        pos = np.asarray(target_position, dtype=float).tolist()
        if is_place:
            pos[2] += 0.04
        # Clamp z to floor — depth noise can push the back-projected point slightly
        # below zero, causing every IK seed to land in collision with the table mesh.
        _MIN_GRASP_Z = 0.005
        if pos[2] < _MIN_GRASP_Z:
            self._logger.debug(
                "target z=%.4f clamped to %.4f (floor guard)", pos[2], _MIN_GRASP_Z
            )
            pos[2] = _MIN_GRASP_Z

        use_side = (preset_orientation == "side") or is_side_grasp
        seed_name = "side" if use_side else "top_down"
        if target_orientation is None:
            target_orientation = preset_quat(seed_name)

        current_joints = self._planner.get_robot_joint_state()
        current_tcp = self._planner.get_robot_tcp_pose()
        success, trajectory, dt = self._planner.move_to_pose(
            target_position=pos,
            target_orientation=target_orientation,
            planning_dt=planning_dt,
            execute=False,
            max_joint_step=max_joint_step,
            position_tolerance=_DEFAULT_POSITION_TOLERANCE_M,
            orientation_tolerance_rad=_DEFAULT_ORIENTATION_TOLERANCE_RAD,
            ignore_labels=ignore_labels,
        )

        if not success or trajectory is None:
            self._logger.info(
                "Default orientation failed — running antipodal grasp sampler (seed=%s)", seed_name
            )
            clearance_profile = _kwargs.get("clearance_profile", None)

            # Use SAM3D mesh vertices for antipodal jaw placement if available.
            object_points = None
            if target_object_id is not None and hasattr(self, "_depth_collider") \
                    and self._depth_collider is not None:
                object_points = self._depth_collider.get_object_points(target_object_id)
                if object_points is not None:
                    self._logger.info(
                        "GraspPlanner: using SAM3D mesh (%d pts) for %s",
                        len(object_points), target_object_id,
                    )

            from ptp.grasp.grasp_planner import GraspPlanner
            candidate = GraspPlanner(self._planner).plan(
                np.asarray(pos),
                object_points=object_points,
                seed_orientation=seed_name,
                clearance_profile=clearance_profile,
                ignore_labels=ignore_labels,
            )
            if candidate is not None:
                self._logger.info(
                    "GraspPlanner selected orientation angle=%.1f° seed=%s manipulability=%.4f",
                    np.degrees(candidate.approach_angle_rad),
                    candidate.seed_orientation,
                    candidate.manipulability,
                )
                target_orientation = candidate.orientation.tolist()
                success, trajectory, dt = self._planner.move_to_pose(
                    target_position=pos,
                    target_orientation=target_orientation,
                    planning_dt=planning_dt,
                    execute=False,
                    max_joint_step=max_joint_step,
                    position_tolerance=_DEFAULT_POSITION_TOLERANCE_M,
                    orientation_tolerance_rad=_DEFAULT_ORIENTATION_TOLERANCE_RAD,
                    ignore_labels=ignore_labels,
                )

        if not success or trajectory is None:
            self._logger.warning(
                "move_gripper_to_pose failed. target=%s orientation=%s "
                "current_joints=%s current_tcp=%s",
                pos, target_orientation,
                None if current_joints is None else current_joints.tolist(),
                None if current_tcp is None else current_tcp[0].tolist(),
            )
            return {
                "success": False,
                "reason": "pose planning failed",
                "target_position": pos,
                "target_orientation": list(target_orientation),
                "current_joints": None if current_joints is None else current_joints.tolist(),
                "current_tcp": None if current_tcp is None else current_tcp[0].tolist(),
            }

        if not execute:
            return {
                "success": True,
                "executed": False,
                "reason": "planned only; execute=False",
                "target_position": pos,
                "target_orientation": list(target_orientation),
                "trajectory_len": int(trajectory.shape[0]),
                "dt": dt,
                "start_joints": trajectory[0].tolist(),
                "goal_joints": trajectory[-1].tolist(),
            }

        executed = self._execute_joint_trajectory(trajectory, dt, speed_factor=speed_factor)
        execution_error = self._last_execution_error
        if execute:
            if not executed:
                return {
                    "success": False,
                    "reason": execution_error or "real robot trajectory execution failed",
                    "target_position": pos,
                    "trajectory_len": int(trajectory.shape[0]),
                    "start_joints": trajectory[0].tolist(),
                    "goal_joints": trajectory[-1].tolist(),
                }

        return {
            "success": True,
            "executed": True,
            "target_position": pos,
            "target_orientation": list(target_orientation),
            "trajectory_len": int(trajectory.shape[0]),
            "dt": dt,
            "start_joints": trajectory[0].tolist(),
            "goal_joints": trajectory[-1].tolist(),
        }

    def move_gripper_To_pose(self, **kwargs: Any) -> Dict[str, Any]:
        """Backward-compatible alias for legacy primitive calls."""
        return self.move_gripper_to_pose(**kwargs)

    def push(
        self,
        distance: float = 0.08,
        force_direction: str = "forward",
        speed_factor: float = _DEFAULT_SAFE_SPEED_FACTOR,
        execute: bool = True,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Push along the requested base-frame axis from the current TCP pose."""
        ignore_labels = self._displaceable_ignore_labels()
        if ignore_labels:
            self._logger.info("push: ignoring displaceable meshes %s", sorted(ignore_labels))
        return self._cartesian_delta_motion(
            direction=self._force_direction_to_vector(force_direction),
            distance=distance,
            label="push",
            speed_factor=speed_factor,
            execute=execute,
            ignore_labels=ignore_labels,
        )

    def pull(
        self,
        distance: float = 0.08,
        force_direction: str = "forward",
        speed_factor: float = _DEFAULT_SAFE_SPEED_FACTOR,
        execute: bool = True,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Pull opposite the requested base-frame axis from the current TCP pose."""
        ignore_labels = self._displaceable_ignore_labels()
        if ignore_labels:
            self._logger.info("pull: ignoring displaceable meshes %s", sorted(ignore_labels))
        return self._cartesian_delta_motion(
            direction=-self._force_direction_to_vector(force_direction),
            distance=distance,
            label="pull",
            speed_factor=speed_factor,
            execute=execute,
            ignore_labels=ignore_labels,
        )

    def pivot_pull(
        self,
        pivot_point: Optional[List[float]] = None,
        arc_angle_deg: float = 90.0,
        segments: int = 18,
        speed_mm_s: float = 30.0,
        execute: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute an arc motion around a hinge point via direct Cartesian waypoints.

        The TCP traces a true circular arc centred on the hinge, with the gripper
        orientation updated at each step to keep it pointing toward the pivot.
        Waypoints are streamed with wait=False on intermediate steps for smooth
        continuous motion (no controller resets between steps).

        Rotation direction is derived geometrically: the positive rotation around
        the hinge axis that moves the TCP away from the surface (pull) is chosen
        automatically; no hardcoded hinge-location overrides needed.

        Falls back to a single straight-line cartesian move when no robot adapter
        supports cartesian_arc (e.g. simulation).
        """
        from scipy.spatial.transform import Rotation

        if not self._sync_planner_to_real_robot():
            return {"success": False, "reason": "cannot read real robot joint state"}
        tcp = self._planner.get_robot_tcp_pose()
        if tcp is None:
            return {"success": False, "reason": "cannot read tcp pose"}
        tcp_pos, tcp_quat = tcp

        _meta = metadata or {}

        # ── Resolve pivot point ────────────────────────────────────────────────
        if pivot_point is None:
            meta_pivot = _meta.get("pivot_point_base")
            pivot_point = list(meta_pivot) if meta_pivot is not None else (
                tcp_pos + np.array([0.0, -0.10, 0.0])
            ).tolist()

        pivot = np.asarray(pivot_point, dtype=float)
        self._logger.info(
            "pivot_pull: tcp_pos=%s  pivot=%s  lever=%.3fm  surface_normal=%s",
            [f"{v:.3f}" for v in tcp_pos],
            [f"{v:.3f}" for v in pivot],
            float(np.linalg.norm(tcp_pos - pivot)),
            [f"{v:.3f}" for v in _meta["surface_normal_base"]] if _meta.get("surface_normal_base") is not None else None,
        )

        # ── Derive hinge rotation axis ─────────────────────────────────────────
        # The hinge axis is perpendicular to both the lever arm (pivot→TCP) and
        # the surface normal.  When no normal is available, default to world Z
        # (vertical hinge — most common for cabinet doors).
        lever = tcp_pos - pivot
        lever_norm = float(np.linalg.norm(lever))
        if lever_norm < 1e-4:
            return {"success": False, "reason": "TCP coincides with pivot point"}

        surface_normal = _normalise_vec(_meta.get("surface_normal_base"))
        if surface_normal is not None:
            hinge_axis = np.cross(lever, surface_normal)
            hinge_axis_norm = float(np.linalg.norm(hinge_axis))
            if hinge_axis_norm < 1e-4:
                # lever and normal are parallel — fall back to world Z
                hinge_axis = np.array([0.0, 0.0, 1.0])
            else:
                hinge_axis = hinge_axis / hinge_axis_norm
        else:
            hinge_axis = np.array([0.0, 0.0, 1.0])

        # ── Choose rotation sign ───────────────────────────────────────────────
        # Try +angle first: if it moves the TCP away from the surface (dot product
        # with INTO-surface normal < 0) use it; otherwise negate.
        # Convention: +normal points INTO the surface (camera faces outward), so
        # moving away from the surface means dot(delta, normal) < 0.
        test_rot = Rotation.from_rotvec(hinge_axis * math.radians(5.0))
        test_pos = pivot + test_rot.apply(lever)
        if surface_normal is not None:
            sign = 1.0 if float(np.dot(test_pos - tcp_pos, surface_normal)) < 0 else -1.0
        else:
            # No normal — pick the direction that decreases X (pull away from wall)
            sign = 1.0 if test_pos[0] < tcp_pos[0] else -1.0

        arc_rad = math.radians(float(arc_angle_deg)) * sign

        self._logger.info(
            "pivot_pull: hinge_axis=%s  sign=%+.0f  arc=%.1f°  segments=%d  speed=%.0fmm/s  "
            "test_delta_dot_normal=%.3f",
            [f"{v:.3f}" for v in hinge_axis],
            sign,
            arc_angle_deg * sign,
            segments,
            speed_mm_s,
            float(np.dot(test_pos - tcp_pos, surface_normal)) if surface_normal is not None else float("nan"),
        )

        # ── Generate arc waypoints ─────────────────────────────────────────────
        waypoints: List[tuple] = []
        tcp_rot = Rotation.from_quat(tcp_quat)
        for i in range(1, segments + 1):
            t = i / segments
            rot = Rotation.from_rotvec(hinge_axis * arc_rad * t)
            pos = pivot + rot.apply(lever)
            # Rotate the gripper orientation by the same amount so it stays
            # tangent to the arc (pointing toward/away from the hinge).
            quat = (rot * tcp_rot).as_quat()  # xyzw
            waypoints.append((pos.tolist(), quat.tolist()))

        self._logger.info(
            "pivot_pull: %d waypoints generated  start=%s  end=%s",
            len(waypoints),
            [f"{v:.3f}" for v in waypoints[0][0]] if waypoints else "—",
            [f"{v:.3f}" for v in waypoints[-1][0]] if waypoints else "—",
        )
        for i, (pos, quat) in enumerate(waypoints):
            self._logger.info(
                "  wp[%02d/%02d]  pos=[%s]  quat(xyzw)=[%s]",
                i + 1, len(waypoints),
                ", ".join(f"{v:.4f}" for v in pos),
                ", ".join(f"{v:.4f}" for v in quat),
            )

        if not execute:
            return {
                "success": True,
                "executed": False,
                "waypoints": [(p, q) for p, q in waypoints],
                "arc_angle_deg": arc_angle_deg,
            }

        cartesian_arc = getattr(self._robot, "cartesian_arc", None)
        if cartesian_arc is None:
            return {"success": False, "reason": "robot adapter does not support cartesian_arc"}

        ok = cartesian_arc(waypoints, speed=speed_mm_s)
        return {
            "success": ok,
            "executed": True,
            "arc_angle_deg": arc_angle_deg,
            "segments": segments,
            "final_position": waypoints[-1][0] if waypoints else None,
        }

    def push(
        self,
        surface_label: str = "",
        is_button: bool = False,
        action_goal: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Move the gripper into a surface along the negative surface normal.

        When ``metadata`` contains ``surface_normal_base`` (injected by the
        executor's Molmo grounding pipeline), the gripper moves along -normal.
        Falls back to pressing straight down if the normal is unavailable.
        """
        del action_goal
        metadata = metadata or {}
        normal_base = _normalise_vec(metadata.get("surface_normal_base"))
        dist = 0.05 if is_button else 0.08
        self._logger.info(
            "push: surface=%s  surface_normal_base=%s  confidence=%.2f  is_button=%s",
            surface_label,
            [f"{v:.3f}" for v in metadata["surface_normal_base"]] if metadata.get("surface_normal_base") is not None else None,
            metadata.get("surface_normal_confidence", 0.0),
            is_button,
        )

        if normal_base is not None:
            move_dir = normal_base  # +normal points into the surface (camera faces outward)
            confidence = metadata.get("surface_normal_confidence", 0.0)
            self._logger.info(
                "push: surface=%s  dir=+normal=[%.2f,%.2f,%.2f]  confidence=%.2f  distance=%.3fm%s",
                surface_label, move_dir[0], move_dir[1], move_dir[2],
                confidence, dist, "  (button)" if is_button else "",
            )
            result = self._cartesian_delta_motion_direct(
                direction=move_dir,
                distance=dist,
                label=f"push ({surface_label})",
                execute=kwargs.get("execute", True),
            )
        else:
            self._logger.info(
                "push: surface=%s  dir=down (fallback)  distance=%.3fm%s",
                surface_label, dist, "  (button)" if is_button else "",
            )
            result = self._cartesian_delta_motion_direct(
                direction=np.array([0.0, 0.0, -1.0]),
                distance=dist,
                label=f"push fallback ({surface_label})",
                execute=kwargs.get("execute", True),
            )

        if is_button and result.get("success") and kwargs.get("execute", True):
            retract_dir = -normal_base if normal_base is not None else np.array([0.0, 0.0, 1.0])
            self._cartesian_delta_motion_direct(
                direction=retract_dir,
                distance=0.03,
                label=f"push button-release ({surface_label})",
                execute=kwargs.get("execute", True),
            )
        return result

    def pull(
        self,
        surface_label: str = "",
        has_pivot: bool = False,
        action_goal: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Move the gripper away from a surface along the positive surface normal.

        When ``has_pivot`` is True, dispatches to ``pivot_pull`` using the
        Molmo-grounded hinge point from ``metadata``. Otherwise moves along
        +normal, falling back to pulling straight back (-X) if unavailable.
        """
        del action_goal
        metadata = metadata or {}
        normal_base = _normalise_vec(metadata.get("surface_normal_base"))
        self._logger.info(
            "pull: surface=%s  surface_normal_base=%s  confidence=%.2f  has_pivot=%s",
            surface_label,
            [f"{v:.3f}" for v in metadata["surface_normal_base"]] if metadata.get("surface_normal_base") is not None else None,
            metadata.get("surface_normal_confidence", 0.0),
            has_pivot,
        )

        if has_pivot:
            pivot_point_base = metadata.get("pivot_point_base")

            # Compute the arc angle needed to sweep the door fully open (90°).
            # The gripper starts at the handle; we rotate until the lever arm is
            # perpendicular to its starting direction, i.e. exactly 90°.
            # If the pivot radius is known, we verify via arc length:
            #   arc_length = radius * angle_rad  → angle_rad = arc_length / radius
            # But since we always target 90° regardless of radius, and the pivot
            # radius is already the correct lever arm, we use 90° directly.
            # The sign (direction) is resolved inside pivot_pull from the surface normal.
            arc_angle_deg = 90.0

            self._logger.info(
                "pull: surface=%s  mode=pivot  hinge=%s  radius=%.3fm  arc=%.1f°",
                surface_label,
                (
                    f"[{pivot_point_base[0]:.3f},{pivot_point_base[1]:.3f},{pivot_point_base[2]:.3f}]"
                    if pivot_point_base is not None else "estimated"
                ),
                metadata.get("pivot_radius_m") or float("nan"),
                arc_angle_deg,
            )
            return self.pivot_pull(
                pivot_point=pivot_point_base,
                arc_angle_deg=arc_angle_deg,
                metadata=metadata,
                execute=kwargs.get("execute", True),
                speed_mm_s=kwargs.get("speed_mm_s", 30.0),
            )

        if normal_base is not None:
            move_dir = -normal_base  # -normal points away from the surface (camera faces outward)
            confidence = metadata.get("surface_normal_confidence", 0.0)
            self._logger.info(
                "pull: surface=%s  dir=-normal=[%.2f,%.2f,%.2f]  confidence=%.2f  distance=0.080m",
                surface_label, move_dir[0], move_dir[1], move_dir[2], confidence,
            )
            return self._cartesian_delta_motion_direct(
                direction=move_dir,
                distance=0.08,
                label=f"pull ({surface_label})",
                execute=kwargs.get("execute", True),
            )

        self._logger.info(
            "pull: surface=%s  dir=-X (fallback)  distance=0.080m", surface_label,
        )
        return self._cartesian_delta_motion_direct(
            direction=np.array([-1.0, 0.0, 0.0]),
            distance=0.08,
            label=f"pull fallback ({surface_label})",
            execute=kwargs.get("execute", True),
        )

    def _displaceable_ignore_labels(self) -> Optional[set]:
        """Return object IDs that are displaceable and can be ignored during push collision checks."""
        return None

    def _lateral_push_direction(self, surface_label: str) -> Optional[np.ndarray]:
        """Return a unit vector from the current TCP toward the object in XY."""
        tcp = self._planner.get_robot_tcp_pose() if self._planner else None
        if tcp is None:
            return None
        tcp_pos = tcp[0]
        obj_pos = self._resolve_point_label(surface_label)
        if obj_pos is None:
            return None
        delta = np.asarray(obj_pos, float) - np.asarray(tcp_pos, float)
        delta[2] = 0.0  # lateral only — ignore Z
        norm = float(np.linalg.norm(delta))
        if norm < 1e-6:
            return None
        return delta / norm

    def twist(
        self,
        direction: str = "clockwise",
        rotation_angle_deg: float = 360.0,
        speed: float = _DEFAULT_SAFE_JOINT_SPEED,
        execute: bool = True,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Twist by commanding the final xArm wrist joint."""
        joints = self._get_real_joint_state()
        if joints is None or len(joints) == 0:
            return {"success": False, "reason": "cannot read real robot joints"}
        joints = np.asarray(joints, dtype=float).copy()
        delta = math.radians(float(rotation_angle_deg))
        if direction == "counterclockwise":
            delta *= -1.0
        joints[-1] += delta
        if execute and not self._set_real_joint_angles(joints.tolist(), wait=True, speed=speed):
            return {"success": False, "reason": "real robot twist command failed"}
        self._planner.set_current_joint_state(joints)
        return {"success": True, "executed": execute, "joint_index": int(len(joints) - 1), "delta_rad": float(delta)}

    def open_gripper(self, **kwargs: Any) -> Dict[str, Any]:
        """Open the real xArm gripper."""
        self._gripper_open = True
        ok = self._robot.open_gripper(**kwargs)
        self._planner.open_gripper()
        return {"success": bool(ok)}

    def close_gripper(self, **kwargs: Any) -> Dict[str, Any]:
        """Close the real xArm gripper."""
        self._gripper_open = False
        ok = self._robot.close_gripper(**kwargs)
        self._planner.close_gripper()
        return {"success": bool(ok)}

    def retract_gripper(
        self,
        speed: float = _DEFAULT_SAFE_JOINT_SPEED,
        execute: bool = True,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Move the real xArm back to the home joint state captured at startup."""
        if execute and not self._set_real_joint_angles(self._home_joints, wait=True, speed=speed):
            return {"success": False, "reason": "real robot retract command failed"}
        self._planner.set_current_joint_state(self._home_joints)
        return {"success": True, "executed": execute}

    def _cartesian_delta_motion(
        self,
        direction: np.ndarray,
        distance: float,
        label: str,
        speed_factor: float,
        execute: bool,
        ignore_labels: Optional[set] = None,
    ) -> Dict[str, Any]:
        if not self._sync_planner_to_real_robot():
            return {"success": False, "reason": "cannot read real robot joint state"}
        tcp = self._planner.get_robot_tcp_pose()
        if tcp is None:
            return {"success": False, "reason": "cannot read tcp pose"}
        pos, quat = tcp
        unit = direction / max(float(np.linalg.norm(direction)), 1e-8)
        return self._plan_and_execute_pose(
            target_position=(pos + unit * float(distance)).tolist(),
            target_orientation=quat.tolist(),
            label=label,
            speed_factor=speed_factor,
            execute=execute,
            ignore_labels=ignore_labels,
        )

    def _cartesian_delta_motion_direct(
        self,
        direction: np.ndarray,
        distance: float,
        label: str,
        speed_mm_s: float = 30.0,
        execute: bool = True,
    ) -> Dict[str, Any]:
        """Send a Cartesian delta directly to the xArm — no PyBullet planning.

        Used for push/pull where the gripper is already in contact with the
        surface and a collision-free joint-space plan will always fail.
        """
        if not self._sync_planner_to_real_robot():
            return {"success": False, "reason": "cannot read real robot joint state"}
        tcp = self._planner.get_robot_tcp_pose()
        if tcp is None:
            return {"success": False, "reason": "cannot read tcp pose"}
        pos, quat = tcp
        unit = direction / max(float(np.linalg.norm(direction)), 1e-8)
        target_pos = (pos + unit * float(distance)).tolist()

        if not execute:
            return {"success": True, "executed": False, "target_position": target_pos}

        cartesian_move = getattr(self._robot, "cartesian_move", None)
        if cartesian_move is None:
            return {"success": False, "reason": "robot adapter does not support cartesian_move"}

        ok = cartesian_move(
            position=target_pos,
            orientation_quat_xyzw=quat.tolist(),
            speed=speed_mm_s,
        )
        if not ok:
            return {"success": False, "reason": f"{label} cartesian move failed"}
        return {"success": True, "executed": True, "target_position": target_pos}

    def _plan_and_execute_pose(
        self,
        target_position: List[float],
        target_orientation: List[float],
        label: str,
        speed_factor: float,
        execute: bool,
        ignore_labels: Optional[set] = None,
    ) -> Dict[str, Any]:
        success, trajectory, dt = self._planner.move_to_pose(
            target_position=target_position,
            target_orientation=target_orientation,
            execute=False,
            position_tolerance=_DEFAULT_POSITION_TOLERANCE_M,
            orientation_tolerance_rad=_DEFAULT_ORIENTATION_TOLERANCE_RAD,
            ignore_labels=ignore_labels,
        )
        if not success or trajectory is None:
            return {"success": False, "reason": f"{label} planning failed"}
        if not execute:
            return {
                "success": True,
                "executed": False,
                "reason": "planned only; execute=False",
                "target_position": target_position,
                "trajectory_len": int(trajectory.shape[0]),
                "dt": dt,
                "start_joints": trajectory[0].tolist(),
                "goal_joints": trajectory[-1].tolist(),
            }
        if not self._execute_joint_trajectory(trajectory, dt, speed_factor=speed_factor):
            return {
                "success": False,
                "reason": self._last_execution_error or f"{label} execution failed",
                "target_position": target_position,
                "trajectory_len": int(trajectory.shape[0]),
                "start_joints": trajectory[0].tolist(),
                "goal_joints": trajectory[-1].tolist(),
            }
        return {
            "success": True,
            "executed": True,
            "target_position": target_position,
            "trajectory_len": int(trajectory.shape[0]),
            "dt": dt,
            "start_joints": trajectory[0].tolist(),
            "goal_joints": trajectory[-1].tolist(),
        }

    def _execute_joint_trajectory(
        self,
        trajectory: np.ndarray,
        dt: Optional[float],
        speed_factor: float = _DEFAULT_SAFE_SPEED_FACTOR,
    ) -> bool:
        self._last_execution_error = None
        frame_dt = (float(dt) if dt is not None else 0.02) / max(float(speed_factor), 1e-6)
        waypoints = np.asarray(trajectory, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[0] == 0:
            self._last_execution_error = "empty or malformed trajectory"
            return False

        if hasattr(self._robot, "set_robot_joint_angles"):
            start_time = time.time()
            last_idx = len(waypoints) - 1
            for idx, waypoint in enumerate(waypoints):
                target_t = idx * frame_dt
                elapsed = time.time() - start_time
                if elapsed < target_t:
                    time.sleep(target_t - elapsed)
                is_last = idx == last_idx
                if not self._robot.set_robot_joint_angles(
                    waypoint.tolist(),
                    wait=is_last,
                    speed=_DEFAULT_SAFE_JOINT_SPEED,
                ):
                    self._last_execution_error = f"set_robot_joint_angles failed at waypoint {idx}"
                    self._logger.warning(self._last_execution_error)
                    return False
                if self._use_gui:
                    self._planner.set_current_joint_state(waypoint.tolist())
                if idx % max(1, len(waypoints) // 5) == 0 or is_last:
                    self._logger.info(
                        "Executed waypoint %d/%d",
                        idx + 1,
                        len(waypoints),
                    )
            self._planner.set_current_joint_state(waypoints[-1])
            try:
                self._robot.set_current_joint_state(waypoints[-1])
            except Exception:
                pass
            return True

        arm = getattr(self._robot, "arm", None)
        arm_lock = getattr(self._robot, "arm_lock", None)
        if arm is not None:
            try:
                if arm_lock is not None:
                    arm_lock.acquire()
                arm.set_mode(0)
                arm.set_state(0)
                time.sleep(0.1)
                start_time = time.time()
                last_idx = len(waypoints) - 1
                for idx, waypoint in enumerate(waypoints):
                    target_t = idx * frame_dt
                    elapsed = time.time() - start_time
                    if elapsed < target_t:
                        time.sleep(target_t - elapsed)
                    code = arm.set_servo_angle(
                        angle=waypoint.tolist(),
                        speed=_DEFAULT_SAFE_JOINT_SPEED,
                        mvacc=_DEFAULT_SAFE_JOINT_ACCEL,
                        is_radian=True,
                        wait=(idx == last_idx),
                    )
                    if code != 0:
                        self._last_execution_error = (
                            f"xArm set_servo_angle failed at waypoint {idx}; code={code}"
                        )
                        self._logger.warning(self._last_execution_error)
                        return False
                    if self._use_gui:
                        self._planner.set_current_joint_state(waypoint.tolist())
            finally:
                if arm_lock is not None:
                    arm_lock.release()
        else:
            self._last_execution_error = "robot has neither set_robot_joint_angles nor arm"
            return False

        if arm is None:
            for waypoint in waypoints:
                if not self._set_real_joint_angles(waypoint.tolist(), wait=True):
                    self._last_execution_error = "real robot joint command failed"
                    return False
                if frame_dt > 0.0:
                    time.sleep(frame_dt)

        self._planner.set_current_joint_state(waypoints[-1])
        try:
            self._robot.set_current_joint_state(waypoints[-1])
        except Exception:
            pass
        return True

    def _get_real_joint_state(self) -> Optional[np.ndarray]:
        joints = self._robot.get_robot_joint_state()
        if joints is None:
            return None
        joints = np.asarray(joints, dtype=float).reshape(-1)
        if joints.size == 0:
            return None
        return joints[:7]

    def _sync_planner_to_real_robot(self) -> bool:
        joints = self._get_real_joint_state()
        if joints is None:
            return False
        self._planner.set_current_joint_state(joints)
        if self._gripper_open:
            self._planner.open_gripper()
        else:
            self._planner.close_gripper()
        self._planner.get_robot_tcp_pose()
        return True

    def _set_real_joint_angles(
        self,
        joints: List[float],
        wait: bool = True,
        speed: float = _DEFAULT_SAFE_JOINT_SPEED,
    ) -> bool:
        if hasattr(self._robot, "set_robot_joint_angles"):
            return bool(self._robot.set_robot_joint_angles(joints, wait=wait, speed=speed))
        arm = getattr(self._robot, "arm", None)
        if arm is None:
            return False
        code = arm.set_servo_angle(
            angle=joints,
            speed=speed,
            mvacc=_DEFAULT_SAFE_JOINT_ACCEL,
            is_radian=True,
            wait=wait,
        )
        return code == 0

    def _force_direction_to_vector(self, force_direction: str) -> np.ndarray:
        mapping = {
            "forward": np.array([1.0, 0.0, 0.0], dtype=float),
            "backward": np.array([-1.0, 0.0, 0.0], dtype=float),
            "left": np.array([0.0, 1.0, 0.0], dtype=float),
            "right": np.array([0.0, -1.0, 0.0], dtype=float),
            "up": np.array([0.0, 0.0, 1.0], dtype=float),
            "down": np.array([0.0, 0.0, -1.0], dtype=float),
            "perpendicular": np.array([0.0, 0.0, -1.0], dtype=float),
            "parallel": np.array([1.0, 0.0, 0.0], dtype=float),
        }
        return mapping.get(force_direction, np.array([1.0, 0.0, 0.0], dtype=float))

    def _resolve_point_label(self, label: str) -> Optional[List[float]]:
        if self._registry is None:
            return None
        obj_id, _, point_name = label.partition("/")
        obj = self._registry.get_object(obj_id)
        if obj is None:
            return None
        if point_name and getattr(obj, "interaction_points", None):
            ip = obj.interaction_points.get(point_name)
            if ip is not None and ip.position_3d is not None:
                return list(ip.position_3d)
        if getattr(obj, "position_3d", None) is not None:
            return list(obj.position_3d)
        return None
