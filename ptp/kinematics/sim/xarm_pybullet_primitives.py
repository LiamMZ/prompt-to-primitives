"""
PyBullet primitive library for xArm using the BasePybulletInterface planner.

This class mirrors the high-level primitive API used by the CuRobo-backed stack,
but plans trajectories with `BasePybulletInterface.move_to_pose()` and applies
the resulting final joint state to the in-memory PyBullet robot model.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from ptp.kinematics.xarm_pybullet_interface import XArmPybulletInterface
from ptp.utils.logging_utils import get_structured_logger

_HOME_JOINTS_DEG = [-8.1, -75.3, -24.9, 88.0, -7.6, 116.2, -34.9]
_HOME_JOINTS = np.deg2rad(_HOME_JOINTS_DEG).tolist()
_DEFAULT_POSITION_TOLERANCE_M = 0.07
_DEFAULT_ORIENTATION_TOLERANCE_RAD = 0.6


class XArmPybulletPrimitives:
    """Primitive interface backed by `XArmPybulletInterface` motion planning."""

    def __init__(
        self,
        robot: Optional[XArmPybulletInterface] = None,
        registry: Optional[Any] = None,
        env: Optional[Any] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self._robot = robot or XArmPybulletInterface()
        self._registry = registry
        self._env = env
        self._logger = logger or get_structured_logger("XArmPybulletPrimitives")
        self._gripper_open = True

    def camera_pose_from_joints(self, joints: Optional[List[float]]):
        """Return (position, Rotation) camera pose for a provided joint state."""
        if joints is not None:
            self._robot.set_current_joint_state(joints)
        return self._robot.get_camera_transform()

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
        visualization_dt: Optional[float] = None,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Move TCP to a target pose using PyBullet IK trajectory planning."""
        del is_top_down_grasp  # retained for API compatibility
        if target_position is None and point_label is not None:
            target_position = self._resolve_point_label(point_label)
        if target_position is None:
            return {"success": False, "reason": "cannot determine target_position"}

        pos = np.asarray(target_position, dtype=float).tolist()
        if is_place:
            pos[2] += 0.04

        if target_orientation is None:
            use_side = (preset_orientation == "side") or is_side_grasp
            target_orientation = [0.0, 0.707, 0.0, 0.707] if use_side else [0.0, 1.0, 0.0, 0.0]

        success, trajectory, dt = self._robot.move_to_pose(
            target_position=pos,
            target_orientation=target_orientation,
            planning_dt=planning_dt,
            execute=False,
            max_joint_step=max_joint_step,
            position_tolerance=_DEFAULT_POSITION_TOLERANCE_M,
            orientation_tolerance_rad=_DEFAULT_ORIENTATION_TOLERANCE_RAD,
        )
        if (not success or trajectory is None) and target_orientation is not None:
            # Fallback: preserve reachability by relaxing orientation constraints.
            success, trajectory, dt = self._robot.move_to_pose(
                target_position=pos,
                target_orientation=None,
                planning_dt=planning_dt,
                execute=False,
                max_joint_step=max_joint_step,
                position_tolerance=_DEFAULT_POSITION_TOLERANCE_M,
            )
        if not success or trajectory is None:
            self._logger.warning("move_gripper_to_pose failed for target %s", pos)
            return {"success": False, "reason": "pose planning failed"}
        self._visualize_trajectory(
            trajectory,
            dt=visualization_dt if visualization_dt is not None else dt,
            label="move_gripper_to_pose",
        )

        return {
            "success": True,
            "target_position": pos,
            "target_orientation": list(target_orientation),
            "trajectory_len": int(trajectory.shape[0]),
            "dt": dt,
        }

    # Backward-compatible alias used by some legacy callers.
    def move_gripper_To_pose(self, **kwargs: Any) -> Dict[str, Any]:
        return self.move_gripper_to_pose(**kwargs)

    def push(
        self,
        distance: float = 0.08,
        force_direction: str = "forward",
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Push along the requested axis from the current TCP pose."""
        return self._cartesian_delta_motion(
            direction=self._force_direction_to_vector(force_direction),
            distance=distance,
            label="push",
        )

    def pull(
        self,
        distance: float = 0.08,
        force_direction: str = "forward",
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Pull opposite the requested axis from the current TCP pose."""
        return self._cartesian_delta_motion(
            direction=-self._force_direction_to_vector(force_direction),
            distance=distance,
            label="pull",
        )

    def pivot_pull(
        self,
        pivot_point: Optional[List[float]] = None,
        pull_distance: float = 0.10,
        arc_angle_deg: float = 25.0,
        direction: str = "counterclockwise",
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Approximate pivot-pull by blending tangential arc and radial pull."""
        tcp = self._robot.get_robot_tcp_pose()
        if tcp is None:
            return {"success": False, "reason": "cannot read tcp pose"}
        tcp_pos, tcp_quat = tcp

        if pivot_point is None:
            pivot_point = (tcp_pos + np.array([0.0, -0.10, 0.0])).tolist()
        pivot = np.asarray(pivot_point, dtype=float)
        radius_vec = tcp_pos - pivot
        radius_xy = radius_vec.copy()
        radius_xy[2] = 0.0
        radius_norm = float(np.linalg.norm(radius_xy))
        if radius_norm < 1e-6:
            return {"success": False, "reason": "pivot radius too small"}

        radial_hat = radius_xy / radius_norm
        tangent_hat = np.array([-radial_hat[1], radial_hat[0], 0.0], dtype=float)
        if direction == "clockwise":
            tangent_hat *= -1.0
        arc_len = math.radians(float(arc_angle_deg)) * radius_norm
        delta = tangent_hat * arc_len - radial_hat * float(pull_distance)
        target_pos = (tcp_pos + delta).tolist()

        success, trajectory, dt = self._robot.move_to_pose(
            target_position=target_pos,
            target_orientation=tcp_quat.tolist(),
            execute=False,
            position_tolerance=_DEFAULT_POSITION_TOLERANCE_M,
            orientation_tolerance_rad=_DEFAULT_ORIENTATION_TOLERANCE_RAD,
        )
        if not success or trajectory is None:
            return {"success": False, "reason": "pivot_pull planning failed"}
        self._visualize_trajectory(trajectory, dt=dt, label="pivot_pull")
        return {"success": True, "target_position": target_pos, "trajectory_len": int(trajectory.shape[0]), "dt": dt}

    def push_pull(
        self,
        surface_label: str,
        force_direction: str = "perpendicular",
        is_button: bool = False,
        has_pivot: bool = False,
        hinge_location: Optional[str] = None,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Compatibility primitive that dispatches to push/pull/pivot_pull."""
        del surface_label, hinge_location  # currently unused without scene geometry
        if has_pivot:
            result = self.pivot_pull()
        elif force_direction == "perpendicular":
            result = self.push(distance=0.05 if is_button else 0.08, force_direction="down")
        elif force_direction == "parallel":
            result = self.pull(distance=0.08, force_direction="forward")
        else:
            result = self.push(distance=0.08, force_direction=force_direction)

        if is_button and result.get("success"):
            self.pull(distance=0.03, force_direction="down")
        return result

    def twist(
        self,
        direction: str = "clockwise",
        rotation_angle_deg: float = 90.0,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Twist by rotating the final wrist joint and writing back joints."""
        joints = self._robot.get_robot_joint_state()
        if joints is None or len(joints) == 0:
            return {"success": False, "reason": "cannot read joints"}
        joints = joints.copy()
        delta = math.radians(float(rotation_angle_deg))
        if direction == "clockwise":
            delta *= -1.0
        joints[-1] += delta
        self._robot.set_current_joint_state(joints)
        return {"success": True, "joint_index": int(len(joints) - 1), "delta_rad": float(delta)}

    def open_gripper(self, **_kwargs: Any) -> Dict[str, Any]:
        self._gripper_open = True
        self._robot.open_gripper()
        return {"success": True}

    def close_gripper(self, **_kwargs: Any) -> Dict[str, Any]:
        self._gripper_open = False
        self._robot.close_gripper()
        return {"success": True}

    def retract_gripper(self, **_kwargs: Any) -> Dict[str, Any]:
        self._robot.set_current_joint_state(_HOME_JOINTS)
        return {"success": True}

    def _cartesian_delta_motion(
        self,
        direction: np.ndarray,
        distance: float,
        label: str,
    ) -> Dict[str, Any]:
        tcp = self._robot.get_robot_tcp_pose()
        if tcp is None:
            return {"success": False, "reason": "cannot read tcp pose"}
        pos, quat = tcp
        unit = direction / max(float(np.linalg.norm(direction)), 1e-8)
        target_pos = (pos + unit * float(distance)).tolist()
        success, trajectory, dt = self._robot.move_to_pose(
            target_position=target_pos,
            target_orientation=quat.tolist(),
            execute=False,
            position_tolerance=_DEFAULT_POSITION_TOLERANCE_M,
            orientation_tolerance_rad=_DEFAULT_ORIENTATION_TOLERANCE_RAD,
        )
        if not success or trajectory is None:
            return {"success": False, "reason": f"{label} planning failed"}
        self._visualize_trajectory(trajectory, dt=dt, label=label)
        return {"success": True, "target_position": target_pos, "trajectory_len": int(trajectory.shape[0]), "dt": dt}

    def _visualize_trajectory(self, trajectory: np.ndarray, dt: Optional[float], label: str) -> None:
        """Replay trajectory in GUI env when available."""
        if self._env is None:
            return
        frame_dt = float(dt) if dt is not None else 0.02
        frame_dt = max(0.005, frame_dt)
        try:
            self._env.set_status(f"Executing: {label}")
        except Exception:
            pass
        for waypoint in trajectory:
            joints = np.asarray(waypoint, dtype=float).tolist()
            self._robot.set_current_joint_state(joints)
            self._robot.get_robot_tcp_pose()
            try:
                # Single-robot mode: viewer and planner are the same interface.
                if hasattr(self._env, "set_current_joint_state"):
                    self._env.set_current_joint_state(joints)
                elif hasattr(self._env, "set_robot_joints"):
                    self._env.set_robot_joints(joints)
                self._env.step(frame_dt)
            except Exception:
                # GUI is best-effort only; planning state should still progress.
                pass

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
