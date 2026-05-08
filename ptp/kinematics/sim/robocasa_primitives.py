"""
RobocasaPrimitives — PRIMITIVE_LIBRARY implementation for robocasa/MuJoCo.

Exposes the same method signatures as PyBulletPrimitives so that
PrimitiveExecutor can call them via getattr(primitives, name)(**params).

Primitives:
  move_gripper_to_pose  — set a delta-position action toward the target
  push_pull             — push the EEF toward a surface label
  open_gripper          — send gripper-open action
  close_gripper         — send gripper-close action
  retract_gripper       — step the env back to the neutral pose
  twist                 — small wrist rotation action

All primitives call env.step() internally and return a result dict.

The action space for PandaOmron is 12-dim:
  [0:3]  EEF delta position (x, y, z)
  [3:6]  EEF delta orientation (axis-angle or euler, depending on controller)
  [6]    gripper open/close  (+1 open, -1 close)
  [7:10] base delta position (omni-directional base)
  [10:12] base rotation
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np


_ACTION_DIM = 12
_EEF_SPEED  = 0.05   # metres per step (scale for delta-pos actions)
_N_STEPS    = 20     # interpolation steps for smooth motion


def _zero_action(action_dim: int = _ACTION_DIM) -> np.ndarray:
    return np.zeros(action_dim, dtype=np.float64)


class RobocasaPrimitives:
    """
    Robocasa action-space primitive interface.

    Args:
        env:           Live robocasa environment (already reset).
        registry:      DetectedObjectRegistry for resolving point/surface labels.
        robot_iface:   RobocasaRobotInterface for reading current EEF state.
        step_sleep:    Seconds to sleep between env.step calls (0 for max speed).
        verbose:       Print step-by-step info.
    """

    def __init__(
        self,
        env: object,
        registry: Any,
        robot_iface: Any,
        step_sleep: float = 0.0,
        verbose: bool = False,
    ) -> None:
        self._env = env
        self._registry = registry
        self._robot = robot_iface
        self._step_sleep = step_sleep
        self._verbose = verbose
        self._last_obs: Optional[dict] = None
        self._action_dim = getattr(env, "action_dim", _ACTION_DIM)

    # ------------------------------------------------------------------
    # Frame injection
    # ------------------------------------------------------------------

    def update(self, obs: dict) -> None:
        self._last_obs = obs
        self._robot.update(obs)

    # ------------------------------------------------------------------
    # camera_pose_from_joints  (PrimitiveExecutor hook)
    # ------------------------------------------------------------------

    def camera_pose_from_joints(self, joints: Optional[List[float]]):
        """Return (position, Rotation) for the eye-in-hand camera."""
        return self._robot.get_camera_transform()

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def move_gripper_to_pose(
        self,
        target_position: Optional[List[float]] = None,
        preset_orientation: str = "top_down",
        is_place: bool = False,
        point_label: Optional[str] = None,
        is_top_down_grasp: bool = True,
        is_side_grasp: bool = False,
        **_: Any,
    ) -> Dict[str, Any]:
        """Move the EEF toward target_position using delta-position actions."""
        if target_position is None and point_label is not None:
            target_position = self._resolve_point_label(point_label)

        if target_position is None:
            return {"success": False, "reason": "no target_position"}

        target = np.array(target_position, dtype=float)
        if is_place:
            target[2] += 0.04

        obs = self._last_obs
        for step_i in range(_N_STEPS):
            current_pos, _ = self._robot.get_robot_tcp_pose()
            if current_pos is None:
                break
            delta = target - current_pos
            dist  = np.linalg.norm(delta)
            if dist < 0.01:
                break
            # Clamp each step to _EEF_SPEED
            step = delta / dist * min(dist, _EEF_SPEED)
            action = _zero_action(self._action_dim)
            action[0:3] = step
            obs, _, done, _ = self._env.step(action)
            self._last_obs = obs
            self._robot.update(obs)
            if self._step_sleep:
                time.sleep(self._step_sleep)
            if done:
                break

        if self._verbose:
            print(f"  move_gripper_to_pose → {target_position}")
        return {"success": True, "target_position": list(target_position)}

    def push_pull(
        self,
        surface_label: str,
        force_direction: str = "perpendicular",
        is_button: bool = False,
        has_pivot: bool = False,
        hinge_location: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Push the EEF toward a surface/object label."""
        surface_pos = self._resolve_point_label(surface_label)
        if surface_pos is None:
            return {"success": False, "reason": f"unknown surface_label '{surface_label}'"}

        current_pos, _ = self._robot.get_robot_tcp_pose()
        if current_pos is None:
            return {"success": False, "reason": "cannot get TCP pose"}

        target = np.array(surface_pos, dtype=float)
        if force_direction == "perpendicular":
            target[2] -= 0.05   # push down into surface
        else:
            to_surface = target - current_pos
            to_surface[2] = 0.0
            n = np.linalg.norm(to_surface)
            if n > 1e-6:
                target = current_pos + to_surface / n * 0.10

        result = self.move_gripper_to_pose(target_position=target.tolist())

        if is_button:
            self.move_gripper_to_pose(target_position=list(current_pos))

        return result

    def open_gripper(self, wait: bool = True, timeout: float = 5.0, **_: Any) -> Dict[str, Any]:
        """Send gripper-open action for several steps."""
        action = _zero_action(self._action_dim)
        action[6] = 1.0   # gripper open
        for _ in range(5):
            obs, _, done, _ = self._env.step(action)
            self._last_obs = obs
            self._robot.update(obs)
            if self._step_sleep:
                time.sleep(self._step_sleep)
        return {"success": True}

    def close_gripper(
        self,
        wait: bool = True,
        timeout: float = 5.0,
        simple_close: bool = True,
        **_: Any,
    ) -> Dict[str, Any]:
        """Send gripper-close action for several steps."""
        action = _zero_action(self._action_dim)
        action[6] = -1.0  # gripper close
        for _ in range(5):
            obs, _, done, _ = self._env.step(action)
            self._last_obs = obs
            self._robot.update(obs)
            if self._step_sleep:
                time.sleep(self._step_sleep)
        return {"success": True}

    def retract_gripper(
        self, distance: float = 0.05, speed_factor: float = 1.0, **_: Any
    ) -> Dict[str, Any]:
        """Retract EEF upward and back toward a neutral position."""
        current_pos, _ = self._robot.get_robot_tcp_pose()
        if current_pos is None:
            return {"success": False, "reason": "cannot get TCP pose"}
        # Lift straight up
        up_target = current_pos.copy()
        up_target[2] += distance
        self.move_gripper_to_pose(target_position=up_target.tolist())
        return {"success": True}

    def twist(
        self,
        direction: str = "clockwise",
        rotation_angle_deg: float = 90.0,
        speed_factor: float = 1.0,
        timeout: float = 5.0,
        **_: Any,
    ) -> Dict[str, Any]:
        """Apply a small wrist rotation via orientation delta action."""
        import math
        delta_rad = math.radians(rotation_angle_deg)
        sign = -1.0 if direction == "clockwise" else 1.0

        action = _zero_action(self._action_dim)
        action[5] = sign * delta_rad / _N_STEPS  # roll around EEF z

        for _ in range(_N_STEPS):
            obs, _, done, _ = self._env.step(action)
            self._last_obs = obs
            self._robot.update(obs)
            if self._step_sleep:
                time.sleep(self._step_sleep)
        return {"success": True}

    # ------------------------------------------------------------------
    # Label resolution
    # ------------------------------------------------------------------

    def _resolve_point_label(self, label: str) -> Optional[List[float]]:
        """Resolve an object_id or object_id/point_name to a world position."""
        obj_id, _, point_name = label.partition("/")

        # Try registry first
        obj = self._registry.get_object(obj_id)
        if obj is not None:
            if point_name and obj.interaction_points:
                ip = obj.interaction_points.get(point_name)
                if ip is not None and ip.position_3d is not None:
                    return list(ip.position_3d)
            if obj.position_3d is not None:
                return list(obj.position_3d)

        # Fallback: ground-truth sim position
        return self._ground_truth_position(obj_id)

    def _ground_truth_position(self, name: str) -> Optional[List[float]]:
        """Read object position directly from MuJoCo sim."""
        try:
            sim = self._env.sim
            obj = self._env.objects.get(name)
            if obj is None:
                return None
            body_id = sim.model.body_name2id(obj.root_body)
            return sim.data.body_xpos[body_id].tolist()
        except Exception:
            return None
