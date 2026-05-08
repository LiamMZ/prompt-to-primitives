"""
PyBullet-backed implementation of the AMP (Atomic Manipulation Primitive) library.

Exposes the same method names as the CuRobo interface so that PrimitiveExecutor
can call them directly via getattr(primitives, primitive.name)(**params).

Primitives implemented (matching skill_plan_types.PRIMITIVE_LIBRARY):
  - move_gripper_to_pose  : IK-solve to a contact point from the Interaction Map
  - push_pull             : constrained motion along/about a Surface Map label
  - open_gripper          : visual gripper open (sim only)
  - close_gripper         : visual gripper close + object attachment
  - retract_gripper       : return arm to home configuration
  - twist                 : wrist rotation about EEF Z axis

The SceneEnvironment is used for all PyBullet calls.  The object registry
(DetectedObjectRegistry) is used to resolve point_label / surface_label
references to 3-D positions at execution time.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    p = None

from ptp.utils.logging_utils import get_structured_logger


# Home joint configuration (camera-aim pose from scene_environment)
_HOME_JOINTS = [0.100085, -1.407677, -0.098652, 1.314592, 0.0, 2.0, -0.112296]

# Interpolation steps for smooth motion visualisation
_MOTION_STEPS = 30
_STEP_SLEEP   = 1.0 / 60.0


class PyBulletPrimitives:
    """
    AMP library backed by PyBullet IK + SceneEnvironment.

    Args:
        env:      SceneEnvironment instance (GUI client).
        registry: DetectedObjectRegistry used to resolve interaction/surface labels.
        logger:   Optional logger.
    """

    def __init__(
        self,
        env: Any,                    # SceneEnvironment
        registry: Any,               # DetectedObjectRegistry
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._env = env
        self._registry = registry
        self._logger = logger or get_structured_logger("PyBulletPrimitives")

        # State
        self._held_object: Optional[str] = None   # object_id currently attached to EEF
        self._gripper_open: bool = True

    # ------------------------------------------------------------------ #
    # Public helper: camera_pose_from_joints (used by PrimitiveExecutor)  #
    # ------------------------------------------------------------------ #

    def camera_pose_from_joints(
        self, joints: Optional[List[float]]
    ) -> Tuple[Optional[np.ndarray], Optional[Any]]:
        """Return (position, Rotation) of the wrist camera for given joint config."""
        if joints is not None:
            self._env.set_robot_joints(joints)
        return self._env.get_camera_transform()

    # ------------------------------------------------------------------ #
    # Primitives                                                           #
    # ------------------------------------------------------------------ #

    def move_gripper_to_pose(
        self,
        target_position: Optional[List[float]] = None,
        preset_orientation: str = "top_down",
        is_place: bool = False,
        # Fallback resolution params
        point_label: Optional[str] = None,
        is_top_down_grasp: bool = True,
        is_side_grasp: bool = False,
        # Absorb any extra executor params without error
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Move the gripper end-effector to a target position.

        Distinguished from navigate_to_pose (base/mobile platform motion, added later).

        target_position is preferred (set by PrimitiveExecutor after back-projecting
        target_pixel_yx from the snapshot).  If absent, point_label is resolved
        against the registry as a fallback (useful in sim without a real snapshot).

        preset_orientation: 'top_down' or 'side'.
        is_place: when True, adds a small upward clearance offset.
        """
        # --- resolve target position ---
        if target_position is None and point_label is not None:
            target_position = self._resolve_point_label(point_label)

        if target_position is None:
            self._logger.warning("move_gripper_to_pose: no target_position and no resolvable point_label")
            return {"success": False, "reason": "cannot determine target position"}

        # Orientation: normalise both old and new style params
        side = (preset_orientation == "side") or is_side_grasp
        if side:
            target_orn = [0.0, 0.707, 0.0, 0.707]  # 90° pitch → side grasp
        else:
            target_orn = [0.0, 1.0, 0.0, 0.0]       # 180° pitch → top-down

        pos = list(target_position)
        if is_place:
            pos[2] = pos[2] + 0.04   # small clearance above place surface

        # Approach with standoff then move to contact
        standoff = (np.array(pos) + np.array([0.0, 0.0, 0.06])).tolist()
        label = point_label or str(pos)
        self._logger.info("move_gripper_to_pose: orientation=%s → %s", preset_orientation, pos)
        self._move_to(standoff, target_orn, label=f"approach {label}")
        self._move_to(pos,      target_orn, label=f"contact  {label}")

        if not self._gripper_open:
            self._try_attach(pos)

        return {"success": True, "target_position": pos}

    def push_pull(
        self,
        surface_label: str,
        force_direction: str = "perpendicular",
        is_button: bool = False,
        has_pivot: bool = False,
        hinge_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Push or pull relative to a Surface Map label.

        In sim this moves the EEF along/into the resolved surface normal.
        """
        surface_pos = self._resolve_surface_label(surface_label)
        if surface_pos is None:
            self._logger.warning("push_pull: cannot resolve surface '%s'", surface_label)
            return {"success": False, "reason": f"unknown surface_label '{surface_label}'"}

        current_tcp = self._env.get_robot_tcp_pose()
        if current_tcp is None:
            return {"success": False, "reason": "cannot get TCP pose"}

        current_pos, current_orn = current_tcp
        surface_normal = np.array([0.0, 0.0, 1.0])   # assume upward-facing surface

        if force_direction == "perpendicular":
            direction = -surface_normal  # push into surface
            distance  = 0.05 if not is_button else 0.02
        else:  # parallel
            # Move parallel to surface toward its centre
            to_surface = np.array(surface_pos) - current_pos
            to_surface[2] = 0.0
            norm = np.linalg.norm(to_surface)
            direction = (to_surface / norm) if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
            distance  = 0.1

        target_pos = (current_pos + direction * distance).tolist()
        target_orn = current_orn.tolist()

        self._logger.info("push_pull: '%s' dir=%s dist=%.3f", surface_label, force_direction, distance)
        self._move_to(target_pos, target_orn, label=f"push_pull {surface_label}")

        if is_button:
            # Retract immediately
            self._move_to(current_pos.tolist(), target_orn, label="button retract")

        return {"success": True, "target_position": target_pos}

    def open_gripper(
        self,
        wait: bool = True,
        timeout: float = 5.0,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Open the gripper and detach any held object."""
        self._logger.info("open_gripper")
        self._gripper_open = True

        if self._held_object is not None:
            # Drop the held object at the current EEF position
            tcp = self._env.get_robot_tcp_pose()
            if tcp is not None:
                drop_pos = tcp[0].tolist()
                drop_pos[2] = max(0.01, drop_pos[2] - 0.03)   # rest on surface
                self._env.move_object(self._held_object, drop_pos)
                self._logger.info("open_gripper: released '%s' at %s", self._held_object, drop_pos)
            self._held_object = None

        self._env.step(0.2)
        return {"success": True}

    def close_gripper(
        self,
        wait: bool = True,
        timeout: float = 5.0,
        simple_close: bool = True,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Close the gripper and attach the nearest object if in contact range."""
        self._logger.info("close_gripper")
        self._gripper_open = False

        tcp = self._env.get_robot_tcp_pose()
        if tcp is not None:
            self._try_attach(tcp[0].tolist())

        self._env.step(0.2)
        return {"success": True, "grasped": self._held_object}

    def retract_gripper(
        self,
        distance: float = 0.05,
        speed_factor: float = 1.0,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Return the arm to the home/camera-aim joint configuration."""
        self._logger.info("retract_gripper: moving to home")

        # If holding an object, keep it attached during retraction
        n_arm = len(self._env._arm_joints) if self._env._arm_joints else len(_HOME_JOINTS)
        raw_start = self._env.get_robot_joint_state()
        if raw_start is None:
            start = np.array(_HOME_JOINTS, dtype=float)
        else:
            # Slice to arm joints only — gripper joints are not controlled here
            start = np.array(raw_start, dtype=float)[:n_arm]

        target = np.array(_HOME_JOINTS, dtype=float)

        for i in range(_MOTION_STEPS):
            alpha = (i + 1) / _MOTION_STEPS
            joints = (start + alpha * (target - start)).tolist()
            self._env.set_robot_joints(joints)
            if self._held_object:
                self._attach_object_to_eef()
            self._env.step(0.02 / max(speed_factor, 0.1))

        return {"success": True}

    def twist(
        self,
        direction: str = "clockwise",
        rotation_angle_deg: float = 90.0,
        speed_factor: float = 1.0,
        timeout: float = 5.0,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Rotate the final wrist joint by *rotation_angle_deg* degrees, then return.

        direction: 'clockwise' → negative joint increment,
                   'counterclockwise' → positive.
        """
        joints = self._env.get_robot_joint_state()
        if joints is None:
            return {"success": False, "reason": "cannot read joint state"}

        joints = joints.tolist()
        rotation_angle = math.radians(rotation_angle_deg)
        delta = -rotation_angle if direction == "clockwise" else rotation_angle
        start_angle = joints[-1]
        target_angle = start_angle + delta

        self._logger.info("twist: %s %.1f° (joint[-1]: %.2f → %.2f)",
                          direction, rotation_angle_deg, start_angle, target_angle)

        # Forward stroke
        for i in range(_MOTION_STEPS):
            alpha = (i + 1) / _MOTION_STEPS
            joints[-1] = start_angle + alpha * delta
            self._env.set_robot_joints(joints)
            if self._held_object:
                self._attach_object_to_eef()
            self._env.step(0.02 / max(speed_factor, 0.1))

        # Return stroke
        for i in range(_MOTION_STEPS):
            alpha = (i + 1) / _MOTION_STEPS
            joints[-1] = target_angle - alpha * delta
            self._env.set_robot_joints(joints)
            if self._held_object:
                self._attach_object_to_eef()
            self._env.step(0.02 / max(speed_factor, 0.1))

        return {"success": True, "final_angle": start_angle}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _move_to(
        self,
        target_pos: List[float],
        target_orn: List[float],
        label: str = "",
    ) -> None:
        """IK-solve for target_pos/orn and interpolate the robot joints."""
        if not PYBULLET_AVAILABLE or self._env._robot_id is None:
            return

        c      = self._env._client
        robot  = self._env._robot_id
        tcp_idx = self._env._link_name_to_index.get("link_tcp")
        if tcp_idx is None:
            self._logger.warning("_move_to: link_tcp not found")
            return

        ik_joints = p.calculateInverseKinematics(
            robot,
            tcp_idx,
            target_pos,
            target_orn,
            physicsClientId=c,
        )

        # Slice to arm joints only — gripper joints are not IK-controlled
        n_arm = len(self._env._arm_joints) if self._env._arm_joints else len(_HOME_JOINTS)
        target = np.array(ik_joints[:n_arm])

        raw_start = self._env.get_robot_joint_state()
        if raw_start is None:
            start = np.array(_HOME_JOINTS)
        else:
            # get_robot_joint_state may return all joints (arm + gripper); slice to arm only
            start = np.array(raw_start)[:n_arm]

        for i in range(_MOTION_STEPS):
            alpha = (i + 1) / _MOTION_STEPS
            joints = (start + alpha * (target - start)).tolist()
            self._env.set_robot_joints(joints)
            if self._held_object:
                self._attach_object_to_eef()
            p.stepSimulation(physicsClientId=c)
            time.sleep(_STEP_SLEEP)

    def _attach_object_to_eef(self) -> None:
        """Teleport the held object to the current EEF (TCP) position."""
        if self._held_object is None:
            return
        tcp = self._env.get_robot_tcp_pose()
        if tcp is not None:
            pos = tcp[0].tolist()
            self._env.move_object(self._held_object, pos)

    def _try_attach(self, eef_pos: List[float], attach_radius: float = 0.08) -> None:
        """Attach the nearest object within attach_radius of eef_pos."""
        if self._held_object is not None:
            return
        eef = np.array(eef_pos)
        best_id, best_dist = None, float("inf")
        for obj in self._registry.get_all_objects():
            obj_pos = self._env.get_object_position(obj.object_id)
            if obj_pos is None:
                continue
            dist = float(np.linalg.norm(eef - np.array(obj_pos)))
            if dist < attach_radius and dist < best_dist:
                best_dist = dist
                best_id = obj.object_id
        if best_id:
            self._held_object = best_id
            self._logger.info("_try_attach: attached '%s' (dist=%.3f)", best_id, best_dist)

    def _resolve_point_label(self, label: str) -> Optional[List[float]]:
        """
        Resolve an Interaction Map label to a 3-D position.

        Accepted formats:
          - "<object_id>"                  → object centroid
          - "<object_id>/<point_name>"     → specific interaction point
        """
        obj_id, _, point_name = label.partition("/")
        obj = self._registry.get_object(obj_id)
        if obj is None:
            return None

        if point_name and obj.interaction_points:
            ip = obj.interaction_points.get(point_name)
            if ip is not None and ip.position_3d is not None:
                return list(ip.position_3d)

        if obj.position_3d is not None:
            return list(obj.position_3d)

        # Fall back to sim object position
        return self._env.get_object_position(obj_id)

    def _resolve_surface_label(self, label: str) -> Optional[List[float]]:
        """Resolve a Surface Map label to a 3-D position (same logic as point label)."""
        return self._resolve_point_label(label)
