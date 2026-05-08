"""
PyBullet-backed AMP (Atomic Manipulation Primitive) library for Hello Robot Stretch.

Mirrors the interface of PyBulletPrimitives but targets the Stretch RE1V0 URDF.

Primitives:
  - navigate_to          : RRT-planned base navigation with collision avoidance
  - move_gripper_to_pose : IK-solve arm to a contact point with collision checking
  - open_gripper         : detach held object
  - close_gripper        : attach nearest object
  - retract_gripper      : return arm to home configuration
"""

from __future__ import annotations

import logging
import math
import random
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

# Stretch movable joint ordering (verified from URDF, 12 total):
#  [0]  joint_right_wheel          (R) PyBullet joint idx 0
#  [1]  joint_left_wheel           (R) PyBullet joint idx 1
#  [2]  joint_lift                 (P) PyBullet joint idx 4
#  [3]  joint_arm_l3               (P) PyBullet joint idx 6
#  [4]  joint_arm_l2               (P) PyBullet joint idx 7
#  [5]  joint_arm_l1               (P) PyBullet joint idx 8
#  [6]  joint_arm_l0               (P) PyBullet joint idx 9
#  [7]  joint_wrist_yaw            (R) PyBullet joint idx 10
#  [8]  joint_gripper_finger_left  (R) PyBullet joint idx 13
#  [9]  joint_gripper_finger_right (R) PyBullet joint idx 15
#  [10] joint_head_pan             (R) PyBullet joint idx 22
#  [11] joint_head_tilt            (R) PyBullet joint idx 23

# Home configuration for all 12 movable joints (n_arm_joints=12):
# wheels=0, lift=0.65m, arm retracted, wrist=0, gripper open,
# head_pan=0 (looks along +X), head_tilt=-0.5 (looking slightly down)
_HOME_JOINTS = [
    0.0,    # [0]  right_wheel
    0.0,    # [1]  left_wheel
    0.65,   # [2]  lift
    0.0,    # [3]  arm_l3
    0.0,    # [4]  arm_l2
    0.0,    # [5]  arm_l1
    0.0,    # [6]  arm_l0
    0.0,    # [7]  wrist_yaw
    0.0,    # [8]  gripper_finger_left
    0.0,    # [9]  gripper_finger_right
    0.0,    # [10] head_pan
    -0.5,   # [11] head_tilt
]

_MOTION_STEPS = 30
_STEP_SLEEP   = 1.0 / 60.0

# Joint indices within _HOME_JOINTS / the arm control array
_IDX_LIFT      = 2
_IDX_ARM_L3    = 3   # outermost arm segment
_IDX_ARM_L0    = 6   # innermost arm segment (extends furthest)
_IDX_WRIST_YAW = 7
_IDX_HEAD_PAN  = 10
_IDX_HEAD_TILT = 11

# Collision checking
_COLLISION_MARGIN  = 0.02   # metres — stop if any robot link is within this of an obstacle
_RRT_MAX_ITER      = 2000   # max RRT tree expansions
_RRT_STEP_SIZE     = 0.1    # metres per RRT extension step
_RRT_GOAL_BIAS     = 0.15   # probability of sampling goal directly
_BASE_RADIUS       = 0.28   # Stretch base footprint radius (metres)

# Horizontal arm workspace: 4 segments × 0.13 m each (from URDF joint limits)
_ARM_REACH_MIN = 0.0    # fully retracted (metres from base to EEF along -Y)
_ARM_REACH_MAX = 0.52   # fully extended  (4 × 0.13 m)
_ARM_REACH_MARGIN = 0.05  # keep this much headroom from the full-extension limit


class StretchPyBulletPrimitives:
    """
    AMP library backed by PyBullet IK + SceneEnvironment, targeting the Stretch RE1V0.

    All motion primitives include collision checking via PyBullet's
    getClosestPoints API.  navigate_to plans the base path using a 2-D RRT
    that treats the base as a disc of radius _BASE_RADIUS.  move_gripper_to_pose
    checks collision at every interpolated arm step and aborts on contact.

    Args:
        env:      SceneEnvironment instance (GUI client, loaded with Stretch URDF).
        registry: DetectedObjectRegistry used to resolve interaction/surface labels.
        logger:   Optional logger.
    """

    def __init__(
        self,
        env: Any,
        registry: Any,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._env = env
        self._registry = registry
        self._logger = logger or get_structured_logger("StretchPyBulletPrimitives")

        self._held_object: Optional[str] = None
        self._gripper_open: bool = True

    # ------------------------------------------------------------------ #
    # Primitives                                                           #
    # ------------------------------------------------------------------ #

    def navigate_to(
        self,
        target_position: Optional[List[float]] = None,
        target_yaw: float = 0.0,
        point_label: Optional[str] = None,
        standoff_distance: float = 0.55,
        head_tilt: float = -0.5,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Navigate the Stretch base to a goal position using RRT collision avoidance.

        Plans a 2-D RRT path from the robot's current base position to a standoff
        point in front of the target, checking each step for collisions using
        PyBullet's getClosestPoints.  Each waypoint is executed by teleporting the
        base body (resetBasePositionAndOrientation) and stepping the simulation.
        After reaching the goal the head camera is re-aimed toward the target.

        Args:
            target_position: [x, y] or [x, y, z] world position to navigate toward.
                             If None, resolved from point_label.
            target_yaw:      Desired final heading in radians (0 = +X forward).
                             Used as the base orientation at the goal.
            point_label:     Registry label to resolve if target_position is None.
            standoff_distance: Stop this far from the target (metres).
            head_tilt:       Head tilt angle after navigation.
        """
        if not PYBULLET_AVAILABLE or self._env._robot_id is None:
            return {"success": False, "reason": "PyBullet not available"}

        # Resolve target
        target_xyz: Optional[List[float]] = None
        if target_position is not None:
            target_xyz = list(target_position)
        elif point_label is not None:
            target_xyz = self._resolve_point_label(point_label)
            if target_xyz is None:
                self._logger.warning("navigate_to: cannot resolve '%s'", point_label)
                return {"success": False, "reason": f"unknown label '{point_label}'"}

        # Aim head toward target regardless of whether base can move
        head_pan = _HOME_JOINTS[_IDX_HEAD_PAN]
        if target_xyz is not None:
            head_pan = math.atan2(target_xyz[1], target_xyz[0])

        # ---- Compute goal XY (standoff in front of target) ----
        if target_xyz is not None:
            goal_xy = self._compute_standoff_xy(target_xyz, standoff_distance)
        else:
            # No target — just re-aim head
            self._set_head(head_pan, head_tilt)
            self._env.step(0.3)
            return {"success": True, "reason": "no target, head aimed only"}

        # ---- Get current base position ----
        start_xy = self._get_base_xy()

        # Skip navigation if already close enough
        if np.linalg.norm(np.array(goal_xy) - np.array(start_xy)) < 0.05:
            self._set_head(head_pan, head_tilt)
            self._env.step(0.3)
            return {"success": True, "path_length": 0, "waypoints": 0}

        # ---- Plan RRT path ----
        self._logger.info(
            "navigate_to: RRT from %s to %s (standoff=%.2fm)",
            [round(v, 2) for v in start_xy],
            [round(v, 2) for v in goal_xy],
            standoff_distance,
        )
        path = self._rrt_plan_base(start_xy, goal_xy, target_yaw)

        if path is None:
            self._logger.warning(
                "navigate_to: RRT failed — executing direct move as fallback"
            )
            path = [start_xy, goal_xy]

        # ---- Execute path ----
        for wp in path[1:]:  # skip start (already there)
            wp_yaw = math.atan2(wp[1] - start_xy[1], wp[0] - start_xy[0])
            self._teleport_base(wp, wp_yaw)
            self._set_head(head_pan, head_tilt)
            if self._held_object:
                self._attach_object_to_eef()
            self._env.step(0.05)
            start_xy = wp

        # Final heading at goal
        self._teleport_base(goal_xy, target_yaw)
        self._set_head(head_pan, head_tilt)
        self._env.step(0.3)

        self._logger.info(
            "navigate_to: arrived at %s (yaw=%.2f, %d waypoints)",
            [round(v, 2) for v in goal_xy], target_yaw, len(path)
        )
        return {
            "success": True,
            "goal_position": goal_xy + [0.0],
            "yaw": target_yaw,
            "waypoints": len(path),
        }

    def move_gripper_to_pose(
        self,
        target_position: Optional[List[float]] = None,
        preset_orientation: str = "top_down",
        is_place: bool = False,
        point_label: Optional[str] = None,
        is_top_down_grasp: bool = True,
        is_side_grasp: bool = False,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Align the Stretch base then extend the arm to reach a target position.

        Stretch has a single-axis horizontal linear actuator: the arm can only
        extend along the robot's local -Y direction.  Before moving the arm the
        base must be:
          1. Rotated so the robot's -Y axis points toward the target XY.
          2. Positioned so the arm extension axis passes through the target
             (i.e. placed directly in front of the target along the -Y axis
             at a distance equal to the arm's reach, ~0.52 m).

        The sequence is:
          a) Compute the required base yaw (angle so -Y faces the target).
          b) Compute the required base XY (standoff along the +Y side of target
             so arm extension along -Y reaches the target).
          c) Navigate the base to that aligned position using RRT.
          d) Extend the arm via IK with per-step collision checking.
        """
        if target_position is None and point_label is not None:
            target_position = self._resolve_point_label(point_label)

        if target_position is None:
            self._logger.warning("move_gripper_to_pose: no target_position")
            return {"success": False, "reason": "cannot determine target position"}

        side = (preset_orientation == "side") or is_side_grasp
        target_orn = [0.0, 0.707, 0.0, 0.707] if side else [0.0, 1.0, 0.0, 0.0]

        pos = list(target_position)
        if is_place:
            pos[2] += 0.04

        label = point_label or str(pos)
        self._logger.info("move_gripper_to_pose: aligning base then extending to %s", pos)

        # ------------------------------------------------------------------
        # a–c) Find the closest base position on the arm's alignment axis
        #      that keeps the target within the arm's horizontal workspace,
        #      then navigate there.
        # ------------------------------------------------------------------
        target_xy = np.array(pos[:2])
        aligned_xy, base_yaw = self._find_aligned_base_position(target_xy)

        if aligned_xy is None:
            self._logger.warning(
                "move_gripper_to_pose: cannot find a collision-free aligned base position"
            )
            return {"success": False, "reason": "cannot align base"}

        self._logger.info(
            "move_gripper_to_pose: base yaw=%.2f rad, aligned_xy=%s",
            base_yaw, [round(v, 3) for v in aligned_xy],
        )

        start_xy = self._get_base_xy()
        if np.linalg.norm(np.array(aligned_xy) - np.array(start_xy)) > 0.05:
            path = self._rrt_plan_base(start_xy, aligned_xy, goal_yaw=base_yaw)
            if path is None:
                self._logger.warning(
                    "move_gripper_to_pose: RRT failed to reach aligned base position"
                )
                return {"success": False, "reason": "cannot align base (RRT failed)"}

            for wp in path[1:]:
                self._teleport_base(wp, base_yaw)
                if self._held_object:
                    self._attach_object_to_eef()
                self._env.step(0.05)

        # Lock in final aligned position and yaw, aim head toward target
        self._teleport_base(aligned_xy, base_yaw)
        head_pan = math.atan2(
            pos[1] - aligned_xy[1],
            pos[0] - aligned_xy[0],
        )
        self._set_head(head_pan, _HOME_JOINTS[_IDX_HEAD_TILT])
        self._env.step(0.2)

        # ------------------------------------------------------------------
        # d) Extend arm: approach (standoff +6 cm above) then contact
        # ------------------------------------------------------------------
        standoff = (np.array(pos) + np.array([0.0, 0.0, 0.06])).tolist()

        ok = self._move_to(standoff, target_orn, label=f"approach {label}")
        if not ok:
            self._logger.warning("move_gripper_to_pose: collision on approach — retracting")
            self.retract_gripper()
            return {"success": False, "reason": "collision on approach"}

        ok = self._move_to(pos, target_orn, label=f"contact {label}")
        if not ok:
            self._logger.warning("move_gripper_to_pose: collision on contact — retracting")
            self.retract_gripper()
            return {"success": False, "reason": "collision on contact"}

        if not self._gripper_open:
            self._try_attach(pos)

        return {"success": True, "target_position": pos, "base_yaw": base_yaw}

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
            tcp = self._env.get_robot_tcp_pose()
            if tcp is not None:
                drop_pos = tcp[0].tolist()
                drop_pos[2] = max(0.01, drop_pos[2] - 0.03)
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
        n_arm = len(self._env._arm_joints) if self._env._arm_joints else len(_HOME_JOINTS)
        raw_start = self._env.get_robot_joint_state()
        start = np.array(raw_start, dtype=float)[:n_arm] if raw_start is not None else np.array(_HOME_JOINTS, dtype=float)
        target = np.array(_HOME_JOINTS, dtype=float)

        for i in range(_MOTION_STEPS):
            alpha = (i + 1) / _MOTION_STEPS
            joints = (start + alpha * (target - start)).tolist()
            self._env.set_robot_joints(joints)
            if self._held_object:
                self._attach_object_to_eef()
            self._env.step(0.02 / max(speed_factor, 0.1))

        return {"success": True}

    # ------------------------------------------------------------------ #
    # Collision checking                                                   #
    # ------------------------------------------------------------------ #

    def _scene_obstacle_ids(self) -> List[int]:
        """Return PyBullet body IDs for all static scene objects, excluding held object."""
        ids = []
        for oid, bid in self._env._obj_ids.items():
            if oid != self._held_object:
                ids.append(bid)
        return ids

    def _robot_in_collision(self, margin: float = _COLLISION_MARGIN) -> bool:
        """
        Return True if any robot link is closer than `margin` metres to any
        scene obstacle, using PyBullet getClosestPoints.

        Ignores the ground plane (body 0).
        """
        if not PYBULLET_AVAILABLE:
            return False
        c     = self._env._client
        robot = self._env._robot_id
        for obs_id in self._scene_obstacle_ids():
            pts = p.getClosestPoints(
                robot, obs_id,
                distance=margin,
                physicsClientId=c,
            )
            if pts:
                return True
        return False

    def _base_disc_in_collision(
        self,
        xy: List[float],
        yaw: float,
        margin: float = _COLLISION_MARGIN,
    ) -> bool:
        """
        Check whether the robot base disc at xy collides with any scene obstacle.

        Uses a temporary sphere body at the candidate base position rather than
        teleporting the whole robot (which would check every link including the arm).
        The sphere radius equals _BASE_RADIUS + margin.
        """
        if not PYBULLET_AVAILABLE:
            return False
        c = self._env._client

        radius = _BASE_RADIUS + margin
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=c)
        probe = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            basePosition=[xy[0], xy[1], 0.15],   # z=0.15 ≈ base centre height
            physicsClientId=c,
        )

        colliding = False
        for obs_id in self._scene_obstacle_ids():
            pts = p.getClosestPoints(probe, obs_id, distance=0.0, physicsClientId=c)
            if pts and any(pt[8] <= 0.0 for pt in pts):
                colliding = True
                break

        p.removeBody(probe, physicsClientId=c)
        return colliding

    # ------------------------------------------------------------------ #
    # RRT base planner                                                     #
    # ------------------------------------------------------------------ #

    def _rrt_plan_base(
        self,
        start_xy: List[float],
        goal_xy: List[float],
        goal_yaw: float = 0.0,
    ) -> Optional[List[List[float]]]:
        """
        Plan a collision-free 2-D path for the robot base using RRT.

        Each node is an [x, y] position.  Collision is checked by teleporting
        the robot to each candidate position and calling _base_disc_in_collision.

        Returns the smoothed path as a list of [x, y] waypoints from start to
        goal, or None if no path is found within _RRT_MAX_ITER iterations.
        """
        start = list(start_xy[:2])
        goal  = list(goal_xy[:2])

        # Quick check: is goal itself collision-free?
        if self._base_disc_in_collision(goal, goal_yaw):
            self._logger.warning("RRT: goal position is in collision")
            return None

        # RRT tree: list of [x, y] nodes, parent index
        nodes:   List[List[float]] = [start]
        parents: List[int]         = [-1]

        # Bounding box for random sampling (padded around start↔goal)
        xs = [start[0], goal[0]]
        ys = [start[1], goal[1]]
        pad = max(2.0, np.linalg.norm(np.array(goal) - np.array(start)) * 0.5)
        x_min, x_max = min(xs) - pad, max(xs) + pad
        y_min, y_max = min(ys) - pad, max(ys) + pad

        for _ in range(_RRT_MAX_ITER):
            # Sample: goal-biased
            if random.random() < _RRT_GOAL_BIAS:
                sample = goal
            else:
                sample = [
                    random.uniform(x_min, x_max),
                    random.uniform(y_min, y_max),
                ]

            # Nearest node
            dists   = [np.linalg.norm(np.array(n) - np.array(sample)) for n in nodes]
            nearest_idx = int(np.argmin(dists))
            nearest = nodes[nearest_idx]

            # Steer
            direction = np.array(sample) - np.array(nearest)
            dist = float(np.linalg.norm(direction))
            if dist < 1e-6:
                continue
            direction /= dist
            new_xy = (np.array(nearest) + direction * min(_RRT_STEP_SIZE, dist)).tolist()

            # Yaw = direction of travel
            step_yaw = math.atan2(direction[1], direction[0])

            # Collision check at the new node
            if self._base_disc_in_collision(new_xy, step_yaw):
                continue

            # Add to tree
            new_idx = len(nodes)
            nodes.append(new_xy)
            parents.append(nearest_idx)

            # Check if we've reached the goal
            if np.linalg.norm(np.array(new_xy) - np.array(goal)) < _RRT_STEP_SIZE:
                # Reconstruct path
                path = [goal]
                idx = new_idx
                while idx != -1:
                    path.append(nodes[idx])
                    idx = parents[idx]
                path.reverse()
                return self._smooth_path(path)

        return None  # RRT exhausted

    def _smooth_path(self, path: List[List[float]]) -> List[List[float]]:
        """
        Greedy path shortcutting: repeatedly try to skip intermediate waypoints
        by checking whether a direct segment between non-adjacent nodes is
        collision-free.
        """
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            # Try to connect directly to the furthest reachable node
            j = len(path) - 1
            while j > i + 1:
                if self._segment_collision_free(path[i], path[j]):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed

    def _segment_collision_free(
        self,
        a: List[float],
        b: List[float],
        n_checks: int = 5,
    ) -> bool:
        """Check n_checks evenly spaced points along the segment a→b for collisions."""
        for k in range(1, n_checks + 1):
            alpha = k / (n_checks + 1)
            pt = [a[0] + alpha * (b[0] - a[0]), a[1] + alpha * (b[1] - a[1])]
            yaw = math.atan2(b[1] - a[1], b[0] - a[0])
            if self._base_disc_in_collision(pt, yaw):
                return False
        return True

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def camera_pose_from_joints(
        self, joints: Optional[List[float]]
    ) -> Tuple[Optional[np.ndarray], Optional[Any]]:
        if joints is not None:
            self._env.set_robot_joints(joints)
        return self._env.get_camera_transform()

    def _move_to(
        self,
        target_pos: List[float],
        target_orn: List[float],
        label: str = "",
    ) -> bool:
        """
        IK-solve for target_pos/orn and interpolate the arm joints.

        Checks for collisions (robot vs scene obstacles) at every step.
        Stops immediately and returns False if a collision is detected;
        returns True if the motion completed without collision.
        """
        if not PYBULLET_AVAILABLE or self._env._robot_id is None:
            return True  # no sim — treat as success

        c     = self._env._client
        robot = self._env._robot_id
        eef_link = self._env._link_name_to_index.get("link_grasp_center")
        if eef_link is None:
            self._logger.warning("_move_to: link_grasp_center not found in URDF")
            return False

        ik_joints = p.calculateInverseKinematics(
            robot, eef_link, target_pos, target_orn, physicsClientId=c,
        )

        n_arm = len(self._env._arm_joints) if self._env._arm_joints else len(_HOME_JOINTS)
        target = np.array(ik_joints[:n_arm])

        raw_start = self._env.get_robot_joint_state()
        start = (
            np.array(raw_start, dtype=float)[:n_arm]
            if raw_start is not None
            else np.array(_HOME_JOINTS, dtype=float)
        )

        for i in range(_MOTION_STEPS):
            alpha = (i + 1) / _MOTION_STEPS
            joints = (start + alpha * (target - start)).tolist()
            self._env.set_robot_joints(joints)
            if self._held_object:
                self._attach_object_to_eef()
            p.stepSimulation(physicsClientId=c)

            if self._robot_in_collision():
                self._logger.warning(
                    "_move_to [%s]: collision detected at step %d/%d — aborting",
                    label, i + 1, _MOTION_STEPS,
                )
                return False

            time.sleep(_STEP_SLEEP)

        return True

    def _get_base_xy(self) -> List[float]:
        """Return the current [x, y] base position from PyBullet."""
        if not PYBULLET_AVAILABLE or self._env._robot_id is None:
            return [0.0, 0.0]
        pos, _ = p.getBasePositionAndOrientation(
            self._env._robot_id, physicsClientId=self._env._client
        )
        return [pos[0], pos[1]]

    def _teleport_base(self, xy: List[float], yaw: float) -> None:
        """Move the base body to xy at the given yaw angle."""
        if not PYBULLET_AVAILABLE or self._env._robot_id is None:
            return
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(
            self._env._robot_id,
            [xy[0], xy[1], 0.0],
            orn,
            physicsClientId=self._env._client,
        )

    def _set_head(self, head_pan: float, head_tilt: float) -> None:
        """Set head pan/tilt joints without changing arm configuration."""
        current = self._env.get_robot_joint_state()
        if current is None:
            return
        joints = list(current)
        joints = (joints + [0.0] * len(_HOME_JOINTS))[:len(_HOME_JOINTS)]
        joints[_IDX_HEAD_PAN]  = head_pan
        joints[_IDX_HEAD_TILT] = head_tilt
        self._env.set_robot_joints(joints)

    def _find_aligned_base_position(
        self,
        target_xy: np.ndarray,
    ) -> Tuple[Optional[List[float]], float]:
        """
        Find the closest collision-free base position that aligns the arm's
        extension axis with target_xy and keeps the target within the arm's
        horizontal workspace.

        Stretch's arm extends along the robot's local -Y axis.  For a given
        base yaw θ the arm's world-frame extension direction is (-sin θ, -cos θ).
        The base must sit along the line  target_xy + t * (sin θ, cos θ)
        for some t in [ARM_REACH_MIN, ARM_REACH_MAX - margin] so the target
        is reachable.

        Strategy:
          1. Compute the yaw that points the arm directly at the target from
             the robot's current position (ideal alignment).
          2. Search along the alignment axis from MIN reach to MAX reach,
             stepping inward from the far end, and return the first
             collision-free position.

        Returns (aligned_xy, base_yaw), or (None, 0.0) if nothing found.
        """
        current_xy = np.array(self._get_base_xy())
        delta = target_xy - current_xy
        dist = float(np.linalg.norm(delta))

        if dist < 1e-4:
            # Robot is already on top of the target — use current yaw
            if PYBULLET_AVAILABLE and self._env._robot_id is not None:
                _, quat = p.getBasePositionAndOrientation(
                    self._env._robot_id, physicsClientId=self._env._client
                )
                base_yaw = p.getEulerFromQuaternion(quat)[2]
            else:
                base_yaw = 0.0
        else:
            dx, dy = float(delta[0]), float(delta[1])
            # Robot -Y must face the target: yaw = atan2(-dx, -dy)
            base_yaw = math.atan2(-dx, -dy)

        # The alignment axis: base positions on  target + t * arm_dir_world
        # where arm_dir_world = (sin θ, cos θ) is the world +Y of the robot.
        arm_dir_world = np.array([math.sin(base_yaw), math.cos(base_yaw)])

        # Usable reach range (with margin from full extension)
        reach_max = _ARM_REACH_MAX - _ARM_REACH_MARGIN
        reach_min = _ARM_REACH_MIN + _BASE_RADIUS  # keep base from overlapping target

        # Search from closest valid position outward, prefer getting close
        step = 0.05  # metres between candidate positions
        n_steps = max(1, int((reach_max - reach_min) / step) + 1)
        candidates = [
            reach_min + i * step for i in range(n_steps)
        ]

        for t in candidates:
            candidate_xy = (target_xy + arm_dir_world * t).tolist()
            if not self._base_disc_in_collision(candidate_xy, base_yaw):
                return candidate_xy, base_yaw

        self._logger.warning(
            "_find_aligned_base_position: no collision-free position found along axis "
            "(target=%s, yaw=%.2f)",
            target_xy.tolist(), base_yaw,
        )
        return None, base_yaw

    def _compute_standoff_xy(
        self,
        target_xyz: List[float],
        standoff_distance: float,
    ) -> List[float]:
        """
        Compute a standoff position `standoff_distance` metres away from the
        target in the XY plane, along the robot→target direction.
        """
        start = np.array(self._get_base_xy())
        goal  = np.array(target_xyz[:2])
        direction = goal - start
        dist = float(np.linalg.norm(direction))
        if dist < 1e-6:
            return start.tolist()
        direction /= dist
        standoff = goal - direction * standoff_distance
        return standoff.tolist()

    def _attach_object_to_eef(self) -> None:
        if self._held_object is None:
            return
        tcp = self._env.get_robot_tcp_pose()
        if tcp is not None:
            self._env.move_object(self._held_object, tcp[0].tolist())

    def _try_attach(self, eef_pos: List[float], attach_radius: float = 0.12) -> None:
        """Attach the nearest object within attach_radius (Stretch has a wider reach)."""
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
        return self._env.get_object_position(obj_id)

    def _resolve_surface_label(self, label: str) -> Optional[List[float]]:
        return self._resolve_point_label(label)
