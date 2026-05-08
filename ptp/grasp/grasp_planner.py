"""Antipodal grasp planner.

Takes a contact position and a point cloud for the target object, samples
candidate grasp orientations around the approach axis, scores them by
centering quality and floor clearance, and optionally collision-checks each
trajectory before returning the best viable grasp.

Usage:
    planner = GraspPlanner(pybullet_interface)
    candidate = planner.plan(contact_position, seed_orientation="top_down")
    if candidate:
        orientation = candidate.orientation   # xyzw quaternion
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Set

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class GraspCandidate:
    """A single viable antipodal grasp hypothesis."""

    orientation: np.ndarray     # xyzw quaternion (world frame)
    position: np.ndarray        # TCP position (world frame)
    approach_angle_rad: float   # rotation about the approach axis relative to seed
    seed_orientation: str       # "top_down" or "side"
    manipulability: float       # centering + width + floor score (higher = better)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

# Default gripper geometry (xArm Robotiq gripper)
_GRIPPER_WIDTH_M    = 0.085
_FINGER_THICKNESS_M = 0.012
_N_ROTATIONS        = 36


class GraspPlanner:
    """Antipodal grasp sampler backed by a PyBullet IK/FK interface.

    Args:
        planner: A BasePybulletInterface instance used for optional trajectory
                 collision checking.  Pass None to disable collision checking.

    Example:
        >>> gp = GraspPlanner(pybullet_iface)
        >>> c = gp.plan(np.array([0.35, 0.0, 0.12]), seed_orientation="top_down")
        >>> c.orientation   # xyzw
    """

    def __init__(self, planner: Optional[Any] = None) -> None:
        self._planner = planner

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        contact_position: np.ndarray,
        object_points: Optional[np.ndarray] = None,
        seed_orientation: str = "top_down",
        clearance_profile: Optional[Any] = None,
        ignore_labels: Optional[Set[str]] = None,
        gripper_width_m: float = _GRIPPER_WIDTH_M,
        finger_thickness_m: float = _FINGER_THICKNESS_M,
        n_rotations: int = _N_ROTATIONS,
        standoff_m: float = 0.0,
        floor_z: float = 0.0,
    ) -> Optional[GraspCandidate]:
        """Sample antipodal grasps and return the best reachable candidate.

        Args:
            contact_position: 3-D world-frame target (from Molmo / point_label).
            object_points: (N, 3) world-frame point cloud for the target object.
                           If None, only orientation sampling is performed (no
                           antipodal jaw-contact refinement).
            seed_orientation: "top_down" (gripper Z axis points down) or "side".
            clearance_profile: Optional ClearanceProfile with approach_corridors.
            ignore_labels: Object labels to exclude from collision checks.
            gripper_width_m: Maximum jaw spread in metres.
            finger_thickness_m: Gripper finger height — used as floor clearance margin.
            n_rotations: Number of in-plane rotation candidates to evaluate.
            standoff_m: Approach standoff added to the returned position along the
                        approach axis (for pre-grasp waypoints).
            floor_z: Support-surface Z in world frame (default 0.0).

        Returns:
            GraspCandidate or None if no valid grasp was found.
        """
        seed_quat = self._seed_quaternion(seed_orientation)
        seed_rot  = Rotation.from_quat(seed_quat)
        approach  = seed_rot.apply(np.array([0.0, 0.0, 1.0]))
        approach  = approach / (np.linalg.norm(approach) + 1e-9)

        # Build orthonormal frame perpendicular to approach.
        u, v = self._perp_frame(approach)

        floor_clearance_z = floor_z + finger_thickness_m
        half_width        = gripper_width_m / 2.0
        angles            = np.linspace(0.0, math.pi, n_rotations, endpoint=False)

        best_score: float = -np.inf
        best: Optional[GraspCandidate] = None

        for angle in angles:
            grasp_axis = np.cos(angle) * u + np.sin(angle) * v
            grasp_axis /= np.linalg.norm(grasp_axis) + 1e-9

            grasp_center, jaw_spread = self._antipodal_center(
                contact_position, object_points, grasp_axis, half_width
            )

            # Reject if jaws would clip the floor.
            jaw_a = grasp_center + half_width * grasp_axis
            jaw_b = grasp_center - half_width * grasp_axis
            if min(jaw_a[2], jaw_b[2]) < floor_clearance_z:
                continue

            grasp_pos  = grasp_center - standoff_m * approach
            cand_quat  = self._build_quaternion(grasp_axis, approach)

            # Optional trajectory collision check.
            if self._planner is not None and not self._trajectory_clear(
                grasp_pos, cand_quat, ignore_labels
            ):
                continue

            score = self._score(grasp_center, contact_position, grasp_axis,
                                jaw_spread, gripper_width_m,
                                floor_clearance_z, floor_z)

            if score > best_score:
                best_score = score
                best = GraspCandidate(
                    orientation=cand_quat,
                    position=grasp_pos,
                    approach_angle_rad=float(angle),
                    seed_orientation=seed_orientation,
                    manipulability=float(score),
                )

        return best

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seed_quaternion(seed_orientation: str) -> np.ndarray:
        """Return the xyzw seed quaternion for "top_down" or "side"."""
        if seed_orientation == "side":
            return np.array([-0.6894, 0.0305, -0.7237, 0.0033])
        return np.array([-0.9983, 0.0314, 0.0438, 0.0223])

    @staticmethod
    def _perp_frame(approach: np.ndarray):
        """Build two orthonormal vectors perpendicular to approach."""
        hint = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(approach, hint)) > 0.9:
            hint = np.array([1.0, 0.0, 0.0])
        u = np.cross(approach, hint)
        u /= np.linalg.norm(u) + 1e-9
        v = np.cross(approach, u)
        v /= np.linalg.norm(v) + 1e-9
        return u, v

    @staticmethod
    def _antipodal_center(
        contact: np.ndarray,
        object_points: Optional[np.ndarray],
        grasp_axis: np.ndarray,
        half_width: float,
    ):
        """Return (grasp_center, jaw_spread) for this grasp axis.

        If no object_points are available, returns (contact, 0) so orientation
        sampling still works without a point cloud.
        """
        if object_points is None or len(object_points) < 4:
            return contact.copy(), 0.0

        relative = object_points - contact
        proj = relative @ grasp_axis

        pos_mask = (proj >= 0) & (proj <= half_width)
        neg_mask = (proj <= 0) & (proj >= -half_width)
        if not (np.any(pos_mask) and np.any(neg_mask)):
            return contact.copy(), 0.0

        jaw_pos  = float(np.max(proj[pos_mask]))
        jaw_neg  = float(np.min(proj[neg_mask]))
        midpoint = (jaw_pos + jaw_neg) / 2.0
        spread   = jaw_pos - jaw_neg
        return contact + midpoint * grasp_axis, spread

    @staticmethod
    def _build_quaternion(grasp_axis: np.ndarray, approach: np.ndarray) -> np.ndarray:
        """Build a rotation matrix where X=grasp_axis, Z=approach, return as xyzw quat."""
        x_axis = grasp_axis
        z_axis = approach
        y_axis = np.cross(z_axis, x_axis)
        norm_y = np.linalg.norm(y_axis)
        if norm_y < 1e-9:
            return Rotation.identity().as_quat()
        y_axis /= norm_y
        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        return Rotation.from_matrix(R).as_quat()

    @staticmethod
    def _score(
        grasp_center: np.ndarray,
        contact: np.ndarray,
        grasp_axis: np.ndarray,
        jaw_spread: float,
        gripper_width_m: float,
        floor_clearance_z: float,
        floor_z: float,
    ) -> float:
        """Composite score: centering + width + floor clearance."""
        half_width = gripper_width_m / 2.0
        midpoint   = float((grasp_center - contact) @ grasp_axis)
        centering  = -abs(midpoint)
        width      = jaw_spread / gripper_width_m if gripper_width_m > 0 else 0.0
        jaw_a      = grasp_center + half_width * grasp_axis
        jaw_b      = grasp_center - half_width * grasp_axis
        floor_margin = min(jaw_a[2], jaw_b[2]) - floor_clearance_z
        floor_score  = min(floor_margin / 0.05, 1.0)
        return centering + 0.3 * width + 0.2 * floor_score

    def _trajectory_clear(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        ignore_labels: Optional[Set[str]],
    ) -> bool:
        """Return True if a planned trajectory to this pose is collision-free."""
        try:
            success, traj, _ = self._planner.move_to_pose(
                target_position=target_position.tolist(),
                target_orientation=target_orientation.tolist(),
                execute=False,
                ignore_labels=ignore_labels,
            )
            return bool(success and traj is not None)
        except Exception:
            return False
