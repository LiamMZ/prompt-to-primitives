"""
Transform calculator for the Unitree B1 quadruped + Z1 arm + Robotiq gripper.

Coordinate frames
-----------------
world   : global/map frame (odometry or SLAM origin)
b1_body : B1 base_link, given by the B1's reported odometry pose
cam     : B1 front-camera optical frame (fixed extrinsic relative to b1_body)
z1_base : Z1 arm base frame (fixed mount offset from b1_body)
tcp     : Z1 end-effector / Robotiq gripper face

Incoming perception data (object positions, depth estimates) arrives in the
cam frame associated with the image that was used. This class converts those
positions to the b1_body frame or to the z1_base frame needed for arm control.

Usage
-----
    calc = B1Z1TransformCalculator(
        front_cam_extrinsic=T_body_cam,   # 4×4 ndarray or (pos, quat_xyzw)
        z1_mount_offset=T_body_z1base,    # 4×4 ndarray or (pos, quat_xyzw)
    )

    # Update B1 pose from odometry (call each time it changes)
    calc.set_b1_pose(position, quaternion_xyzw)

    # Update Z1 joint state (call after each getQ response)
    calc.set_z1_joints(q)  # length-6 list/array of radians

    # Convert a point seen by the front camera → z1_base frame
    p_z1 = calc.cam_to_z1_base(p_cam)

    # Get TCP pose in world frame
    pos_world, quat_world = calc.get_tcp_pose_world()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known fixed geometry (placeholder defaults — override at construction time)
# ---------------------------------------------------------------------------

# B1 front camera: roughly 0.35 m forward, 0.05 m up from body origin,
# tilted slightly down. Replace with calibrated values.
_DEFAULT_CAM_POS_BODY = np.array([0.35, 0.0, 0.05])
_DEFAULT_CAM_RPY_BODY = np.array([0.0, -0.15, 0.0])  # slight pitch down (rad)

# Z1 base mounts on the B1's top deck: roughly centred, ~0.15 m up.
_DEFAULT_Z1_BASE_POS_BODY = np.array([0.0, 0.0, 0.15])
_DEFAULT_Z1_BASE_RPY_BODY = np.array([0.0, 0.0, 0.0])

# Path to a standalone Z1 URDF for arm FK (optional — FK falls back to None).
_Z1_URDF_PATH = Path(__file__).parent / "sim" / "urdfs" / "z1" / "z1.urdf"
_Z1_TCP_LINK = "gripperStator"   # adjust to actual link name once URDF is available
_Z1_BASE_LINK = "base"
_Z1_N_JOINTS = 6


def _make_T(pos: Any, rot: Any) -> np.ndarray:
    """Build a 4×4 homogeneous transform from position + rotation.

    rot can be: Rotation, (3,3) matrix, quaternion [x,y,z,w], or RPY [r,p,y].
    """
    pos = np.asarray(pos, dtype=float)
    if isinstance(rot, Rotation):
        R = rot.as_matrix()
    elif isinstance(rot, np.ndarray) and rot.shape == (3, 3):
        R = rot
    else:
        arr = np.asarray(rot, dtype=float)
        if arr.shape == (4,):
            R = Rotation.from_quat(arr).as_matrix()   # xyzw
        elif arr.shape == (3,):
            R = Rotation.from_euler("xyz", arr).as_matrix()  # RPY
        else:
            raise ValueError(f"Unsupported rotation format: {arr.shape}")
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def _parse_extrinsic(spec: Any, default_pos: np.ndarray, default_rpy: np.ndarray) -> np.ndarray:
    if spec is None:
        return _make_T(default_pos, default_rpy)
    if isinstance(spec, np.ndarray) and spec.shape == (4, 4):
        return spec.astype(float)
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        return _make_T(spec[0], spec[1])
    raise ValueError(f"Extrinsic must be a 4×4 matrix or (pos, rot) tuple, got {type(spec)}")


class B1Z1TransformCalculator:
    """
    Geometry engine for the B1 + Z1 + Robotiq system.

    All heavy lifting is pure NumPy; PyBullet is used only when a Z1 URDF is
    present and arm FK is needed.
    """

    def __init__(
        self,
        front_cam_extrinsic: Optional[Any] = None,
        z1_mount_offset: Optional[Any] = None,
        z1_urdf_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Args:
            front_cam_extrinsic: T_body_cam — 4×4 ndarray or (pos, rot).
                Defaults to hardcoded approximate geometry.
            z1_mount_offset: T_body_z1base — 4×4 ndarray or (pos, rot).
                Defaults to hardcoded approximate geometry.
            z1_urdf_path: Path to Z1 URDF. When provided, arm FK is available.
                If None and default path exists, that is used automatically.
        """
        self._T_body_cam: np.ndarray = _parse_extrinsic(
            front_cam_extrinsic, _DEFAULT_CAM_POS_BODY, _DEFAULT_CAM_RPY_BODY
        )
        self._T_body_z1base: np.ndarray = _parse_extrinsic(
            z1_mount_offset, _DEFAULT_Z1_BASE_POS_BODY, _DEFAULT_Z1_BASE_RPY_BODY
        )

        # B1 pose in world frame (identity until first odometry update)
        self._T_world_body: np.ndarray = np.eye(4)

        # Z1 FK interface (optional)
        self._z1_fk: Optional[Any] = None
        urdf = Path(z1_urdf_path) if z1_urdf_path else _Z1_URDF_PATH
        if urdf.exists():
            self._init_z1_fk(urdf)
        else:
            logger.info(
                "Z1 URDF not found at %s — arm FK disabled. "
                "Place the URDF at that path to enable it.",
                urdf,
            )

        logger.info("B1Z1TransformCalculator ready")

    # ------------------------------------------------------------------
    # Z1 FK via PyBullet
    # ------------------------------------------------------------------

    def _init_z1_fk(self, urdf: Path) -> None:
        try:
            from ptp.kinematics.base_pybullet_interface import BasePybulletInterface
            self._z1_fk = BasePybulletInterface(
                urdf_path=urdf,
                camera_link_name=_Z1_BASE_LINK,   # unused; we only need TCP FK
                tcp_link_name=_Z1_TCP_LINK,
                base_link_name=_Z1_BASE_LINK,
                n_arm_joints=_Z1_N_JOINTS,
            )
            logger.info("Z1 PyBullet FK loaded from %s", urdf)
        except Exception as exc:
            logger.warning("Could not init Z1 FK: %s", exc)
            self._z1_fk = None

    # ------------------------------------------------------------------
    # State setters
    # ------------------------------------------------------------------

    def set_b1_pose(
        self,
        position: Any,
        orientation: Any,
    ) -> None:
        """
        Update B1 body pose in world frame from odometry.

        Args:
            position: (3,) xyz in metres
            orientation: quaternion [x,y,z,w] or (3,3) rotation matrix
        """
        self._T_world_body = _make_T(position, orientation)

    def set_z1_joints(self, q: Any) -> None:
        """Update Z1 joint angles (radians). No-op if FK is not available."""
        if self._z1_fk is not None:
            self._z1_fk.set_current_joint_state(q)

    # ------------------------------------------------------------------
    # Derived transforms
    # ------------------------------------------------------------------

    @property
    def T_world_cam(self) -> np.ndarray:
        """Camera pose in world frame."""
        return self._T_world_body @ self._T_body_cam

    @property
    def T_world_z1base(self) -> np.ndarray:
        """Z1 base frame in world frame."""
        return self._T_world_body @ self._T_body_z1base

    @property
    def T_cam_z1base(self) -> np.ndarray:
        """Z1 base frame expressed in camera frame (body pose cancels out)."""
        T_body_z1base = self._T_body_z1base
        T_body_cam = self._T_body_cam
        return np.linalg.inv(T_body_cam) @ T_body_z1base

    # ------------------------------------------------------------------
    # Point / pose conversions
    # ------------------------------------------------------------------

    def cam_to_body(self, point_cam: Any) -> np.ndarray:
        """Transform a 3-D point from camera frame to B1 body frame."""
        p = np.ones(4)
        p[:3] = np.asarray(point_cam, dtype=float).flatten()[:3]
        return (self._T_body_cam @ p)[:3]

    def cam_to_world(self, point_cam: Any) -> np.ndarray:
        """Transform a 3-D point from camera frame to world frame."""
        p = np.ones(4)
        p[:3] = np.asarray(point_cam, dtype=float).flatten()[:3]
        return (self.T_world_cam @ p)[:3]

    def cam_to_z1_base(self, point_cam: Any) -> np.ndarray:
        """
        Transform a 3-D point from camera frame to Z1 base frame.

        This is the primary conversion needed for arm motion planning:
        take a point observed by the B1 front camera and express it in the
        coordinate frame the Z1 controller expects.

        Chain: cam → body → z1_base
               T_body_cam  ·  p_cam  gives p_body (homogeneous)
               T_z1base_body · p_body gives p_z1base
        """
        p = np.ones(4)
        p[:3] = np.asarray(point_cam, dtype=float).flatten()[:3]
        T_z1base_body = np.linalg.inv(self._T_body_z1base)
        p_body = self._T_body_cam @ p
        return (T_z1base_body @ p_body)[:3]

    def world_to_z1_base(self, point_world: Any) -> np.ndarray:
        """Transform a 3-D point from world frame to Z1 base frame."""
        p = np.ones(4)
        p[:3] = np.asarray(point_world, dtype=float).flatten()[:3]
        T_z1base_world = np.linalg.inv(self.T_world_z1base)
        return (T_z1base_world @ p)[:3]

    def pose_cam_to_z1_base(
        self, position_cam: Any, orientation_cam: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform a full pose from camera frame to Z1 base frame.

        Args:
            position_cam: (3,) position in camera frame
            orientation_cam: quaternion [x,y,z,w] or (3,3) rotation matrix

        Returns:
            (position_z1base, quaternion_xyzw_z1base)
        """
        if isinstance(orientation_cam, np.ndarray) and orientation_cam.shape == (3, 3):
            R_cam_obj = Rotation.from_matrix(orientation_cam)
        else:
            R_cam_obj = Rotation.from_quat(np.asarray(orientation_cam, dtype=float))

        R_body_cam = Rotation.from_matrix(self._T_body_cam[:3, :3])
        R_body_z1 = Rotation.from_matrix(self._T_body_z1base[:3, :3])

        # Orientation: z1base ← body ← cam ← obj
        R_z1base_cam = R_body_z1.inv() * R_body_cam
        R_z1base_obj = R_z1base_cam * R_cam_obj

        pos_z1base = self.cam_to_z1_base(position_cam)
        return pos_z1base, R_z1base_obj.as_quat()

    # ------------------------------------------------------------------
    # TCP pose (requires Z1 URDF)
    # ------------------------------------------------------------------

    def get_tcp_pose_z1base(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Return (position, quaternion_xyzw) of the TCP in Z1 base frame.

        Requires a Z1 URDF. Returns None if FK is unavailable.
        """
        if self._z1_fk is None:
            logger.warning("Z1 FK not available — no URDF loaded")
            return None
        return self._z1_fk.get_robot_tcp_pose()

    def get_tcp_pose_world(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Return (position, quaternion_xyzw) of the TCP in world frame.

        Requires Z1 URDF and a B1 pose update.
        """
        result = self.get_tcp_pose_z1base()
        if result is None:
            return None
        pos_z1, quat_z1 = result
        T_z1_tcp = _make_T(pos_z1, quat_z1)
        T_world_tcp = self.T_world_z1base @ T_z1_tcp
        pos_w = T_world_tcp[:3, 3]
        quat_w = Rotation.from_matrix(T_world_tcp[:3, :3]).as_quat()
        return pos_w, quat_w

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_geometry_summary(self) -> dict:
        """Return current transform state as a plain dict (for logging/debug)."""
        b1_pos = self._T_world_body[:3, 3].tolist()
        b1_quat = Rotation.from_matrix(self._T_world_body[:3, :3]).as_quat().tolist()
        cam_pos_world = self.T_world_cam[:3, 3].tolist()
        z1base_pos_world = self.T_world_z1base[:3, 3].tolist()
        summary = {
            "b1_body_world": {"position": b1_pos, "quaternion_xyzw": b1_quat},
            "front_cam_world": {"position": cam_pos_world},
            "z1_base_world": {"position": z1base_pos_world},
            "z1_fk_available": self._z1_fk is not None,
        }
        tcp = self.get_tcp_pose_world()
        if tcp is not None:
            summary["tcp_world"] = {"position": tcp[0].tolist(), "quaternion_xyzw": tcp[1].tolist()}
        return summary

    def cleanup(self) -> None:
        if self._z1_fk is not None:
            self._z1_fk.cleanup()
            self._z1_fk = None

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_b1_z1_calculator(
    front_cam_extrinsic: Optional[Any] = None,
    z1_mount_offset: Optional[Any] = None,
    z1_urdf_path: Optional[Union[str, Path]] = None,
) -> B1Z1TransformCalculator:
    """
    Create a B1Z1TransformCalculator with calibrated or default geometry.

    Pass calibrated 4×4 matrices (or (pos, quat) tuples) once measured.
    """
    return B1Z1TransformCalculator(
        front_cam_extrinsic=front_cam_extrinsic,
        z1_mount_offset=z1_mount_offset,
        z1_urdf_path=z1_urdf_path,
    )
