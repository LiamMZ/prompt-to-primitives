"""
PyBullet Simulated Camera

Renders RGB and depth images from the xArm7's wrist camera frame using
PyBullet's built-in renderer. The camera pose is read directly from the
URDF link state, so it stays in sync with the robot joint configuration.

Usage:
    from ptp.kinematics.sim.pybullet_camera import PyBulletCamera
    from ptp.kinematics.xarm_pybullet_interface import XArmPybulletInterface

    robot = XArmPybulletInterface()
    cam = PyBulletCamera(robot, physics_client=robot._physics_client)
    frame, intrinsics = cam.capture()
    # frame.color  -> (H, W, 3) uint8 RGB
    # frame.depth  -> (H, W) float32 metres
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    p = None

from ptp.camera.base_camera import CameraFrame, CameraIntrinsics
from ptp.kinematics.xarm_pybullet_interface import XArmPybulletInterface

logger = logging.getLogger(__name__)

# RealSense D435 defaults (used when no intrinsics supplied)
_DEFAULT_WIDTH  = 640
_DEFAULT_HEIGHT = 480
_DEFAULT_FX     = 610.0
_DEFAULT_FY     = 610.0

# Near/far clip planes for depth rendering (metres)
_NEAR = 0.05
_FAR  = 3.0


class PyBulletCamera:
    """
    Simulated wrist camera rendered from the robot's camera_color_optical_frame.

    Shares the PyBullet physics client with the robot — no extra connection needed.
    """

    def __init__(
        self,
        robot: XArmPybulletInterface,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        fx: float = _DEFAULT_FX,
        fy: float = _DEFAULT_FY,
        near: float = _NEAR,
        far: float = _FAR,
    ) -> None:
        if not PYBULLET_AVAILABLE:
            raise ImportError("PyBullet is required. Install with: pip install pybullet")

        self.robot = robot
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.near = near
        self.far = far

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture(self) -> Tuple[CameraFrame, CameraIntrinsics]:
        """
        Render a frame from the robot's current camera pose.

        Returns:
            (CameraFrame, CameraIntrinsics)
            CameraFrame.color  — (H, W, 3) uint8 RGB
            CameraFrame.depth  — (H, W) float32 metres
        """
        view_matrix, proj_matrix = self._build_matrices()

        _, _, rgba, depth_buf, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.robot._physics_client,
        )

        color = np.array(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)[..., :3]
        depth = self._linearise_depth(np.array(depth_buf, dtype=np.float32).reshape(self.height, self.width))

        intrinsics = CameraIntrinsics(
            fx=self.fx, fy=self.fy,
            cx=self.cx, cy=self.cy,
            width=self.width, height=self.height,
        )
        return CameraFrame(color=color, depth=depth), intrinsics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_camera_world_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (position, rotation_matrix) of camera_color_optical_frame in world frame.
        Uses PyBullet link state directly so it reflects the current joint config.
        """
        self.robot._apply_joints_to_sim()

        cam_idx = self.robot._get_link_index(self.robot.camera_link_name)
        if cam_idx is None:
            raise ValueError(
                f"Camera link '{self.robot.camera_link_name}' not found in URDF"
            )

        if cam_idx == -1:
            pos, orn = p.getBasePositionAndOrientation(
                self.robot._robot_id,
                physicsClientId=self.robot._physics_client,
            )
        else:
            state = p.getLinkState(
                self.robot._robot_id,
                cam_idx,
                physicsClientId=self.robot._physics_client,
            )
            pos, orn = state[4], state[5]

        rot = np.array(
            p.getMatrixFromQuaternion(orn, physicsClientId=self.robot._physics_client)
        ).reshape(3, 3)
        return np.array(pos, dtype=float), rot

    def _build_matrices(self) -> Tuple[list, list]:
        """
        Build PyBullet view and projection matrices from the camera pose.

        The optical frame convention (Z forward, X right, Y down) matches
        the standard camera convention used by PyBullet's renderer.
        """
        pos, rot = self._get_camera_world_pose()

        # Camera axes in world frame
        # Optical frame: +Z = looking direction, +X = right, +Y = down
        cam_z = rot[:, 2]   # forward / look direction
        cam_y = rot[:, 1]   # down in image = up_vec negated

        eye    = pos
        target = pos + cam_z
        up     = -cam_y  # PyBullet up vector is world-up; negate optical Y

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
            physicsClientId=self.robot._physics_client,
        )

        fov_y = 2.0 * np.degrees(np.arctan2(self.height / 2.0, self.fy))
        aspect = self.width / self.height

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov_y,
            aspect=aspect,
            nearVal=self.near,
            farVal=self.far,
            physicsClientId=self.robot._physics_client,
        )
        return view_matrix, proj_matrix

    def _linearise_depth(self, depth_buf: np.ndarray) -> np.ndarray:
        """
        Convert PyBullet's non-linear depth buffer [0,1] to metric depth (metres).

        PyBullet stores depth as:  d = (far * (z - near)) / (z * (far - near))
        Inverting:                 z = (far * near) / (far - d * (far - near))
        """
        n, f = self.near, self.far
        return (f * n / (f - depth_buf * (f - n))).astype(np.float32)
