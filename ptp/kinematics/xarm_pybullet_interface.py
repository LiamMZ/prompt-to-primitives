"""xArm7 PyBullet interface — thin subclass of BasePybulletInterface with xArm7 defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pybullet as p

from ptp.kinematics.base_pybullet_interface import BasePybulletInterface

_SIM_DIR = Path(__file__).parent / "sim"
_XARM7_URDF = _SIM_DIR / "urdfs" / "xarm7_camera" / "xarm7.urdf"
_CAMERA_LINK = "camera_color_optical_frame"
_TCP_LINK = "link_tcp"
_BASE_LINK = "link_base"
_N_ARM_JOINTS = 7
_GRIPPER_OPEN_ANGLE = 0.0
_GRIPPER_CLOSED_ANGLE = 0.85
_GRIPPER_HOLD_FORCE = 50.0
_GRIPPER_MAX_VELOCITY = 2.0
_GRIPPER_JOINTS = (
    "drive_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)


class XArmPybulletInterface(BasePybulletInterface):
    def __init__(
        self,
        urdf_path: Optional[Union[str, Path]] = None,
        camera_link_name: str = _CAMERA_LINK,
        tcp_link_name: str = _TCP_LINK,
        static_camera_tf: Optional[Any] = None,
        use_gui: bool = False,
        floor_z: float = -0.04,
    ) -> None:
        self._gripper_angle = _GRIPPER_OPEN_ANGLE
        super().__init__(
            urdf_path=urdf_path or _XARM7_URDF,
            camera_link_name=camera_link_name,
            tcp_link_name=tcp_link_name,
            base_link_name=_BASE_LINK,
            n_arm_joints=_N_ARM_JOINTS,
            static_camera_tf=static_camera_tf,
            use_gui=use_gui,
            floor_z=floor_z,
        )
        self._initialize_gripper_pose()

    def _initialize_gripper_pose(self) -> None:
        """Force a consistent open gripper pose for mimic-joint URDFs.

        Example:
            >>> robot = XArmPybulletInterface(use_gui=False)
            >>> robot.open_gripper()
        """
        self.set_gripper_angle(_GRIPPER_OPEN_ANGLE)

    def set_gripper_angle(self, angle: float) -> None:
        """Set and hold all xArm gripper mimic joints at the requested angle.

        PyBullet does not enforce URDF mimic joints during simulation stepping,
        so each gripper joint is explicitly reset and motor-held. This keeps the
        GUI demo fingers from sagging under gravity between primitive waypoints.

        Example:
            >>> robot = XArmPybulletInterface(use_gui=False)
            >>> robot.set_gripper_angle(0.85)
        """
        self._gripper_angle = float(
            np.clip(angle, _GRIPPER_OPEN_ANGLE, _GRIPPER_CLOSED_ANGLE)
        )
        for joint_name in _GRIPPER_JOINTS:
            joint_idx = self._get_joint_index(joint_name)
            if joint_idx is None:
                continue
            p.resetJointState(
                self._robot_id,
                joint_idx,
                self._gripper_angle,
                physicsClientId=self._physics_client,
            )
            p.setJointMotorControl2(
                bodyUniqueId=self._robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=self._gripper_angle,
                force=_GRIPPER_HOLD_FORCE,
                maxVelocity=_GRIPPER_MAX_VELOCITY,
                physicsClientId=self._physics_client,
            )

    def open_gripper(self) -> None:
        """Open and hold the simulated xArm gripper."""
        self.set_gripper_angle(_GRIPPER_OPEN_ANGLE)

    def close_gripper(self) -> None:
        """Close and hold the simulated xArm gripper."""
        self.set_gripper_angle(_GRIPPER_CLOSED_ANGLE)

    def _apply_joints_to_sim(self) -> None:
        """Apply arm joints and reassert the held gripper pose."""
        super()._apply_joints_to_sim()
        self.set_gripper_angle(self._gripper_angle)


def create_sim_interface(
    camera_link: str = _CAMERA_LINK,
    static_camera_tf: Optional[Any] = None,
) -> XArmPybulletInterface:
    return XArmPybulletInterface(camera_link_name=camera_link, static_camera_tf=static_camera_tf)
