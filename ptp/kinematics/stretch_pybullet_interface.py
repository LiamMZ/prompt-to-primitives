"""Hello Robot Stretch RE1V0 PyBullet interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from ptp.kinematics.base_pybullet_interface import BasePybulletInterface

_SIM_DIR = Path(__file__).parent / "sim"
_STRETCH_URDF = _SIM_DIR / "urdfs" / "stretch" / "stretch_description_RE1V0_tool_stretch_gripper.urdf"
_CAMERA_LINK = "camera_color_optical_frame"  # head-mounted RealSense
_TCP_LINK = "link_grasp_center"
_BASE_LINK = "base_link"
_N_ARM_JOINTS = 12  # all movable joints: 2 wheels + lift + 4 arm + wrist + 2 gripper + head_pan + head_tilt


class StretchPybulletInterface(BasePybulletInterface):
    def __init__(
        self,
        urdf_path: Optional[Union[str, Path]] = None,
        camera_link_name: str = _CAMERA_LINK,
        tcp_link_name: str = _TCP_LINK,
        static_camera_tf: Optional[Any] = None,
    ) -> None:
        super().__init__(
            urdf_path=urdf_path or _STRETCH_URDF,
            camera_link_name=camera_link_name,
            tcp_link_name=tcp_link_name,
            base_link_name=_BASE_LINK,
            n_arm_joints=_N_ARM_JOINTS,
            static_camera_tf=static_camera_tf,
        )


def create_stretch_interface(
    camera_link: str = _CAMERA_LINK,
    static_camera_tf: Optional[Any] = None,
) -> StretchPybulletInterface:
    return StretchPybulletInterface(camera_link_name=camera_link, static_camera_tf=static_camera_tf)
