"""
Camera-to-robot transform calculator using PyBullet.

Uses the robot URDF to automatically compute the transformation
from camera frame to robot base frame.
"""

import logging
import numpy as np
from typing import Optional
from pathlib import Path

from ptp.kinematics.base_pybullet_interface import BasePybulletInterface

logger = logging.getLogger(__name__)

_SIM_DIR = Path(__file__).parent
_DEFAULT_URDF = _SIM_DIR / "urdfs" / "xarm7_camera" / "xarm7.urdf"
_DEFAULT_CAMERA_LINK = "camera_color_optical_frame"
_DEFAULT_BASE_LINK = "link_base"
_DEFAULT_TCP_LINK = "link_tcp"
_DEFAULT_N_ARM_JOINTS = 7


class TransformCalculator:
    """
    Calculate camera-to-robot transform using PyBullet and URDF.

    This uses PyBullet's forward kinematics to get the exact transform
    from the camera optical frame to the robot base frame.
    """

    def __init__(self, urdf_path: str, camera_link_name: str = _DEFAULT_CAMERA_LINK,
                 use_gui: bool = False, base_link_name: str = _DEFAULT_BASE_LINK,
                 tcp_link_name: str = _DEFAULT_TCP_LINK, n_arm_joints: int = _DEFAULT_N_ARM_JOINTS):
        """
        Initialize transform calculator.

        Args:
            urdf_path: Path to robot URDF file
            camera_link_name: Name of camera link in URDF (optical frame for RealSense)
            use_gui: If True, use PyBullet GUI for visualization (default: False, headless)
            base_link_name: Name of the robot base link in the URDF
            tcp_link_name: Name of the TCP/end-effector link in the URDF
            n_arm_joints: Number of arm joints to track
        """
        if use_gui:
            logger.warning("use_gui=True is not supported; running in DIRECT mode")

        self.urdf_path = urdf_path
        self.camera_link_name = camera_link_name
        self.use_gui = use_gui

        self._interface = BasePybulletInterface(
            urdf_path=urdf_path,
            camera_link_name=camera_link_name,
            tcp_link_name=tcp_link_name,
            base_link_name=base_link_name,
            n_arm_joints=n_arm_joints,
        )

        logging.info(f"Transform calculator initialized with URDF: {urdf_path}")

    def get_camera_to_base_transform(self, joint_positions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get 4x4 transformation matrix from camera frame to robot base frame.

        Args:
            joint_positions: Optional joint angles (radians). If None, uses zero position.

        Returns:
            4x4 homogeneous transformation matrix
        """
        if joint_positions is not None:
            self._interface.set_current_joint_state(joint_positions)

        pos, rot = self._interface.get_camera_transform()
        if pos is None or rot is None:
            raise ValueError(f"Camera link '{self.camera_link_name}' not found in URDF")

        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = pos

        logging.debug(f"Camera to base transform computed at joint config: {joint_positions}")
        logging.debug(f"Transform:\n{T}")

        return T

    def get_transform_at_current_state(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Get camera-to-base transform for current robot configuration.

        Args:
            joint_positions: Current joint angles (radians)

        Returns:
            4x4 transformation matrix from camera to base
        """
        return self.get_camera_to_base_transform(joint_positions)

    def cleanup(self):
        """Cleanup PyBullet connection."""
        self._interface.cleanup()

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass


def get_urdf_path() -> str:
    """Get path to xArm7 URDF file."""
    return str(_DEFAULT_URDF.resolve())


def create_transform_calculator(camera_link: str = _DEFAULT_CAMERA_LINK,
                               use_gui: bool = True) -> TransformCalculator:
    """
    Create a TransformCalculator with default xArm7 URDF.

    Args:
        camera_link: Name of camera link in URDF
        use_gui: If True, visualize in PyBullet GUI

    Returns:
        TransformCalculator instance
    """
    urdf_path = get_urdf_path()
    return TransformCalculator(urdf_path, camera_link, use_gui=use_gui)


if __name__ == "__main__":
    # Test the transform calculator
    logging.basicConfig(level=logging.INFO)

    print("Testing TransformCalculator...")

    # Create calculator
    calc = create_transform_calculator()

    # Get transform at zero position
    print("\n1. Transform at zero position:")
    T = calc.get_camera_to_base_transform()
    print(T)

    # Get transform at a different configuration
    print("\n2. Transform at different joint angles:")
    joint_angles = np.array([0, 0.5, 0, 0.5, 0, 0.5, 0])  # Example configuration
    T2 = calc.get_camera_to_base_transform(joint_angles)
    print(T2)

    # Extract position from transform
    camera_pos_in_base = T[:3, 3]
    print(f"\nCamera position in base frame: {camera_pos_in_base}")
    print(f"Distance from base: {np.linalg.norm(camera_pos_in_base):.3f} meters")

    calc.cleanup()
    print("\nTest completed")
