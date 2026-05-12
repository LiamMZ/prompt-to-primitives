"""Real xArm robot adapter for XArmPybulletPlannedPrimitives.

Wraps the xArm SDK into the duck-typed robot interface expected by
XArmPybulletPlannedPrimitives: get_robot_joint_state, set_robot_joint_angles,
open_gripper, close_gripper, disconnect.
"""

from __future__ import annotations

import threading
from typing import Any, List, Optional

import numpy as np

_DEFAULT_SAFE_JOINT_SPEED = 0.12
_DEFAULT_SAFE_JOINT_ACCEL = 0.25
_GRIPPER_OPEN   = 850
_GRIPPER_CLOSED = 0


class XArmRobotAdapter:
    """Real xArm adapter for XArmPybulletPlannedPrimitives.

    Args:
        robot_ip: xArm controller IP address.

    Example:
        >>> robot = XArmRobotAdapter("192.168.1.224")
        >>> robot.get_robot_joint_state()
        >>> robot.disconnect()
    """

    def __init__(self, robot_ip: str) -> None:
        try:
            from xarm.wrapper import XArmAPI
        except ImportError as exc:
            raise RuntimeError("xarm SDK is not available") from exc

        self.robot_ip = robot_ip
        self.arm = XArmAPI(robot_ip, is_radian=True)
        self.arm_lock = threading.Lock()
        self.current_joints: Optional[np.ndarray] = None

        self.arm.connect()
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)

    def get_robot_joint_state(self) -> Optional[np.ndarray]:
        """Return current robot joints in radians, or None on failure."""
        with self.arm_lock:
            code, angles = self.arm.get_servo_angle(is_radian=True)
        if code != 0 or angles is None:
            return None
        self.set_current_joint_state(angles)
        return self.current_joints.copy()

    def set_current_joint_state(self, joint_positions: Any) -> None:
        """Cache the latest known real xArm joint state."""
        self.current_joints = np.asarray(joint_positions, dtype=float).reshape(-1)[:7]

    def set_robot_joint_angles(
        self,
        joint_angles: List[float],
        wait: bool = True,
        speed: float = _DEFAULT_SAFE_JOINT_SPEED,
        acc: float = _DEFAULT_SAFE_JOINT_ACCEL,
    ) -> bool:
        """Move the real xArm to a joint target in radians."""
        with self.arm_lock:
            if wait:
                self.arm.set_mode(0)
                self.arm.set_state(0)
            code = self.arm.set_servo_angle(
                angle=list(joint_angles),
                speed=speed,
                mvacc=acc,
                wait=wait,
                is_radian=True,
            )
        if code == 0:
            self.set_current_joint_state(joint_angles)
            return True
        return False

    def cartesian_move(
        self,
        position: List[float],
        orientation_quat_xyzw: List[float],
        speed: float = 50.0,
        acc: float = 500.0,
        wait: bool = True,
    ) -> bool:
        """Move TCP directly to a Cartesian pose via xArm set_position (mm + rad).

        Args:
            position: [x, y, z] in metres (converted to mm internally).
            orientation_quat_xyzw: quaternion [x, y, z, w].
            speed: linear speed in mm/s.
            acc: linear acceleration in mm/s².
        """
        from scipy.spatial.transform import Rotation
        x_mm, y_mm, z_mm = [v * 1000.0 for v in position]
        roll, pitch, yaw = Rotation.from_quat(orientation_quat_xyzw).as_euler("xyz")
        with self.arm_lock:
            self.arm.set_mode(0)
            self.arm.set_state(0)
            code = self.arm.set_position(
                x=x_mm, y=y_mm, z=z_mm,
                roll=roll, pitch=pitch, yaw=yaw,
                speed=speed, mvacc=acc,
                is_radian=True, wait=wait,
            )
        return code == 0

    def cartesian_arc(
        self,
        waypoints: List[tuple],
        speed: float = 30.0,
        acc: float = 200.0,
    ) -> bool:
        """Send a sequence of Cartesian poses as a smooth arc motion.

        Sets mode/state once before the loop, then streams waypoints with
        wait=False on intermediate steps and wait=True on the final step,
        giving smooth continuous motion without controller resets.

        Args:
            waypoints: List of (position_m, quaternion_xyzw) tuples.
            speed: linear speed in mm/s.
            acc: linear acceleration in mm/s².
        """
        from scipy.spatial.transform import Rotation
        if not waypoints:
            return True
        with self.arm_lock:
            self.arm.set_mode(0)
            self.arm.set_state(0)
            for i, (pos, quat) in enumerate(waypoints):
                x_mm, y_mm, z_mm = [v * 1000.0 for v in pos]
                roll, pitch, yaw = Rotation.from_quat(quat).as_euler("xyz")
                is_last = (i == len(waypoints) - 1)
                code = self.arm.set_position(
                    x=x_mm, y=y_mm, z=z_mm,
                    roll=roll, pitch=pitch, yaw=yaw,
                    speed=speed, mvacc=acc,
                    is_radian=True, wait=is_last,
                )
                if code != 0:
                    return False
        return True

    def open_gripper(self, wait: bool = True, **_: Any) -> bool:
        """Open the xArm gripper."""
        return self._set_gripper(_GRIPPER_OPEN, wait=wait)

    def close_gripper(self, wait: bool = True, **_: Any) -> bool:
        """Close the xArm gripper."""
        return self._set_gripper(_GRIPPER_CLOSED, wait=wait)

    def disconnect(self) -> None:
        """Disconnect from the xArm controller."""
        try:
            self.arm.set_state(0)
        finally:
            self.arm.disconnect()

    def _set_gripper(self, position: int, wait: bool = True) -> bool:
        with self.arm_lock:
            if hasattr(self.arm, "set_gripper_mode"):
                self.arm.set_gripper_mode(0)
            if hasattr(self.arm, "set_gripper_enable"):
                self.arm.set_gripper_enable(True)
            if hasattr(self.arm, "clean_gripper_error"):
                self.arm.clean_gripper_error()
            if hasattr(self.arm, "set_gripper_speed"):
                self.arm.set_gripper_speed(2000)
            if hasattr(self.arm, "set_gripper_position"):
                code = self.arm.set_gripper_position(
                    position,
                    wait=wait,
                    auto_enable=True,
                )
                return code == 0
            if position == _GRIPPER_OPEN and hasattr(self.arm, "open_lite6_gripper"):
                return self.arm.open_lite6_gripper() == 0
            if position == _GRIPPER_CLOSED and hasattr(self.arm, "close_lite6_gripper"):
                return self.arm.close_lite6_gripper() == 0
        return False
