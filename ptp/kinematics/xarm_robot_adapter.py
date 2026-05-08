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
