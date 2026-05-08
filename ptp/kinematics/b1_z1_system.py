"""
B1 + Z1 + Robotiq system — single entry point for the full robot.

Composes:
    B1RobotInterface          — high-level locomotion
    Z1RobotInterface          — arm + gripper control
    B1Z1TransformCalculator   — geometry / frame conversions

The system keeps the transform calculator's B1 pose in sync automatically:
call sync_pose() at the top of any control loop iteration and the calculator
will reflect the latest odometry from the B1.

Typical usage
-------------
    system = B1Z1System.from_defaults()

    system.b1.stand_up(hold=1.5)

    # sync odometry into the transform calculator
    state = system.sync_pose()
    print("battery:", state.battery_soc, "%")

    # perception gave us an object position in front-camera frame
    p_z1 = system.tf.cam_to_z1_base(p_cam)

    # move the arm
    system.z1.move_j(q_target)
    system.z1.close_gripper()

    system.close()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

from ptp.kinematics.b1_robot_interface import B1RobotInterface, B1State
from ptp.kinematics.z1_robot_interface import Z1RobotInterface
from ptp.kinematics.b1_z1_transform_calculator import B1Z1TransformCalculator

logger = logging.getLogger(__name__)

_DEFAULT_B1_IP = "192.168.123.161"
_DEFAULT_Z1_URL = "http://192.168.123.220:12000/unitree/z1"


class B1Z1System:
    """
    Unified interface for the B1 quadruped + Z1 arm + Robotiq gripper.

    Attributes
    ----------
    b1 : B1RobotInterface
    z1 : Z1RobotInterface
    tf : B1Z1TransformCalculator
    """

    def __init__(
        self,
        b1: B1RobotInterface,
        z1: Z1RobotInterface,
        tf: B1Z1TransformCalculator,
    ) -> None:
        self.b1 = b1
        self.z1 = z1
        self.tf = tf
        logger.info("B1Z1System ready")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_defaults(
        cls,
        b1_ip: str = _DEFAULT_B1_IP,
        z1_url: str = _DEFAULT_Z1_URL,
        front_cam_extrinsic: Optional[Any] = None,
        z1_mount_offset: Optional[Any] = None,
        z1_urdf_path: Optional[Union[str, Path]] = None,
        sdk_path: Optional[str] = None,
    ) -> "B1Z1System":
        """
        Create a B1Z1System with network defaults.

        Args:
            b1_ip:               B1 high-level SDK server IP
            z1_url:              Z1 HTTP API base URL
            front_cam_extrinsic: T_body_cam  (4×4 or (pos, rot)); uses
                                 approximate defaults if None
            z1_mount_offset:     T_body_z1base (4×4 or (pos, rot)); uses
                                 approximate defaults if None
            z1_urdf_path:        path to standalone Z1 URDF for arm FK
            sdk_path:            path to unitree_legged_sdk Python bindings
        """
        b1 = B1RobotInterface(server_ip=b1_ip, sdk_path=sdk_path)
        z1 = Z1RobotInterface(url=z1_url)
        tf = B1Z1TransformCalculator(
            front_cam_extrinsic=front_cam_extrinsic,
            z1_mount_offset=z1_mount_offset,
            z1_urdf_path=z1_urdf_path,
        )
        return cls(b1=b1, z1=z1, tf=tf)

    # ------------------------------------------------------------------
    # State sync
    # ------------------------------------------------------------------

    def sync_pose(self) -> B1State:
        """
        Poll the B1 for its current odometry and push it into the transform
        calculator. Call this at the start of each control loop iteration.

        Returns the full B1State snapshot for convenience.
        """
        state = self.b1.get_state()
        # B1 quaternion is wxyz; _make_T expects xyzw — convert
        q_wxyz = state.quaternion          # [w, x, y, z]
        q_xyzw = q_wxyz[[1, 2, 3, 0]]     # [x, y, z, w]
        self.tf.set_b1_pose(state.position, q_xyzw)
        return state

    def sync_arm(self) -> None:
        """
        Poll the Z1 for current joint positions and push them into the
        transform calculator's FK solver (if a Z1 URDF is loaded).
        """
        q = self.z1.get_robot_joint_state()
        if q is not None:
            self.tf.set_z1_joints(q)

    def sync(self) -> B1State:
        """Sync both B1 odometry and Z1 joints. Returns B1State."""
        state = self.sync_pose()
        self.sync_arm()
        return state

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self.b1.close()
        except Exception:
            pass
        try:
            self.tf.cleanup()
        except Exception:
            pass
        logger.info("B1Z1System closed")

    def __enter__(self) -> "B1Z1System":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
