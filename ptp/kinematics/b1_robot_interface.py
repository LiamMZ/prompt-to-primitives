"""
Unitree B1 high-level control interface.

Wraps the unitree_legged_sdk Python bindings (robot_interface) and exposes a
clean API for high-level locomotion commands. Low-level motor control is
deliberately not exposed here.

SDK bindings location: unitree_legged_sdk/lib/python/amd64/robot_interface.so
Add that directory to sys.path before importing this module, or pass
sdk_path to the constructor.

Coordinate conventions (from HighCmd / HighState):
    velocity[0]  : forward speed  vx  (m/s), body frame
    velocity[1]  : lateral speed  vy  (m/s), body frame
    yawSpeed     : yaw rate            (rad/s), body frame
    euler[0/1/2] : roll / pitch / yaw  (rad), used in force-stand mode

Modes (HighCmd.mode):
    0  idle / default stand
    1  force stand  (body pose controlled via euler + bodyHeight)
    2  velocity walk (controlled via velocity + yawSpeed)
    5  stand down
    6  stand up
    7  damping
    8  recovery stand

GaitType (HighCmd.gaitType):
    0  idle
    1  trot
    2  trot running
    3  climb stair
    4  trot obstacle
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import os

import numpy as np

logger = logging.getLogger(__name__)

# SDK default network constants — override via environment variables:
#   B1_UDP_LOCAL_PORT, B1_UDP_SERVER_IP, B1_UDP_SERVER_PORT, B1_SDK_PATH
_HIGH_LEVEL = 0xEE
_UDP_LOCAL_PORT = int(os.environ.get("B1_UDP_LOCAL_PORT", 8080))
_UDP_SERVER_IP  = os.environ.get("B1_UDP_SERVER_IP",  "192.168.12.1")
_UDP_SERVER_PORT = int(os.environ.get("B1_UDP_SERVER_PORT", 8082))
_LOOP_DT = 0.002  # 500 Hz control loop

# HighCmd mode constants — exposed for callers
class Mode:
    IDLE = 0
    FORCE_STAND = 1
    VELOCITY_WALK = 2
    STAND_DOWN = 5
    STAND_UP = 6
    DAMPING = 7
    RECOVERY_STAND = 8

class GaitType:
    IDLE = 0
    TROT = 1
    TROT_RUN = 2
    STAIR = 3
    OBSTACLE = 4


@dataclass
class B1State:
    """Snapshot of HighState fields we care about."""
    stamp: float = 0.0

    # IMU
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))  # wxyz
    rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyroscope: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accelerometer: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Odometry (drifts over time — use only for relative motion)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # xyz metres
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # vx vy omega

    # Robot status
    mode: int = 0
    gait_type: int = 0
    body_height: float = 0.0
    foot_raise_height: float = 0.0
    yaw_speed: float = 0.0

    # Battery
    battery_soc: int = 0  # 0-100 %

    # Foot forces
    foot_force: np.ndarray = field(default_factory=lambda: np.zeros(4))


class B1RobotInterface:
    """
    High-level Python interface for the Unitree B1.

    Typical usage::

        robot = B1RobotInterface()
        robot.stand_up()
        time.sleep(2)
        robot.walk(vx=0.3, vy=0.0, yaw_rate=0.0)
        time.sleep(3)
        robot.stop()
        robot.stand_down()
        robot.close()
    """

    def __init__(
        self,
        sdk_path: Optional[str] = None,
        local_port: int = _UDP_LOCAL_PORT,
        server_ip: str = _UDP_SERVER_IP,
        server_port: int = _UDP_SERVER_PORT,
    ) -> None:
        self._load_sdk(sdk_path)

        self._udp = self._sdk.UDP(
            _HIGH_LEVEL, local_port, server_ip, server_port
        )
        self._cmd = self._sdk.HighCmd()
        self._state_raw = self._sdk.HighState()
        self._udp.InitCmdData(self._cmd)

        self._reset_cmd()
        logger.info(
            "B1RobotInterface ready — %s:%d", server_ip, server_port
        )

    # ------------------------------------------------------------------
    # SDK loading
    # ------------------------------------------------------------------

    def _load_sdk(self, sdk_path: Optional[str]) -> None:
        candidates = [
            sdk_path,
            os.environ.get("B1_SDK_PATH"),
            str(Path(__file__).parents[3] / "lib" / "python" / "amd64"),
            "/usr/local/lib/python/amd64",
        ]
        for p in candidates:
            if p and Path(p).exists() and p not in sys.path:
                sys.path.insert(0, p)

        try:
            import robot_interface as sdk
            self._sdk = sdk
        except ImportError as exc:
            raise ImportError(
                "unitree_legged_sdk Python bindings not found. "
                "Add the path to robot_interface.so to sdk_path or sys.path."
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_cmd(self) -> None:
        self._cmd.mode = Mode.IDLE
        self._cmd.gaitType = GaitType.IDLE
        self._cmd.speedLevel = 0
        self._cmd.footRaiseHeight = 0.0
        self._cmd.bodyHeight = 0.0
        self._cmd.euler = [0.0, 0.0, 0.0]
        self._cmd.velocity = [0.0, 0.0]
        self._cmd.yawSpeed = 0.0
        self._cmd.reserve = 0

    def _send_cmd(self) -> None:
        self._udp.SetSend(self._cmd)
        self._udp.Send()

    def _recv_state(self) -> None:
        self._udp.Recv()
        self._udp.GetRecv(self._state_raw)

    def _send_for(self, duration: float) -> None:
        """Send the current command repeatedly for `duration` seconds."""
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self._recv_state()
            self._send_cmd()
            time.sleep(_LOOP_DT)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> B1State:
        """Poll the robot and return a B1State snapshot."""
        self._recv_state()
        s = self._state_raw
        return B1State(
            stamp=time.time(),
            quaternion=np.array(s.imu.quaternion, dtype=float),   # wxyz
            rpy=np.array(s.imu.rpy, dtype=float),
            gyroscope=np.array(s.imu.gyroscope, dtype=float),
            accelerometer=np.array(s.imu.accelerometer, dtype=float),
            position=np.array(s.position, dtype=float),
            velocity=np.array(s.velocity, dtype=float),
            mode=int(s.mode),
            gait_type=int(s.gaitType),
            body_height=float(s.bodyHeight),
            foot_raise_height=float(s.footRaiseHeight),
            yaw_speed=float(s.yawSpeed),
            battery_soc=int(s.bms.SOC),
            foot_force=np.array(s.footForce, dtype=float),
        )

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (position_xyz, quaternion_wxyz) from onboard odometry.

        Note: B1 odometry drifts. Use an external localisation source for
        anything longer than a few metres.
        """
        state = self.get_state()
        return state.position.copy(), state.quaternion.copy()

    # ------------------------------------------------------------------
    # High-level locomotion commands
    # ------------------------------------------------------------------

    def recovery_stand(self, hold: float = 2.0) -> None:
        """Command the robot to stand up from an arbitrary pose."""
        self._reset_cmd()
        self._cmd.mode = Mode.RECOVERY_STAND
        self._send_for(hold)
        logger.info("recovery_stand complete")

    def stand_up(self, hold: float = 1.0) -> None:
        """Transition to standing position."""
        self._reset_cmd()
        self._cmd.mode = Mode.STAND_UP
        self._send_for(hold)
        logger.info("stand_up complete")

    def stand_down(self, hold: float = 1.0) -> None:
        """Lower the robot to the ground."""
        self._reset_cmd()
        self._cmd.mode = Mode.STAND_DOWN
        self._send_for(hold)
        logger.info("stand_down complete")

    def damping(self, hold: float = 1.0) -> None:
        """Enter damping mode (motors back-drivable)."""
        self._reset_cmd()
        self._cmd.mode = Mode.DAMPING
        self._send_for(hold)

    def stop(self, hold: float = 0.5) -> None:
        """Stop locomotion and return to idle stand."""
        self._reset_cmd()
        self._cmd.mode = Mode.IDLE
        self._send_for(hold)

    def walk(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        yaw_rate: float = 0.0,
        gait: int = GaitType.TROT,
        foot_raise: float = 0.0,
        body_height: float = 0.0,
        duration: Optional[float] = None,
    ) -> None:
        """
        Command continuous velocity walking.

        Args:
            vx:          forward speed   (m/s, trot: -1.1 ~ 1.5)
            vy:          lateral speed   (m/s, trot: -1.0 ~ 1.0)
            yaw_rate:    yaw rate        (rad/s, trot: -4.0 ~ 4.0)
            gait:        GaitType constant (default: TROT)
            foot_raise:  foot lift height delta from default (m, -0.06 ~ 0.03)
            body_height: body height delta from default      (m, -0.13 ~ 0.03)
            duration:    if given, block for this many seconds then return
        """
        self._reset_cmd()
        self._cmd.mode = Mode.VELOCITY_WALK
        self._cmd.gaitType = gait
        self._cmd.velocity = [float(vx), float(vy)]
        self._cmd.yawSpeed = float(yaw_rate)
        self._cmd.footRaiseHeight = float(foot_raise)
        self._cmd.bodyHeight = float(body_height)

        if duration is not None:
            self._send_for(duration)
        else:
            self._send_cmd()

    def set_body_pose(
        self,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        body_height: float = 0.0,
        duration: Optional[float] = None,
    ) -> None:
        """
        Force-stand mode: control body orientation and height.

        Args:
            roll, pitch, yaw: euler angles (rad, range ±0.75 each, yaw ±0.6)
            body_height: delta from default 0.31 m (m, range -0.13 ~ 0.03)
            duration: if given, block for this many seconds
        """
        self._reset_cmd()
        self._cmd.mode = Mode.FORCE_STAND
        self._cmd.euler = [float(roll), float(pitch), float(yaw)]
        self._cmd.bodyHeight = float(body_height)

        if duration is not None:
            self._send_for(duration)
        else:
            self._send_cmd()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Stop the robot and release resources."""
        try:
            self.stop()
        except Exception:
            pass
        logger.info("B1RobotInterface closed")

    def __enter__(self) -> "B1RobotInterface":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
