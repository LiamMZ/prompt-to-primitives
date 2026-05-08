"""
Unitree Z1 + Robotiq gripper hardware interface.

Communicates with the Z1 HTTP API. Handles only arm motion and gripper
commands — camera transforms are handled separately by B1Z1TransformCalculator
since the camera lives on the B1 body, not the arm.

Robot control:
  move_j(q, gripper_pos, speed)
  set_gripper(position, speed, force)
  open_gripper() / close_gripper()
  back_to_start()
  passive()
  label_run(label) / label_save(label)

State queries:
  get_robot_joint_state()  -> np.ndarray | None
  get_robot_state()        -> dict
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

_DEFAULT_URL = "http://192.168.123.220:12000/unitree/z1"
_N_JOINTS = 6


class Z1RobotInterface:
    """Real-hardware interface for the Unitree Z1 arm with Robotiq gripper."""

    def __init__(
        self,
        url: str = _DEFAULT_URL,
        timeout: float = 5.0,
    ) -> None:
        self.url = url
        self.timeout = timeout
        logger.info("Z1RobotInterface ready (url=%s)", self.url)

    # ------------------------------------------------------------------
    # Internal HTTP helper
    # ------------------------------------------------------------------

    def _send(self, func_name: str, args: Optional[Dict] = None) -> Optional[requests.Response]:
        payload: Dict[str, Any] = {"func": func_name, "args": args or {}}
        try:
            resp = requests.post(self.url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            logger.error("Z1 API error [%s]: %s", func_name, exc)
            return None

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_robot_joint_state(self) -> Optional[np.ndarray]:
        """Query current joint positions from the robot (radians)."""
        resp = self._send("getQ")
        if resp is None:
            return None
        try:
            data = resp.json()
            q = data if isinstance(data, list) else data.get("q") or data.get("joints")
            if q is not None:
                return np.array(q, dtype=float)
            logger.warning("getQ response has no joint data: %s", data)
        except (ValueError, AttributeError) as exc:
            logger.error("Failed to parse getQ response: %s", exc)
        return None

    def get_robot_state(self) -> Dict[str, Any]:
        """Return a JSON-serialisable state dict."""
        state: Dict[str, Any] = {"stamp": time.time(), "provider": "Z1RobotInterface"}
        joints = self.get_robot_joint_state()
        if joints is not None:
            state["joints"] = joints.tolist()
        return state

    # ------------------------------------------------------------------
    # Robot motion commands
    # ------------------------------------------------------------------

    def move_j(
        self,
        q: list,
        gripper_pos: float = 0.0,
        speed: float = 0.5,
    ) -> Optional[requests.Response]:
        """Joint-space move. q is 6 joint angles in radians."""
        if len(q) != _N_JOINTS:
            raise ValueError(f"Z1 expects {_N_JOINTS} joint angles, got {len(q)}")
        return self._send("MoveJ", {"q": list(q), "gripperPos": gripper_pos, "maxSpeed": speed})

    def set_gripper(
        self,
        position: int,
        speed: int = 128,
        force: int = 128,
    ) -> Optional[requests.Response]:
        """
        Command the Robotiq gripper.

        Args:
            position: 0 (open) – 255 (closed)
            speed:    0 – 255
            force:    0 – 255
        """
        return self._send("setGripper", {
            "position": max(0, min(255, int(position))),
            "speed":    max(0, min(255, int(speed))),
            "force":    max(0, min(255, int(force))),
        })

    def back_to_start(self) -> Optional[requests.Response]:
        return self._send("backToStart")

    def passive(self) -> Optional[requests.Response]:
        """Release motor torque (passive/gravity mode)."""
        return self._send("Passive")

    def label_run(self, label: str) -> Optional[requests.Response]:
        if len(label) >= 10:
            raise ValueError("Label must be fewer than 10 characters")
        return self._send("labelRun", {"label": label})

    def label_save(self, label: str) -> Optional[requests.Response]:
        if len(label) >= 10:
            raise ValueError("Label must be fewer than 10 characters")
        return self._send("labelSave", {"label": label})

    # ------------------------------------------------------------------
    # Gripper convenience helpers
    # ------------------------------------------------------------------

    def open_gripper(self, speed: int = 255, force: int = 255) -> Optional[requests.Response]:
        return self.set_gripper(0, speed, force)

    def close_gripper(self, speed: int = 255, force: int = 255) -> Optional[requests.Response]:
        return self.set_gripper(255, speed, force)


def create_z1_interface(
    url: str = _DEFAULT_URL,
    timeout: float = 5.0,
) -> Z1RobotInterface:
    return Z1RobotInterface(url=url, timeout=timeout)
