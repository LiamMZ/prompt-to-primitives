"""
RobocasaRobotInterface — duck-typed robot provider for TaskOrchestrator.

Exposes the same interface the orchestrator expects:
  get_robot_state()      → dict  (joint_pos, eef_pos, eef_quat, gripper_qpos, ...)
  get_robot_joint_state()→ np.ndarray
  get_robot_tcp_pose()   → (pos_array, quat_xyzw_array) | (None, None)
  get_camera_transform() → (pos, Rotation)  — eye-in-hand in world frame

The interface reads from the live obs dict injected by update(obs) after each
env.step()/env.reset(), so it stays in sync with the simulation without any
extra MuJoCo calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


class RobocasaRobotInterface:
    """
    Robot state provider wrapping a live robocasa observation.

    Args:
        env:         Robocasa environment (already reset).
        camera_name: Eye-in-hand camera name used for get_camera_transform().
                     The transform is derived from the EEF pose — MuJoCo camera
                     extrinsics are not exposed at obs level, so we use EEF as
                     a proxy (valid for eye-in-hand configurations).
    """

    def __init__(self, env: object, camera_name: str = "robot0_eye_in_hand") -> None:
        self._env = env
        self._camera_name = camera_name
        self._last_obs: Optional[Dict[str, Any]] = None

        # Fixed camera-to-EEF offset for PandaOmron eye-in-hand
        # (small offset forward along EEF x-axis)
        self._cam_offset = np.array([0.0, 0.0, -0.05])

    # ------------------------------------------------------------------
    # Frame injection
    # ------------------------------------------------------------------

    def update(self, obs: Dict[str, Any]) -> None:
        """Inject latest obs from env.reset() / env.step()."""
        self._last_obs = obs

    # ------------------------------------------------------------------
    # Orchestrator-required interface
    # ------------------------------------------------------------------

    def get_robot_state(self) -> Dict[str, Any]:
        """Return a JSON-serialisable robot state dict."""
        if self._last_obs is None:
            return {}
        obs = self._last_obs
        state: Dict[str, Any] = {}
        for key in (
            "robot0_joint_pos",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_base_pos",
            "robot0_base_quat",
        ):
            val = obs.get(key)
            if val is not None:
                state[key.replace("robot0_", "")] = np.asarray(val).tolist()
        # Alias 'joints' for PrimitiveExecutor snapshot compatibility
        if "joint_pos" in state:
            state["joints"] = state["joint_pos"]
        return state

    def get_robot_joint_state(self) -> Optional[np.ndarray]:
        """Return arm joint positions as np.ndarray."""
        if self._last_obs is None:
            return None
        val = self._last_obs.get("robot0_joint_pos")
        return np.asarray(val, dtype=float) if val is not None else None

    def get_robot_tcp_pose(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (eef_pos [3], eef_quat_xyzw [4]) or (None, None)."""
        if self._last_obs is None:
            return None, None
        pos = self._last_obs.get("robot0_eef_pos")
        quat = self._last_obs.get("robot0_eef_quat")  # robosuite: wxyz
        if pos is None or quat is None:
            return None, None
        pos_arr = np.asarray(pos, dtype=float)
        # Convert robosuite wxyz → scipy xyzw
        w, x, y, z = quat
        quat_xyzw = np.array([x, y, z, w], dtype=float)
        return pos_arr, quat_xyzw

    def get_camera_transform(self) -> Tuple[Optional[np.ndarray], Optional[Rotation]]:
        """
        Return (camera_pos_world [3], Rotation) for the eye-in-hand camera.

        Uses MuJoCo camera extrinsics when available, falling back to the EEF
        pose with a fixed camera-to-EEF offset.
        """
        # Try to get the actual MuJoCo camera pose
        try:
            sim = self._env.sim
            cam_id = sim.model.camera_name2id(self._camera_name)
            # cam_xpos: (ncam, 3) world position
            cam_pos = sim.data.cam_xpos[cam_id].copy()
            # cam_xmat: (ncam, 9) rotation matrix, row-major
            cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3).copy()
            rot = Rotation.from_matrix(cam_mat)
            return cam_pos, rot
        except Exception:
            pass

        # Fallback: EEF pose + offset
        pos_arr, quat_xyzw = self.get_robot_tcp_pose()
        if pos_arr is None:
            return None, None
        rot = Rotation.from_quat(quat_xyzw)  # xyzw
        cam_pos = pos_arr + rot.apply(self._cam_offset)
        return cam_pos, rot

    # ------------------------------------------------------------------
    # Convenience: ground-truth object positions for grounding
    # ------------------------------------------------------------------

    def get_object_positions(self) -> Dict[str, List[float]]:
        """
        Return {object_name: [x, y, z]} for all sim objects.

        Useful for seeding the registry with ground-truth positions before
        running GSAM2 perception (or as a fallback when perception is skipped).
        """
        positions: Dict[str, List[float]] = {}
        try:
            sim = self._env.sim
            for name, obj in self._env.objects.items():
                body_id = sim.model.body_name2id(obj.root_body)
                pos = sim.data.body_xpos[body_id].tolist()
                positions[name] = pos
        except Exception:
            pass
        return positions
