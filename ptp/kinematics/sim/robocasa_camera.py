"""
RobocasaCamera — BaseCamera adapter for a live robocasa/MuJoCo environment.

Provides RGB frames and estimated depth from the robot's eye-in-hand (or any
named MuJoCo camera) so the rest of the perception stack can treat the sim
identically to a RealSense camera.

Depth is estimated from the MuJoCo depth buffer and converted to metric metres
using the camera's near/far clip planes.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from ptp.camera.base_camera import BaseCamera, CameraIntrinsics


class RobocasaCamera(BaseCamera):
    """
    Wraps a live robocasa environment as a BaseCamera.

    The caller is responsible for stepping the environment; this class just
    reads the latest rendered frames from the obs dict.

    Args:
        env:         Robocasa environment (already reset).
        camera_name: MuJoCo camera to read (default: "robot0_eye_in_hand").
        width:       Rendered image width (must match env creation kwargs).
        height:      Rendered image height.
        near_clip:   MuJoCo near clip plane in metres (default: 0.01).
        far_clip:    MuJoCo far clip plane in metres (default: 10.0).
    """

    def __init__(
        self,
        env: object,
        camera_name: str = "robot0_eye_in_hand",
        width: int = 640,
        height: int = 480,
        near_clip: float = 0.01,
        far_clip: float = 10.0,
    ) -> None:
        self._env = env
        self._camera_name = camera_name
        self._width = width
        self._height = height
        self._near = near_clip
        self._far = far_clip
        self._last_obs: Optional[dict] = None

    # ------------------------------------------------------------------
    # Frame injection — call this after each env.step()/env.reset()
    # ------------------------------------------------------------------

    def update(self, obs: dict) -> None:
        """Inject the latest obs dict from env.reset() / env.step()."""
        self._last_obs = obs

    # ------------------------------------------------------------------
    # BaseCamera interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def capture_frame(self) -> np.ndarray:
        """Return the latest RGB frame (H, W, 3) uint8."""
        if self._last_obs is None:
            return np.zeros((self._height, self._width, 3), dtype=np.uint8)
        key = f"{self._camera_name}_image"
        img = self._last_obs.get(key)
        if img is None:
            return np.zeros((self._height, self._width, 3), dtype=np.uint8)
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        return img.copy()

    def get_depth(self) -> np.ndarray:
        """
        Return metric depth (H, W) float32 in metres.

        MuJoCo renders a linearised depth buffer in [0, 1]; we back-convert
        to metric using the near/far clip planes.
        """
        if self._last_obs is None:
            return np.zeros((self._height, self._width), dtype=np.float32)
        key = f"{self._camera_name}_depth"
        depth_buf = self._last_obs.get(key)
        if depth_buf is None:
            # Try rendering it ourselves from the sim
            depth_buf = self._render_depth_from_sim()
        if depth_buf is None:
            return np.full((self._height, self._width), self._far, dtype=np.float32)
        # MuJoCo depth buffer to metric: d_metric = near * far / (far - buf*(far-near))
        buf = np.asarray(depth_buf, dtype=np.float64)
        near, far = self._near, self._far
        metric = (near * far) / (far - buf * (far - near))
        return metric.astype(np.float32)

    def get_camera_intrinsics(self) -> CameraIntrinsics:
        """
        Derive pinhole intrinsics from the MuJoCo camera fovy.

        fovy is vertical field-of-view in degrees.  Aspect ratio is w/h.
        fx = fy = (h / 2) / tan(fovy/2)  (square pixels assumed).
        """
        fovy_deg = self._get_fovy()
        fovy_rad = math.radians(fovy_deg)
        fy = (self._height / 2.0) / math.tan(fovy_rad / 2.0)
        fx = fy  # MuJoCo uses square pixels
        cx = self._width / 2.0
        cy = self._height / 2.0
        return CameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=self._width, height=self._height,
        )

    def get_aligned_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (rgb, depth_metric) from the same obs snapshot."""
        return self.capture_frame(), self.get_depth()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_fovy(self) -> float:
        """Look up the camera's vertical FoV from the MuJoCo model."""
        try:
            sim = self._env.sim
            cam_id = sim.model.camera_name2id(self._camera_name)
            return float(sim.model.cam_fovy[cam_id])
        except Exception:
            return 60.0  # sensible fallback

    def _render_depth_from_sim(self) -> Optional[np.ndarray]:
        """Render a depth frame directly when camera_depths=False at env creation.

        robosuite's MjRenderContext binding does not expose depth rendering after
        env creation in all versions.  When unavailable we return None and callers
        receive the far-clip fallback (far_clip metres).  RGB perception works fine
        without metric depth; the planning pipeline handles None/flat depth safely.
        """
        try:
            sim = self._env.sim
            # robosuite MjSimWrapper exposes render() with depth kwarg on some builds
            result = sim.render(
                width=self._width,
                height=self._height,
                camera_name=self._camera_name,
                depth=True,
            )
            if isinstance(result, tuple) and len(result) == 2:
                _, depth_buf = result
            else:
                depth_buf = result
            if depth_buf is None:
                return None
            # robosuite renders bottom-up; flip vertically
            return np.flipud(np.asarray(depth_buf, dtype=np.float32))
        except Exception:
            return None
