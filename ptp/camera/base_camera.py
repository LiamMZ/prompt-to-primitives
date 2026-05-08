"""Abstract base class and core data types for camera interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
            "distortion": self.distortion.tolist() if self.distortion is not None else None,
        }

    def to_matrix(self) -> np.ndarray:
        """Return 3×3 camera intrinsics matrix K."""
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])


@dataclass
class CameraFrame:
    """Container for a single camera frame."""

    color: np.ndarray           # RGB (H, W, 3) uint8
    depth: Optional[np.ndarray] = None  # metres (H, W) float32
    timestamp: Optional[float] = None
    frame_id: Optional[int] = None


class BaseCamera(ABC):
    """Abstract base for all camera backends."""

    @abstractmethod
    def capture_frame(self) -> np.ndarray:
        """Return an RGB image (H, W, 3)."""

    @abstractmethod
    def get_depth(self) -> np.ndarray:
        """Return a depth image (H, W) in metres."""

    @abstractmethod
    def get_camera_intrinsics(self) -> CameraIntrinsics:
        """Return camera intrinsic parameters."""

    @abstractmethod
    def start(self) -> None:
        """Start the camera stream."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the camera stream and release resources."""

    def get_aligned_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (color, depth) as a pair.

        Override in subclasses for hardware-accelerated depth alignment.
        """
        return self.capture_frame(), self.get_depth()

    def get_frame(self) -> CameraFrame:
        """Return a CameraFrame with all available data."""
        color = self.capture_frame()
        try:
            depth = self.get_depth()
        except NotImplementedError:
            depth = None
        return CameraFrame(color=color, depth=depth)

    def is_depth_available(self) -> bool:
        try:
            self.get_depth()
            return True
        except NotImplementedError:
            return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
