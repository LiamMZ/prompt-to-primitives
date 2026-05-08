"""Camera utility functions — factory, 3D projection, point cloud, depth visualisation."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ptp.camera.base_camera import BaseCamera, CameraIntrinsics

try:
    from ptp.camera.realsense_camera import RealSenseCamera
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_camera(
    camera_type: str = "realsense",
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    enable_depth: bool = True,
    **kwargs,
) -> BaseCamera:
    """Create a camera instance by type string.

    Args:
        camera_type: ``"realsense"`` (default) — add more backends here as needed.
        width: Frame width.
        height: Frame height.
        fps: Target frame rate.
        enable_depth: Whether to enable the depth stream.
        **kwargs: Forwarded to the camera constructor.

    Returns:
        A started BaseCamera instance.

    Raises:
        ValueError: If the requested camera type is not supported.
    """
    if camera_type == "realsense":
        if not REALSENSE_AVAILABLE:
            raise ValueError(
                "pyrealsense2 is not installed. Install with: pip install pyrealsense2"
            )
        return RealSenseCamera(
            width=width, height=height, fps=fps, enable_depth=enable_depth, **kwargs
        )
    raise ValueError(f"Unsupported camera type: {camera_type!r}")


# ---------------------------------------------------------------------------
# 3-D geometry helpers
# ---------------------------------------------------------------------------

def pixel_to_3d(u: int, v: int, depth: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Back-project a single pixel to a 3-D point in the camera frame.

    Args:
        u: Pixel column (x).
        v: Pixel row (y).
        depth: Metric depth at (u, v) in metres.
        intrinsics: Camera intrinsic parameters.

    Returns:
        3-D point ``[x, y, z]`` in metres (camera frame).
    """
    x = (u - intrinsics.cx) * depth / intrinsics.fx
    y = (v - intrinsics.cy) * depth / intrinsics.fy
    return np.array([x, y, depth])


def depth_image_to_point_cloud(
    depth_image: np.ndarray,
    intrinsics: CameraIntrinsics,
    color_image: Optional[np.ndarray] = None,
    max_depth: float = 3.0,
    stride: int = 1,
) -> np.ndarray:
    """Convert a depth image to a 3-D point cloud in the camera frame.

    Args:
        depth_image: (H, W) float32 array in metres.
        intrinsics: Camera intrinsic parameters.
        color_image: Optional (H, W, 3) uint8 RGB — appended as normalised colours.
        max_depth: Points beyond this depth are discarded.
        stride: Pixel stride for downsampling (1 = full resolution).

    Returns:
        (N, 3) array of 3-D points, or (N, 6) if *color_image* is provided
        (columns 3-5 are normalised RGB in [0, 1]).
    """
    h, w = depth_image.shape
    u_grid, v_grid = np.meshgrid(
        np.arange(0, w, stride, dtype=float),
        np.arange(0, h, stride, dtype=float),
    )
    d_grid = depth_image[::stride, ::stride]
    valid = (d_grid > 0.0) & (d_grid < max_depth)

    u_v = u_grid[valid]
    v_v = v_grid[valid]
    d_v = d_grid[valid]

    x = (u_v - intrinsics.cx) * d_v / intrinsics.fx
    y = (v_v - intrinsics.cy) * d_v / intrinsics.fy
    points = np.stack([x, y, d_v], axis=-1)

    if color_image is not None:
        colors = color_image[::stride, ::stride][valid] / 255.0
        points = np.concatenate([points, colors], axis=-1)

    return points


def estimate_object_depth(
    bbox: Tuple[float, float, float, float],
    depth_image: np.ndarray,
    method: str = "median",
) -> float:
    """Estimate object depth from a bounding-box region of the depth image.

    Args:
        bbox: ``(x_min, y_min, x_max, y_max)`` in pixel coordinates.
        depth_image: (H, W) float32 depth in metres.
        method: ``"median"`` (default), ``"mean"``, or ``"min"``.

    Returns:
        Estimated depth in metres, or 0.0 if no valid depth values exist.
    """
    x_min, y_min, x_max, y_max = (int(v) for v in bbox)
    region = depth_image[y_min:y_max, x_min:x_max]
    valid = region[(region > 0.0) & (region < 10.0)]
    if len(valid) == 0:
        return 0.0
    if method == "median":
        return float(np.median(valid))
    if method == "mean":
        return float(np.mean(valid))
    if method == "min":
        return float(np.min(valid))
    raise ValueError(f"Unknown depth estimation method: {method!r}")


def visualize_depth(depth_image: np.ndarray, max_depth: float = 2.0) -> np.ndarray:
    """Render a depth image as a colour-coded RGB array (JET colormap).

    Requires OpenCV (``pip install opencv-python``).

    Args:
        depth_image: (H, W) float32 depth in metres.
        max_depth: Depth corresponding to the maximum colour value.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    import cv2  # type: ignore
    normalized = np.clip(depth_image / max_depth * 255, 0, 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
