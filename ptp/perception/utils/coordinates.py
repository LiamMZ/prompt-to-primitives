"""Coordinate transformation utilities for 2D ↔ 3D conversions.

Conventions:
  - Normalised 2D: [y, x] in 0-1000 scale (VLM output).
  - Pixel 2D:      (pixel_y, pixel_x) in image space.
  - 3D camera:     [x, y, z] in metres using the pinhole model.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np


def normalized_to_pixel(
    normalized_pos: List[int],
    image_shape: Tuple[int, int],
) -> Tuple[int, int]:
    """Convert a VLM normalised [y, x] (0-1000) to (pixel_y, pixel_x).

    Args:
        normalized_pos: [y, x] in 0-1000 scale.
        image_shape: (height, width) of the target image.

    Returns:
        (pixel_y, pixel_x) clamped to image bounds.

    Example:
        >>> normalized_to_pixel([500, 750], (480, 640))
        (240, 480)
    """
    height, width = image_shape
    py = max(0, min(height - 1, int(normalized_pos[0] / 1000.0 * height)))
    px = max(0, min(width  - 1, int(normalized_pos[1] / 1000.0 * width)))
    return py, px


def pixel_to_normalized(
    pixel_pos: Tuple[int, int],
    image_shape: Tuple[int, int],
) -> List[int]:
    """Convert (pixel_y, pixel_x) to a VLM normalised [y, x] (0-1000).

    Example:
        >>> pixel_to_normalized((240, 480), (480, 640))
        [500, 750]
    """
    py, px = pixel_pos
    height, width = image_shape
    ny = max(0, min(1000, int(py / height * 1000.0)))
    nx = max(0, min(1000, int(px / width  * 1000.0)))
    return [ny, nx]


def compute_3d_position(
    position_2d: List[int],
    depth_frame: np.ndarray,
    camera_intrinsics: Any,
) -> Optional[np.ndarray]:
    """Back-project a normalised 2D point to 3D camera-frame coordinates.

    Args:
        position_2d: [y, x] in 0-1000 normalised scale.
        depth_frame: Depth image in metres (H, W).
        camera_intrinsics: Object with fx, fy, cx/ppx, cy/ppy attributes.

    Returns:
        np.ndarray [x, y, z] in metres, or None on invalid depth.

    Pinhole model:
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        z = depth[v, u]
    """
    try:
        h, w = depth_frame.shape[:2]
        py, px = normalized_to_pixel(position_2d, (h, w))

        z = float(np.ravel(depth_frame[py, px])[0])
        if z <= 0 or np.isnan(z):
            return None

        fx = getattr(camera_intrinsics, "fx", w / 2)
        fy = getattr(camera_intrinsics, "fy", h / 2)
        cx = getattr(camera_intrinsics, "ppx", getattr(camera_intrinsics, "cx", w / 2))
        cy = getattr(camera_intrinsics, "ppy", getattr(camera_intrinsics, "cy", h / 2))

        x = (px - cx) * z / fx
        y = (py - cy) * z / fy
        return np.array([x, y, z], dtype=np.float32)

    except Exception:
        return None


def project_3d_to_2d(
    position_3d: np.ndarray,
    camera_intrinsics: Any,
    image_shape: Tuple[int, int],
) -> Optional[List[int]]:
    """Project a 3D camera-frame point to normalised 2D [y, x] (0-1000).

    Returns None if the point is behind the camera (z ≤ 0).
    """
    try:
        x, y, z = position_3d
        if z <= 0:
            return None

        h, w = image_shape
        fx = getattr(camera_intrinsics, "fx", w / 2)
        fy = getattr(camera_intrinsics, "fy", h / 2)
        cx = getattr(camera_intrinsics, "ppx", getattr(camera_intrinsics, "cx", w / 2))
        cy = getattr(camera_intrinsics, "ppy", getattr(camera_intrinsics, "cy", h / 2))

        px = int(fx * (x / z) + cx)
        py = int(fy * (y / z) + cy)
        return pixel_to_normalized((py, px), image_shape)

    except Exception:
        return None


def batch_compute_3d_positions(
    positions_2d: List[List[int]],
    depth_frame: np.ndarray,
    camera_intrinsics: Any,
) -> List[Optional[np.ndarray]]:
    """Back-project a batch of normalised 2D points to 3D."""
    return [compute_3d_position(p, depth_frame, camera_intrinsics) for p in positions_2d]


def get_depth_at_normalized_position(
    position_2d: List[int],
    depth_frame: np.ndarray,
) -> Optional[float]:
    """Return the depth in metres at a normalised 2D position, or None if invalid."""
    try:
        h, w = depth_frame.shape[:2]
        py, px = normalized_to_pixel(position_2d, (h, w))
        d = float(depth_frame[py, px])
        return d if (d > 0 and not np.isnan(d)) else None
    except Exception:
        return None
