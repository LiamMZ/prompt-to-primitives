"""Surface normal estimation from depth images.

Given a depth image, camera intrinsics, and a 2-D center point (e.g. from
Molmo), back-projects a circular patch of depth pixels to 3-D and fits a
plane.  Returns a unit normal vector in the camera frame.

Two fitting methods are supported:
  pca   — fast, suitable for clean depth (RealSense with filtering)
  ransac — robust to outliers, better for noisy or mixed-surface patches

The returned normal always points toward the camera (positive-Z in camera
frame).  The wrist camera faces outward toward the scene, so +Z camera points
away from the robot into the scene — i.e. the normal points INTO the surface
from the robot's perspective.  Use +normal to push and -normal to pull.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def compute_surface_normal(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    center_yx: Tuple[float, float],
    radius_px: float = 30.0,
    method: str = "pca",
    min_points: int = 10,
    max_points: int = 2000,
    ransac_iterations: int = 100,
    ransac_threshold_m: float = 0.01,
) -> Tuple[Optional[np.ndarray], float]:
    """Estimate the surface normal at a depth-image location.

    Args:
        depth: (H, W) float32 depth in metres.
        fx, fy, cx, cy: Camera intrinsic parameters.
        center_yx: (row, col) of the Molmo-grounded surface point in pixels.
            Values may be fractional.
        radius_px: Circular patch radius in pixels (default 30).
        method: ``"pca"`` (default) or ``"ransac"``.
        min_points: Minimum valid depth points required (else returns None).
        max_points: Subsample cap for efficiency.
        ransac_iterations: Number of RANSAC trials (ignored for ``"pca"``).
        ransac_threshold_m: Inlier distance threshold for RANSAC (metres).

    Returns:
        ``(normal, confidence)`` where *normal* is a unit (3,) ndarray in
        the camera frame pointing toward the camera, and *confidence* is in
        [0, 1].  Returns ``(None, 0.0)`` if estimation fails.
    """
    h, w = depth.shape
    cy_px, cx_px = float(center_yx[0]), float(center_yx[1])
    r = float(radius_px)

    # Build pixel grid clipped to the patch bounding box.
    r0, r1 = max(0, int(cy_px - r)), min(h, int(cy_px + r) + 1)
    c0, c1 = max(0, int(cx_px - r)), min(w, int(cx_px + r) + 1)

    rows = np.arange(r0, r1, dtype=float)
    cols = np.arange(c0, c1, dtype=float)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")

    # Circular mask within patch.
    in_circle = (rr - cy_px) ** 2 + (cc - cx_px) ** 2 <= r ** 2
    d = depth[r0:r1, c0:c1]

    valid = in_circle & (d > 0.05) & (d < 4.0)
    if not np.any(valid):
        return None, 0.0

    d_v = d[valid].astype(float)
    u_v = cc[valid]
    v_v = rr[valid]

    # Back-project to camera-frame 3-D.
    x3 = (u_v - cx) * d_v / fx
    y3 = (v_v - cy) * d_v / fy
    pts = np.stack([x3, y3, d_v], axis=1)  # (N, 3)

    if len(pts) < min_points:
        return None, 0.0

    # Subsample for speed.
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]

    if method == "ransac":
        return _ransac_normal(pts, ransac_iterations, ransac_threshold_m)
    return _pca_normal(pts)


# ---------------------------------------------------------------------------
# Fitting backends
# ---------------------------------------------------------------------------

def _pca_normal(pts: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Fit a plane via PCA; normal = eigenvector of smallest eigenvalue."""
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = (centered.T @ centered) / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    normal = eigenvectors[:, 0]  # smallest eigenvalue → normal direction
    if normal[2] < 0:
        normal = -normal
    normal /= np.linalg.norm(normal) + 1e-9
    # Planarity confidence: ratio of smallest to second eigenvalue.
    confidence = float(1.0 - eigenvalues[0] / (eigenvalues[1] + 1e-9))
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return normal, confidence


def _ransac_normal(
    pts: np.ndarray,
    iterations: int,
    threshold_m: float,
) -> Tuple[Optional[np.ndarray], float]:
    """Fit a plane via RANSAC; returns best-inlier normal."""
    n = len(pts)
    best_normal = np.array([0.0, 0.0, 1.0])
    best_inliers = 0

    for _ in range(iterations):
        idx = np.random.choice(n, 3, replace=False)
        p1, p2, p3 = pts[idx]
        v1, v2 = p2 - p1, p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        d = -float(np.dot(normal, p1))
        distances = np.abs(pts @ normal + d)
        inliers = int(np.sum(distances < threshold_m))
        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal.copy()

    if best_normal[2] < 0:
        best_normal = -best_normal
    best_normal /= np.linalg.norm(best_normal) + 1e-9
    confidence = float(best_inliers / n)
    return best_normal, confidence


def transform_normal_to_base(
    normal_cam: np.ndarray,
    cam_rotation,
) -> np.ndarray:
    """Rotate a camera-frame normal into the robot base frame.

    Args:
        normal_cam: (3,) unit normal in camera frame.
        cam_rotation: ``scipy.spatial.transform.Rotation`` representing the
            camera orientation in the base frame (from FK).

    Returns:
        (3,) unit normal in base frame.
    """
    n_base = cam_rotation.apply(normal_cam)
    norm = np.linalg.norm(n_base)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return n_base / norm
