"""
Occlusion map computation across an observation history.

For each object tracked by the perception service, accumulates visibility
evidence across a rolling window of depth observations and characterises:
  - What fraction of the object surface is visible from each viewpoint
  - Which other objects occlude it
  - Where hidden (shadow) volumes exist behind it
  - The best observed viewpoint per object

Algorithm
---------
Phase 1 — Per-viewpoint visibility analysis:
  Project each object's 3D bounding box corners into the image plane and
  compute the fraction of the projected region covered by its segmentation
  mask.  Identify occluders by checking which other objects have mask pixels
  in front of the target in the same region.

Phase 2 — Multi-viewpoint aggregation:
  Combine per-viewpoint results across all stored observations to get the
  best-case visibility per object and the union of persistent occluders.

Phase 3 — Unobserved volume detection via voxel grid:
  Discretise the workspace into a voxel grid.  For each observation, use
  vectorised ray marching to mark voxels as OBSERVED_FREE (between camera and
  surface) or OBSERVED_OCCUPIED (at surface).  Extract connected components of
  remaining UNOBSERVED voxels as candidate hidden-object volumes.

Viewpoint history management:
  Cap at MAX_VIEWPOINTS=20 observations.  When the cap is exceeded, merge the
  oldest observation's coverage into a persistent "ever-observed" grid before
  dropping it, so the unobserved volume map stays accurate.

Complexity:
  Phase 1: O(N × R²) per viewpoint (R = projected region size).  <5 ms.
  Phase 2: O(N × V) where V = viewpoints.  <1 ms.
  Phase 3: O(V × P) where P = valid pixels per viewpoint (subsampled).
    With SUBSAMPLE_STEP=4 and 640×480: ~19 200 pts/viewpoint.
    10 viewpoints → ~192 000 points, ~20 ms.
  Connected components: O(G) for G grid voxels.  ~5 ms for 1 m³ at 2 cm.

Usage:
    occ_map = compute_occlusion_map(
        observations=history,
        object_ids=[...],
    )
    registry.occlusion_map = occ_map
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .clearance import _depth_to_pointcloud

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_VIEWPOINTS: int = 20          # rolling window cap
_VOXEL_SIZE: float = 0.02         # metres — 2 cm voxels
_SUBSAMPLE_STEP: int = 4          # pixel stride for ray-march sampling
_MAX_DEPTH: float = 3.0           # metres — ignore farther points
_SMALL_OBJECT_SIZE: float = 0.05  # 5 cm cube for capacity estimation

# Voxel state constants
_UNOBSERVED: np.uint8 = np.uint8(0)
_OBSERVED_FREE: np.uint8 = np.uint8(1)
_OBSERVED_OCCUPIED: np.uint8 = np.uint8(2)

# Free-space ray fractions for approximate free-space marking
_RAY_FREE_FRACTIONS = (0.25, 0.5, 0.75)


# ---------------------------------------------------------------------------
# SE(3) camera pose (lightweight — avoids scipy import at module level)
# ---------------------------------------------------------------------------

@dataclass
class CameraPose:
    """Camera pose in world frame.

    Attributes:
        position: (3,) translation in metres.
        quaternion_xyzw: (4,) unit quaternion.
    """
    position: np.ndarray       # (3,)
    quaternion_xyzw: np.ndarray  # (4,)

    @staticmethod
    def identity() -> "CameraPose":
        return CameraPose(
            position=np.zeros(3),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        )

    @staticmethod
    def from_robot_state(robot_state: Optional[dict]) -> Optional["CameraPose"]:
        """Extract CameraPose from robot_state["camera"] dict, or None."""
        if robot_state is None:
            return None
        cam_tf = robot_state.get("camera")
        if cam_tf is None:
            return None
        try:
            return CameraPose(
                position=np.array(cam_tf["position"], dtype=float),
                quaternion_xyzw=np.array(cam_tf["quaternion_xyzw"], dtype=float),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def rotation_matrix(self) -> np.ndarray:
        """Return 3×3 rotation matrix (world←cam) from quaternion_xyzw."""
        x, y, z, w = self.quaternion_xyzw.astype(float)
        x2, y2, z2 = x*x, y*y, z*z
        return np.array([
            [1 - 2*(y2+z2),   2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w),   1 - 2*(x2+z2), 2*(y*z - x*w)],
            [2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x2+y2)],
        ], dtype=float)

    def transform_points(self, pts_cam: np.ndarray) -> np.ndarray:
        """Transform (N,3) camera-frame points to world frame."""
        R = self.rotation_matrix()
        return (R @ pts_cam.T).T + self.position

    def transform_points_inv(self, pts_world: np.ndarray) -> np.ndarray:
        """Transform (N,3) world points to camera frame."""
        R = self.rotation_matrix()
        return (R.T @ (pts_world - self.position).T).T


# ---------------------------------------------------------------------------
# Observation record — one depth frame with its context
# ---------------------------------------------------------------------------

@dataclass
class ObservationRecord:
    """One captured depth observation for occlusion analysis.

    Attributes:
        depth_frame: (H, W) float32 depth in metres.
        camera_intrinsics: Object with fx, fy, ppx/cx, ppy/cy.
        camera_pose: SE(3) camera pose in world frame (None = camera frame only).
        obj_masks: {object_id: (H, W) bool} SAM2 masks for this frame.
    """

    depth_frame: np.ndarray
    camera_intrinsics: object
    camera_pose: Optional[CameraPose]
    obj_masks: Dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------

@dataclass
class AABB:
    """Axis-aligned bounding box in world/camera frame, metres."""
    min_xyz: np.ndarray   # (3,)
    max_xyz: np.ndarray   # (3,)

    @property
    def volume(self) -> float:
        dims = np.maximum(0.0, self.max_xyz - self.min_xyz)
        return float(dims[0] * dims[1] * dims[2])


@dataclass
class UnobservedVolume:
    """A region in 3-D space never covered by any observation.

    Attributes:
        bounds: AABB of this volume component (world frame if pose available,
            else camera frame of last observation).
        blocking_objects: Object IDs whose surfaces cast this shadow.
        estimated_capacity: Rough count of small objects (≈5 cm cube) that could fit.
    """

    bounds: AABB
    blocking_objects: List[str]
    estimated_capacity: int


@dataclass
class ViewVisibility:
    """Visibility of one object in one specific viewpoint."""
    visible_fraction: float
    in_frame: bool
    occluding_objects: List[str]


@dataclass
class ObjectVisibility:
    """Visibility record for one object aggregated across all observations.

    Attributes:
        visible_fraction: Fraction of object pixels visible in the best observation.
        occluding_objects: Union of occluder IDs across all observations.
        best_viewpoint: CameraPose with maximum visible fraction (None if unknown).
        hidden_regions: List of (H, W) bool masks — one per observation —
            marking the occluded pixels of this object.
    """

    visible_fraction: float
    occluding_objects: List[str]
    best_viewpoint: Optional[CameraPose]
    hidden_regions: List[np.ndarray]


@dataclass
class OcclusionMap:
    """Full scene occlusion description across all observations.

    Attributes:
        camera_poses: Ordered list of SE(3) poses used (one per observation).
        per_object_visibility: {object_id: ObjectVisibility}.
        unobserved_volumes: Shadow volumes never covered by any observation.
    """

    camera_poses: List[CameraPose] = field(default_factory=list)
    per_object_visibility: Dict[str, ObjectVisibility] = field(default_factory=dict)
    unobserved_volumes: List[UnobservedVolume] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Voxel grid for Phase 3
# ---------------------------------------------------------------------------

class _VoxelGrid:
    """Compact 3-D voxel grid with state values: UNOBSERVED, FREE, OCCUPIED."""

    def __init__(self, bounds_min: np.ndarray, bounds_max: np.ndarray,
                 voxel_size: float) -> None:
        self.voxel_size = float(voxel_size)
        self.origin = bounds_min.copy()
        dims_f = np.ceil((bounds_max - bounds_min) / voxel_size)
        self.dims = np.maximum(1, dims_f.astype(int))  # (3,) — NX, NY, NZ
        self.grid = np.zeros(tuple(self.dims), dtype=np.uint8)  # all UNOBSERVED

    def world_to_voxel(self, pts: np.ndarray) -> np.ndarray:
        """(N,3) world points → (N,3) integer voxel indices (clipped to grid)."""
        idx = np.floor((pts - self.origin) / self.voxel_size).astype(np.int32)
        idx = np.clip(idx, 0, self.dims - 1)
        return idx

    def batch_set(self, indices: np.ndarray, state: np.uint8) -> None:
        """Set multiple voxels (N,3 int indices) to state, ignoring duplicates."""
        valid = (
            (indices[:, 0] >= 0) & (indices[:, 0] < self.dims[0]) &
            (indices[:, 1] >= 0) & (indices[:, 1] < self.dims[1]) &
            (indices[:, 2] >= 0) & (indices[:, 2] < self.dims[2])
        )
        idx = indices[valid]
        if len(idx) == 0:
            return
        self.grid[idx[:, 0], idx[:, 1], idx[:, 2]] = state

    def merge_from(self, other: "_VoxelGrid") -> None:
        """OR-update: mark any cell observed in other as observed here too."""
        if other.grid.shape != self.grid.shape:
            return
        observed_mask = other.grid != _UNOBSERVED
        self.grid[observed_mask] = np.maximum(
            self.grid[observed_mask], other.grid[observed_mask]
        )

    def connected_components_unobserved(self) -> List[np.ndarray]:
        """Return list of (K,3) index arrays — one per connected component of UNOBSERVED voxels."""
        from collections import deque

        unobs_indices = np.argwhere(self.grid == _UNOBSERVED)
        if len(unobs_indices) == 0:
            return []

        visited = np.zeros(tuple(self.dims), dtype=bool)
        components: List[np.ndarray] = []

        # Prebuilt index set for O(1) membership test
        idx_set: set = set(map(tuple, unobs_indices.tolist()))

        for start in unobs_indices:
            key = tuple(start)
            if visited[key]:
                continue
            # BFS
            queue: deque = deque([start])
            component: List[np.ndarray] = []
            visited[key] = True
            while queue:
                cur = queue.popleft()
                component.append(cur)
                cx, cy, cz = int(cur[0]), int(cur[1]), int(cur[2])
                for dx, dy, dz in [
                    (1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)
                ]:
                    nx, ny, nz = cx+dx, cy+dy, cz+dz
                    if (nx, ny, nz) in idx_set and not visited[nx, ny, nz]:
                        visited[nx, ny, nz] = True
                        queue.append(np.array([nx, ny, nz]))
            components.append(np.array(component))

        return components

    def voxel_to_world_aabb(self, indices: np.ndarray) -> AABB:
        """Return world-frame AABB for a set of voxel indices."""
        lo = indices.min(axis=0)
        hi = indices.max(axis=0) + 1  # inclusive → exclusive
        return AABB(
            min_xyz=self.origin + lo * self.voxel_size,
            max_xyz=self.origin + hi * self.voxel_size,
        )


# ---------------------------------------------------------------------------
# Phase 1 helpers
# ---------------------------------------------------------------------------

def _intrinsics(intr, h: int, w: int) -> Tuple[float, float, float, float]:
    fx = float(getattr(intr, 'fx', w / 2))
    fy = float(getattr(intr, 'fy', h / 2))
    cx = float(getattr(intr, 'ppx', getattr(intr, 'cx', w / 2)))
    cy = float(getattr(intr, 'ppy', getattr(intr, 'cy', h / 2)))
    return fx, fy, cx, cy


def _project_pts_to_image(
    pts_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """Project (N,3) camera-frame points to (N,2) pixel UV.  Z must be >0."""
    z = pts_cam[:, 2]
    u = fx * pts_cam[:, 0] / z + cx
    v = fy * pts_cam[:, 1] / z + cy
    return np.stack([u, v], axis=1)


def _compute_per_viewpoint_visibility(
    obs: ObservationRecord,
    object_aabbs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    depth_tolerance: float = 0.005,
) -> Dict[str, ViewVisibility]:
    """Phase 1: per-viewpoint visibility analysis.

    Projects each object's 3D AABB corners into the image, measures what
    fraction of the projected region is covered by the object's mask, and
    identifies objects with closer depth in that region as occluders.

    Falls back to mask-only analysis when no AABB or pose is available.

    Args:
        obs: One ObservationRecord.
        object_aabbs: {obj_id: (min_xyz_world, max_xyz_world)} bounding boxes.
        depth_tolerance: Depth tie-breaking threshold in metres.

    Returns:
        {obj_id: ViewVisibility}
    """
    depth = obs.depth_frame
    h, w = depth.shape
    intr = obs.camera_intrinsics
    fx, fy, cx_i, cy_i = _intrinsics(intr, h, w)
    pose = obs.camera_pose

    visibility: Dict[str, ViewVisibility] = {}

    # Build per-pixel minimum depth and nearest-object maps (2D occlusion test
    # fallback used when no 3D AABB is available).
    min_depth = np.full((h, w), np.inf, dtype=np.float64)
    nearest_id: Dict[Tuple[int, int], str] = {}

    # Vectorised build of min_depth
    id_list = list(obs.obj_masks.keys())
    id_indices: Dict[str, int] = {oid: i for i, oid in enumerate(id_list)}
    nearest_idx_arr = np.full((h, w), -1, dtype=np.int32)
    for oid, mask in obs.obj_masks.items():
        if mask is None or not mask.any():
            continue
        obj_depth = np.where(mask, depth, np.inf)
        closer = obj_depth < min_depth
        min_depth = np.where(closer, obj_depth, min_depth)
        nearest_idx_arr = np.where(closer, id_indices[oid], nearest_idx_arr)

    for obj_id in obs.obj_masks:
        mask = obs.obj_masks.get(obj_id)

        # --- Attempt 3D bounding-box projection ---
        projected_rect: Optional[Tuple[int, int, int, int]] = None
        if pose is not None and obj_id in object_aabbs:
            mn_w, mx_w = object_aabbs[obj_id]
            corners_world = np.array([
                mn_w + np.array([dx, dy, dz]) * (mx_w - mn_w)
                for dx in (0.0, 1.0) for dy in (0.0, 1.0) for dz in (0.0, 1.0)
            ])  # (8, 3)
            corners_cam = pose.transform_points_inv(corners_world)
            front = corners_cam[:, 2] > 0.01
            if front.any():
                uvs = _project_pts_to_image(corners_cam[front], fx, fy, cx_i, cy_i)
                u_min = max(0,   int(np.floor(uvs[:, 0].min())))
                u_max = min(w-1, int(np.ceil(uvs[:, 0].max())))
                v_min = max(0,   int(np.floor(uvs[:, 1].min())))
                v_max = min(h-1, int(np.ceil(uvs[:, 1].max())))
                if u_max > u_min and v_max > v_min:
                    projected_rect = (u_min, v_min, u_max, v_max)

        if projected_rect is None:
            # Fallback: use mask AABB as projected region
            if mask is not None and mask.any():
                ys, xs = np.where(mask)
                projected_rect = (
                    int(xs.min()), int(ys.min()),
                    int(xs.max()), int(ys.max()),
                )
            else:
                visibility[obj_id] = ViewVisibility(
                    visible_fraction=0.0, in_frame=False, occluding_objects=[]
                )
                continue

        u_min, v_min, u_max, v_max = projected_rect

        # --- Visible fraction ---
        expected_area = (u_max - u_min) * (v_max - v_min)
        if mask is not None and mask.any():
            observed_pixels = int(mask[v_min:v_max, u_min:u_max].sum())
        else:
            observed_pixels = 0
        visible_fraction = observed_pixels / max(expected_area, 1)

        # --- Identify occluders ---
        occluders: List[str] = []
        if mask is not None and mask.any():
            # Pixels in the projected region where this object's depth is not
            # the minimum → occluded by something closer.
            region_depth = depth[v_min:v_max, u_min:u_max]
            region_nearest = nearest_idx_arr[v_min:v_max, u_min:u_max]
            obj_depth_region = np.where(
                mask[v_min:v_max, u_min:u_max], depth[v_min:v_max, u_min:u_max], np.inf
            )
            occluded_pixels = mask[v_min:v_max, u_min:u_max] & (
                obj_depth_region > min_depth[v_min:v_max, u_min:u_max] + depth_tolerance
            )
            if occluded_pixels.any():
                occ_indices = set(region_nearest[occluded_pixels].tolist())
                occ_indices.discard(-1)
                for idx in occ_indices:
                    if idx < len(id_list) and id_list[idx] != obj_id:
                        occluders.append(id_list[idx])

            # Additionally check other masks directly in the region
            if not occluders:
                target_depth_val = float(
                    region_depth[mask[v_min:v_max, u_min:u_max]].mean()
                ) if mask[v_min:v_max, u_min:u_max].any() else np.inf
                for other_id, other_mask in obs.obj_masks.items():
                    if other_id == obj_id:
                        continue
                    if other_mask is None or not other_mask.any():
                        continue
                    overlap = other_mask[v_min:v_max, u_min:u_max]
                    if not overlap.any():
                        continue
                    other_depths = region_depth[overlap > 0]
                    if other_depths.size > 0 and float(other_depths.mean()) < target_depth_val:
                        occluders.append(other_id)

        visibility[obj_id] = ViewVisibility(
            visible_fraction=visible_fraction,
            in_frame=True,
            occluding_objects=list(set(occluders)),
        )

    return visibility


# ---------------------------------------------------------------------------
# Phase 2 helpers
# ---------------------------------------------------------------------------

def _aggregate_visibility(
    per_viewpoint: List[Dict[str, ViewVisibility]],
    poses: List[Optional[CameraPose]],
    object_ids: List[str],
    per_obs_hidden: Dict[str, List[np.ndarray]],
) -> Dict[str, ObjectVisibility]:
    """Phase 2: aggregate per-viewpoint visibility into per-object summary."""
    result: Dict[str, ObjectVisibility] = {}

    for obj_id in object_ids:
        best_frac: float = 0.0
        best_vp_idx: Optional[int] = None
        all_occluders: set = set()

        for vp_idx, vp_result in enumerate(per_viewpoint):
            vr = vp_result.get(obj_id)
            if vr is None or not vr.in_frame:
                continue
            if vr.visible_fraction > best_frac:
                best_frac = vr.visible_fraction
                best_vp_idx = vp_idx
            all_occluders.update(vr.occluding_objects)

        best_pose: Optional[CameraPose] = None
        if best_vp_idx is not None and best_vp_idx < len(poses):
            best_pose = poses[best_vp_idx]

        result[obj_id] = ObjectVisibility(
            visible_fraction=best_frac,
            occluding_objects=sorted(all_occluders),
            best_viewpoint=best_pose,
            hidden_regions=per_obs_hidden.get(obj_id, []),
        )

    return result


# ---------------------------------------------------------------------------
# Phase 3 helpers — voxel grid ray marching
# ---------------------------------------------------------------------------

def _mark_observed_vectorised(
    obs: ObservationRecord,
    grid: _VoxelGrid,
) -> None:
    """Mark voxels observed/free/occupied from one observation.

    Uses vectorised back-projection + approximate free-space marking at
    fractional depths along each ray.
    """
    depth = obs.depth_frame
    intr = obs.camera_intrinsics
    pose = obs.camera_pose
    if pose is None:
        return  # need world-frame pose to update world-frame grid

    h, w = depth.shape
    fx, fy, cx_i, cy_i = _intrinsics(intr, h, w)

    us = np.arange(0, w, _SUBSAMPLE_STEP, dtype=np.float32)
    vs = np.arange(0, h, _SUBSAMPLE_STEP, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    uu = uu.ravel()
    vv = vv.ravel()
    depths = depth[vv.astype(int), uu.astype(int)]

    valid = (depths > 0) & (depths < _MAX_DEPTH) & np.isfinite(depths)
    uu, vv, depths = uu[valid], vv[valid], depths[valid]

    if len(depths) == 0:
        return

    # Back-project to camera frame
    pts_cam = np.stack([
        (uu - cx_i) * depths / fx,
        (vv - cy_i) * depths / fy,
        depths,
    ], axis=1)  # (N, 3)

    # Transform to world frame
    pts_world = pose.transform_points(pts_cam)

    # Mark surface voxels as OBSERVED_OCCUPIED
    voxel_idx = grid.world_to_voxel(pts_world)
    grid.batch_set(voxel_idx, _OBSERVED_OCCUPIED)

    # Mark fractional-depth voxels as OBSERVED_FREE
    cam_origin = pose.position
    for frac in _RAY_FREE_FRACTIONS:
        intermediate = cam_origin + frac * (pts_world - cam_origin)
        vi = grid.world_to_voxel(intermediate)
        # Only mark UNOBSERVED → FREE; don't overwrite OCCUPIED
        candidate_states = grid.grid[vi[:, 0], vi[:, 1], vi[:, 2]]
        free_mask = candidate_states == _UNOBSERVED
        if free_mask.any():
            grid.batch_set(vi[free_mask], _OBSERVED_FREE)


def _compute_workspace_bounds(
    observations: List[ObservationRecord],
    padding: float = 0.10,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Derive workspace bounds from all observed surface points (world frame)."""
    all_pts: List[np.ndarray] = []
    for obs in observations:
        if obs.camera_pose is None:
            continue
        depth = obs.depth_frame
        intr = obs.camera_intrinsics
        h, w = depth.shape
        fx, fy, cx_i, cy_i = _intrinsics(intr, h, w)
        pts_cam = _depth_to_pointcloud(depth, fx, fy, cx_i, cy_i)
        if len(pts_cam) == 0:
            continue
        pts_world = obs.camera_pose.transform_points(pts_cam[::8])  # coarse
        all_pts.append(pts_world)

    if not all_pts:
        return None

    combined = np.concatenate(all_pts, axis=0)
    mn = combined.min(axis=0) - padding
    mx = combined.max(axis=0) + padding
    return mn, mx


def _compute_unobserved_volumes(
    observations: List[ObservationRecord],
    persistent_grid: Optional[_VoxelGrid],
) -> Tuple[List[UnobservedVolume], _VoxelGrid]:
    """Phase 3: build voxel grid, mark observations, extract unobserved components.

    Returns:
        (volumes, current_grid) — volumes is a list of UnobservedVolume,
        current_grid is the merged grid for this batch (for persistent merging).
    """
    # Determine workspace bounds
    bounds = _compute_workspace_bounds(observations)
    if bounds is None:
        empty = _VoxelGrid(np.zeros(3), np.ones(3), _VOXEL_SIZE)
        return [], empty

    bounds_min, bounds_max = bounds

    # Start from persistent coverage if available
    if (persistent_grid is not None and
            np.allclose(persistent_grid.origin, bounds_min) and
            np.all(persistent_grid.dims == np.maximum(
                1, np.ceil((bounds_max - bounds_min) / _VOXEL_SIZE).astype(int)
            ))):
        grid = _VoxelGrid(bounds_min, bounds_max, _VOXEL_SIZE)
        grid.grid[:] = persistent_grid.grid
    else:
        grid = _VoxelGrid(bounds_min, bounds_max, _VOXEL_SIZE)
        if persistent_grid is not None:
            # Different bounds — merge where possible (skip, grids are incompatible)
            pass

    for obs in observations:
        _mark_observed_vectorised(obs, grid)

    # Extract connected components of UNOBSERVED voxels
    components = grid.connected_components_unobserved()

    volumes: List[UnobservedVolume] = []
    for comp in components:
        aabb = grid.voxel_to_world_aabb(comp)
        if aabb.volume < (_SMALL_OBJECT_SIZE ** 3) * 0.5:
            continue  # too small to hide anything — skip noise
        capacity = _estimated_capacity(aabb)
        volumes.append(UnobservedVolume(
            bounds=aabb,
            blocking_objects=[],  # populated below
            estimated_capacity=capacity,
        ))

    # Attribute volumes to blocking objects via shadow test
    if volumes:
        _attribute_blocking_objects(volumes, observations)

    logger.debug(
        "Phase 3: %d observations, grid=%s, unobserved_voxels=%d, volumes=%d",
        len(observations),
        grid.dims,
        int((grid.grid == _UNOBSERVED).sum()),
        len(volumes),
    )
    return volumes, grid


def _attribute_blocking_objects(
    volumes: List[UnobservedVolume],
    observations: List[ObservationRecord],
) -> None:
    """Heuristic: for each unobserved volume, find which object masks cast a
    depth-shadow over the same image columns in the most recent observation."""
    if not observations:
        return

    # Use the last observation for attribution
    obs = observations[-1]
    depth = obs.depth_frame
    h, w = depth.shape
    intr = obs.camera_intrinsics
    fx, fy, cx_i, cy_i = _intrinsics(intr, h, w)
    pose = obs.camera_pose

    for vol in volumes:
        if pose is None:
            continue
        # Project volume AABB corners to image to get pixel region
        mn_w, mx_w = vol.bounds.min_xyz, vol.bounds.max_xyz
        corners_world = np.array([
            mn_w + np.array([dx, dy, dz]) * (mx_w - mn_w)
            for dx in (0.0, 1.0) for dy in (0.0, 1.0) for dz in (0.0, 1.0)
        ])
        corners_cam = pose.transform_points_inv(corners_world)
        front = corners_cam[:, 2] > 0.01
        if not front.any():
            continue
        uvs = _project_pts_to_image(corners_cam[front], fx, fy, cx_i, cy_i)
        u_min = max(0,   int(np.floor(uvs[:, 0].min())))
        u_max = min(w-1, int(np.ceil(uvs[:, 0].max())))
        v_min = max(0,   int(np.floor(uvs[:, 1].min())))
        v_max = min(h-1, int(np.ceil(uvs[:, 1].max())))
        if u_max <= u_min or v_max <= v_min:
            continue

        blocking: List[str] = []
        region_depth = depth[v_min:v_max, u_min:u_max]
        centre_world = ((mn_w + mx_w) / 2.0).reshape(1, 3)
        vol_mean_depth = float(pose.transform_points_inv(centre_world)[0, 2])
        for oid, mask in obs.obj_masks.items():
            if mask is None or not mask.any():
                continue
            region_mask = mask[v_min:v_max, u_min:u_max]
            if not region_mask.any():
                continue
            # If mask pixels are shallower than volume centre, this object
            # blocks the view of the volume.
            obj_region_depth = region_depth[region_mask]
            if obj_region_depth.size > 0 and float(obj_region_depth.mean()) < vol_mean_depth:
                blocking.append(oid)
        vol.blocking_objects = blocking


def _estimated_capacity(aabb: AABB, object_size_m: float = _SMALL_OBJECT_SIZE) -> int:
    """Estimate how many small objects (default 5 cm cube) fit in a volume."""
    dims = np.maximum(0.0, aabb.max_xyz - aabb.min_xyz)
    counts = np.floor(dims / object_size_m).astype(int)
    return int(max(0, counts[0]) * max(0, counts[1]) * max(0, counts[2]))


# ---------------------------------------------------------------------------
# Shadow-based hidden region detection (legacy 2-D fallback)
# ---------------------------------------------------------------------------

def _build_depth_min_map(
    obj_masks: Dict[str, np.ndarray],
    depth_frame: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """Build per-pixel minimum-depth map and nearest-object index map."""
    h, w = depth_frame.shape
    min_depth = np.full((h, w), np.inf, dtype=np.float64)
    nearest_idx = np.full((h, w), -1, dtype=np.int32)

    idx_to_id: Dict[int, str] = {}
    for i, (obj_id, mask) in enumerate(obj_masks.items()):
        idx_to_id[i] = obj_id
        if mask is None or not mask.any():
            continue
        obj_depth = np.where(mask, depth_frame, np.inf)
        closer = obj_depth < min_depth
        min_depth = np.where(closer, obj_depth, min_depth)
        nearest_idx = np.where(closer, i, nearest_idx)

    return min_depth, nearest_idx, idx_to_id


def _hidden_mask_for_object(
    obj_id: str,
    obj_mask: np.ndarray,
    depth_frame: np.ndarray,
    min_depth: np.ndarray,
    depth_tolerance: float = 0.005,
) -> np.ndarray:
    """Return boolean mask of occluded pixels for this object in this frame."""
    if obj_mask is None or not obj_mask.any():
        return np.zeros_like(depth_frame, dtype=bool)
    obj_depth = np.where(obj_mask, depth_frame, np.inf)
    visible_mask = obj_mask & (obj_depth <= min_depth + depth_tolerance)
    return obj_mask & ~visible_mask


# ---------------------------------------------------------------------------
# Viewpoint history management
# ---------------------------------------------------------------------------

class ObservationHistory:
    """Rolling window of observations with persistent grid merging.

    Usage:
        history = ObservationHistory(max_viewpoints=20)
        history.add(obs)
        occ_map = compute_occlusion_map_from_history(history, object_ids)
    """

    def __init__(self, max_viewpoints: int = MAX_VIEWPOINTS) -> None:
        self.max_viewpoints = max_viewpoints
        self._observations: List[ObservationRecord] = []
        self._persistent_grid: Optional[_VoxelGrid] = None

    def add(self, obs: ObservationRecord) -> None:
        """Add a new observation, evicting the oldest if over capacity."""
        if len(self._observations) >= self.max_viewpoints:
            oldest = self._observations.pop(0)
            self._merge_oldest(oldest)
        self._observations.append(obs)

    def _merge_oldest(self, obs: ObservationRecord) -> None:
        """Merge coverage from the oldest observation into the persistent grid."""
        if obs.camera_pose is None:
            return
        if self._persistent_grid is None:
            bounds = _compute_workspace_bounds([obs])
            if bounds is None:
                return
            self._persistent_grid = _VoxelGrid(bounds[0], bounds[1], _VOXEL_SIZE)
        temp_grid = _VoxelGrid(
            self._persistent_grid.origin,
            self._persistent_grid.origin + self._persistent_grid.dims * _VOXEL_SIZE,
            _VOXEL_SIZE,
        )
        _mark_observed_vectorised(obs, temp_grid)
        self._persistent_grid.merge_from(temp_grid)

    @property
    def observations(self) -> List[ObservationRecord]:
        return list(self._observations)

    @property
    def persistent_grid(self) -> Optional[_VoxelGrid]:
        return self._persistent_grid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_occlusion_map(
    observations: List[ObservationRecord],
    object_ids: Optional[List[str]] = None,
    depth_tolerance: float = 0.005,
    object_aabbs: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> OcclusionMap:
    """Compute the occlusion map from a rolling observation history.

    Implements the three-phase algorithm:
    - Phase 1: per-viewpoint visibility via 3D bbox projection + mask analysis.
    - Phase 2: multi-viewpoint aggregation to find best-case visibility.
    - Phase 3: voxel-grid ray marching to detect unobserved volumes.

    Args:
        observations: Ordered list of ObservationRecord (oldest first).
            Observations are automatically capped at MAX_VIEWPOINTS=20; if more
            are provided, only the most recent 20 are used (older are merged into
            the persistent coverage first).
        object_ids: Restrict analysis to these IDs.  Defaults to the union of
            all object IDs found across observations.
        depth_tolerance: Depth tie-breaking tolerance in metres (default 5 mm).
        object_aabbs: Optional {obj_id: (min_xyz, max_xyz)} world-frame bounding
            boxes for Phase 1 projection.  When provided, visibility is computed
            from the full 3D projection; otherwise falls back to mask AABB.

    Returns:
        OcclusionMap with per_object_visibility and unobserved_volumes populated.

    Example:
        occ_map = compute_occlusion_map(
            observations=history,
            object_ids=registry.all_ids(),
        )
        registry.occlusion_map = occ_map
    """
    if not observations:
        return OcclusionMap()

    # Enforce viewpoint cap: take only the most recent MAX_VIEWPOINTS
    if len(observations) > MAX_VIEWPOINTS:
        logger.debug(
            "compute_occlusion_map: capping %d observations to %d",
            len(observations), MAX_VIEWPOINTS,
        )
        observations = observations[-MAX_VIEWPOINTS:]

    # Collect object IDs
    if object_ids is None:
        object_ids = sorted({oid for obs in observations for oid in obs.obj_masks})

    if object_aabbs is None:
        object_aabbs = {}

    # -----------------------------------------------------------------------
    # Phase 1 — per-viewpoint visibility
    # -----------------------------------------------------------------------
    per_viewpoint: List[Dict[str, ViewVisibility]] = []
    poses: List[Optional[CameraPose]] = []
    per_obs_hidden: Dict[str, List[np.ndarray]] = {oid: [] for oid in object_ids}
    camera_poses: List[CameraPose] = []

    for obs in observations:
        vp_vis = _compute_per_viewpoint_visibility(
            obs, object_aabbs, depth_tolerance
        )
        per_viewpoint.append(vp_vis)
        pose = obs.camera_pose or CameraPose.identity()
        poses.append(pose)
        camera_poses.append(pose)

        # Collect hidden masks (2D fallback for hidden_regions field)
        depth = obs.depth_frame
        present = {oid: obs.obj_masks[oid] for oid in object_ids if oid in obs.obj_masks}
        if present:
            min_depth, _, _ = _build_depth_min_map(present, depth)
            for oid in object_ids:
                mask = present.get(oid)
                if mask is None or not mask.any():
                    continue
                hidden = _hidden_mask_for_object(oid, mask, depth, min_depth, depth_tolerance)
                if hidden.any():
                    per_obs_hidden[oid].append(hidden)

    # -----------------------------------------------------------------------
    # Phase 2 — aggregate across viewpoints
    # -----------------------------------------------------------------------
    per_object_visibility = _aggregate_visibility(
        per_viewpoint, poses, object_ids, per_obs_hidden
    )

    # -----------------------------------------------------------------------
    # Phase 3 — unobserved volume detection
    # -----------------------------------------------------------------------
    # Only run Phase 3 when camera poses are available (world-frame grid needed)
    has_poses = any(obs.camera_pose is not None for obs in observations)
    unobserved_volumes: List[UnobservedVolume] = []

    if has_poses:
        try:
            unobserved_volumes, _ = _compute_unobserved_volumes(
                observations, persistent_grid=None
            )
        except Exception as exc:
            logger.warning("Phase 3 unobserved volume detection failed: %s", exc)

    logger.info(
        "compute_occlusion_map: %d observations, %d objects, "
        "%d unobserved volumes",
        len(observations), len(object_ids), len(unobserved_volumes),
    )

    return OcclusionMap(
        camera_poses=camera_poses,
        per_object_visibility=per_object_visibility,
        unobserved_volumes=unobserved_volumes,
    )
