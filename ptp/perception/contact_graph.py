"""
Contact graph computation for a detected object scene.

For each pair of objects, computes whether they are in contact (within a
configurable distance threshold), classifies the contact type from relative
pose geometry, and runs a static-equilibrium analysis to determine the
consequence of removing each object.

Algorithm
---------
Phase 1 — Broad phase (AABB proximity):
  Expand each AABB by contact_threshold_m on all sides, check pairwise
  overlap.  Also check object-surface proximity using point-to-plane
  distance.  For >50 objects a spatial hash (cell_size=0.05 m) is used;
  O(N²) for ≤50 objects.

Phase 2 — Narrow phase (point cloud distance):
  For each candidate pair from Phase 1, compute the actual minimum
  point-cloud distance with a KD-tree (clouds downsampled to ≤2000 points
  with voxel downsampling, voxel_size=0.005 m).  Confirm contact if
  min_dist < threshold.  Find contact region: points within 2×threshold,
  centroid, normal, area.

Phase 3 — Contact classification:
  Classify each confirmed contact into supporting / nested / none
  based on vertical relationship and footprint geometry.

Phase 4 — Support tree:
  Map lower_id → [upper_ids] for "supporting" edges.

Phase 5 — Removal consequences:
  For each object being removed, check directly supported objects; if any
  would tip mark "unstable"; check secondary cascade; mark "cascade" if so.

Phase 6 — Stability scores:
  0.5 * margin_score + 0.5 * redundancy_score for objects with support
  contacts above them; 1.0 otherwise.

Usage:
    graph = compute_contact_graph(
        objects=registry.get_all_objects(),
        obj_masks=masks_dict,          # {object_id: bool ndarray HxW}
        depth_frame=depth_m,
        camera_intrinsics=intr,
        contact_threshold_m=0.008,
    )
    registry.contact_graph = graph
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree, ConvexHull

from .clearance import _depth_to_pointcloud  # reuse existing back-projection helper
from .object_registry import DetectedObject

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

ContactType = Literal["supporting", "nested", "none"]
RemovalConsequence = Literal["stable", "unstable", "cascade"]

# Voxel size used for point cloud downsampling before KD-tree narrow phase.
_VOXEL_SIZE: float = 0.005  # metres
_MAX_PTS: int = 2000        # max points after downsampling


@dataclass
class ContactRegion:
    """Geometric description of a contact region between two objects.

    Attributes:
        point: Contact centroid in camera frame, metres.
        normal: Unit vector pointing from obj_a toward obj_b at contact.
        area: Estimated contact area in m².
    """

    point: np.ndarray        # (3,) camera frame
    normal: np.ndarray       # (3,) unit vector obj_a → obj_b
    area: float              # m²


@dataclass
class ContactEdge:
    """A single contact relationship between two objects.

    Attributes:
        obj_a: ID of the first object (conventionally the lower/supporting one).
        obj_b: ID of the second object.
        contact_type: Geometric classification of the contact.
        contact_region: Point, normal, and area of the contact patch.
        removal_consequence: What happens to obj_b if obj_a is removed.
    """

    obj_a: str
    obj_b: str
    contact_type: ContactType
    contact_region: ContactRegion
    removal_consequence: RemovalConsequence


@dataclass
class ContactGraph:
    """Full scene contact graph.

    Attributes:
        edges: All pairwise contact edges (undirected — obj_a is the lower object).
        support_tree: Maps each object ID to the list of object IDs it supports.
        stability_scores: Per-object [0,1] stability estimate.
        removal_consequences: Per-object consequence of removing that object.
        room_available: LLM-assigned flag per surface object — True when the
            surface appears to have room to receive a displaced object.  Defaults
            to True for surfaces not mentioned by the LLM (conservative: assume
            space exists; the executor grounds the actual placement pose at
            runtime).
    """

    edges: List[ContactEdge] = field(default_factory=list)
    support_tree: Dict[str, List[str]] = field(default_factory=dict)
    stability_scores: Dict[str, float] = field(default_factory=dict)
    removal_consequences: Dict[str, RemovalConsequence] = field(default_factory=dict)
    object_aabbs: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    room_available: Dict[str, bool] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AABB helpers
# ---------------------------------------------------------------------------


def _aabb(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min_xyz, max_xyz) axis-aligned bounding box for a point set."""
    return points.min(axis=0), points.max(axis=0)


def _aabbs_overlap_expanded(
    min_a: np.ndarray, max_a: np.ndarray,
    min_b: np.ndarray, max_b: np.ndarray,
    expansion: float,
) -> bool:
    """Return True if the two AABBs, each expanded by *expansion* on all sides, overlap."""
    for i in range(3):
        if (min_a[i] - expansion) > (max_b[i] + expansion):
            return False
        if (min_b[i] - expansion) > (max_a[i] + expansion):
            return False
    return True


def _aabb_gap(min_a: np.ndarray, max_a: np.ndarray,
              min_b: np.ndarray, max_b: np.ndarray) -> float:
    """Minimum gap between two AABBs (0 = touching, negative = overlapping).

    Returns the Euclidean distance between the closest surface points.
    """
    sep = np.maximum(0.0, np.maximum(min_a - max_b, min_b - max_a))
    return float(np.linalg.norm(sep))


def _footprint_contains(min_outer: np.ndarray, max_outer: np.ndarray,
                        min_inner: np.ndarray, max_inner: np.ndarray) -> bool:
    """Check if inner AABB horizontal footprint fits inside outer's footprint.

    Camera frame: X=right, Y=down, Z=depth.  Horizontal footprint → X and Z axes.
    """
    return (min_inner[0] >= min_outer[0] and max_inner[0] <= max_outer[0] and
            min_inner[2] >= min_outer[2] and max_inner[2] <= max_outer[2])


def _nested(min_inner: np.ndarray, max_inner: np.ndarray,
            min_outer: np.ndarray, max_outer: np.ndarray,
            use_world: bool = False) -> bool:
    """Check if inner AABB is physically nested inside outer AABB.

    Requires containment on horizontal axes AND vertical enclosure from both
    sides (outer bounds inner both above and below on the vertical axis).
    This prevents flat surfaces (tables) from being classified as nesting the
    objects that rest on top of them.

    Camera frame: vertical = Y (down), so inner_min_y >= outer_min_y and
    inner_max_y <= outer_max_y.  World frame: vertical = Z (up), same logic.
    """
    ax = 2 if use_world else 1  # vertical axis index
    horiz = [i for i in range(3) if i != ax]

    # Horizontal containment: inner footprint must be inside outer on both horiz axes
    for i in horiz:
        if min_inner[i] < min_outer[i] or max_inner[i] > max_outer[i]:
            return False

    # Vertical enclosure from BOTH sides: outer must bound inner above AND below.
    # A flat surface only bounds from below — that is supporting, not nested.
    vert_margin = 0.01  # 1 cm — outer must extend at least this far past inner top
    inner_top = max_inner[ax]
    outer_top = max_outer[ax]
    outer_bot = min_outer[ax]
    inner_bot = min_inner[ax]

    if use_world:
        # world Z up: outer_top > inner_top + margin AND outer_bot < inner_bot
        return bool(outer_top > inner_top + vert_margin and outer_bot < inner_bot)
    else:
        # camera Y down: outer_top (min_y) < inner_top - margin AND outer_bot (max_y) > inner_bot
        return bool(outer_bot > inner_bot + vert_margin and outer_top < min_inner[ax])


# ---------------------------------------------------------------------------
# Point cloud helpers
# ---------------------------------------------------------------------------


def _voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample a point cloud to one point per voxel cell (centroid).

    Returns an (M, 3) array with M ≤ len(pts).
    """
    if len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel_size).astype(np.int32)
    # Use a dict to accumulate points per voxel.  For large clouds this is
    # faster than a pandas groupby and avoids the dependency.
    voxel_dict: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, k in enumerate(map(tuple, keys)):
        voxel_dict.setdefault(k, []).append(idx)
    centroids = np.array(
        [pts[idxs].mean(axis=0) for idxs in voxel_dict.values()],
        dtype=np.float32,
    )
    return centroids


def _ensure_max_pts(pts: np.ndarray, max_pts: int) -> np.ndarray:
    """Randomly subsample pts to at most max_pts rows."""
    if len(pts) <= max_pts:
        return pts
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(len(pts), size=max_pts, replace=False)
    return pts[idx]


def _prepare_cloud(pts: np.ndarray) -> np.ndarray:
    """Voxel-downsample then cap at _MAX_PTS for KD-tree queries."""
    down = _voxel_downsample(pts, _VOXEL_SIZE)
    return _ensure_max_pts(down, _MAX_PTS)


# ---------------------------------------------------------------------------
# Spatial hash for broad phase (>50 objects)
# ---------------------------------------------------------------------------


def _spatial_hash_candidates(
    obj_ids: List[str],
    obj_aabbs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    threshold: float,
    cell_size: float = 0.05,
) -> List[Tuple[str, str]]:
    """Return candidate pairs using a spatial hash.

    Assigns each object to all grid cells it touches (after expanding by
    threshold), then yields unique pairs that share at least one cell.
    """
    from collections import defaultdict

    cell_map: Dict[Tuple[int, int, int], List[str]] = defaultdict(list)

    for oid in obj_ids:
        mn, mx = obj_aabbs[oid]
        # Expand by threshold, then find all cells spanned.
        lo = np.floor((mn - threshold) / cell_size).astype(int)
        hi = np.floor((mx + threshold) / cell_size).astype(int)
        for cx in range(lo[0], hi[0] + 1):
            for cy in range(lo[1], hi[1] + 1):
                for cz in range(lo[2], hi[2] + 1):
                    cell_map[(cx, cy, cz)].append(oid)

    seen: set = set()
    candidates: List[Tuple[str, str]] = []
    for cell_objs in cell_map.values():
        for i in range(len(cell_objs)):
            for j in range(i + 1, len(cell_objs)):
                a, b = cell_objs[i], cell_objs[j]
                if a > b:
                    a, b = b, a
                key = (a, b)
                if key not in seen:
                    seen.add(key)
                    candidates.append(key)
    return candidates


# ---------------------------------------------------------------------------
# Phase 1 — Broad phase
# ---------------------------------------------------------------------------


def _broad_phase_candidates(
    obj_ids: List[str],
    obj_aabbs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    surface_ids: List[str],
    threshold: float,
    use_world: bool = False,
) -> List[Tuple[str, str]]:
    """Return (id_a, id_b) pairs that are within threshold of each other.

    For >50 objects uses spatial hash; otherwise O(N²).
    Also checks object-surface proximity using point-to-plane (bottom face).
    """
    n = len(obj_ids)
    candidates: List[Tuple[str, str]] = []

    if n > 50:
        logger.debug("broad_phase: using spatial hash for %d objects", n)
        candidates = _spatial_hash_candidates(obj_ids, obj_aabbs, threshold)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                id_i, id_j = obj_ids[i], obj_ids[j]
                mn_i, mx_i = obj_aabbs[id_i]
                mn_j, mx_j = obj_aabbs[id_j]
                if _aabbs_overlap_expanded(mn_i, mx_i, mn_j, mx_j, threshold):
                    candidates.append((id_i, id_j))

    # Additional surface proximity: for non-surface objects check against each
    # surface's bottom-face plane (point-to-plane distance).
    non_surface_ids = [oid for oid in obj_ids if oid not in set(surface_ids)]
    existing = {(min(a, b), max(a, b)) for a, b in candidates}

    for s_id in surface_ids:
        s_mn, s_mx = obj_aabbs[s_id]
        if use_world:
            # World frame Z-up: surface top face is at max Z.
            surface_top = float(s_mx[2])
        else:
            # Camera frame Y-down: surface top face is at min Y (closest to camera top).
            surface_top = float(s_mn[1])

        for obj_id in non_surface_ids:
            if obj_id == s_id:
                continue
            o_mn, o_mx = obj_aabbs[obj_id]
            if use_world:
                # Object bottom is at min Z; distance to surface top face.
                obj_bottom = float(o_mn[2])
            else:
                # Camera frame Y-down: object bottom is at max Y.
                obj_bottom = float(o_mx[1])
            dist = abs(obj_bottom - surface_top)
            # Use a generous threshold for surface proximity — noisy depth
            # AABBs can have object bottoms sitting several cm above the
            # percentile-clipped surface top.  The narrow phase will confirm.
            if dist <= max(threshold, 0.15):
                key = (min(obj_id, s_id), max(obj_id, s_id))
                if key not in existing:
                    existing.add(key)
                    candidates.append((obj_id, s_id))

    logger.debug("broad_phase: surface_ids=%s", surface_ids)
    logger.debug("broad_phase: %d candidate pairs from %d objects", len(candidates), n)
    for a, b in candidates:
        logger.debug("  candidate: %s — %s", a, b)
    return candidates


# ---------------------------------------------------------------------------
# Phase 2 — Narrow phase (KD-tree)
# ---------------------------------------------------------------------------


def _narrow_phase(
    pts_a: Optional[np.ndarray],
    pts_b: Optional[np.ndarray],
    com_a: np.ndarray,
    com_b: np.ndarray,
    min_a: np.ndarray,
    max_a: np.ndarray,
    threshold: float,
) -> Optional[Tuple[float, np.ndarray, np.ndarray, float]]:
    """Return (min_dist, contact_point, normal, area) or None if not in contact.

    If point clouds are available: use KD-tree to find exact minimum distance
    and derive contact region from points within 2×threshold.
    Falls back to AABB midpoint geometry if no point cloud is available.
    """
    if pts_a is not None and len(pts_a) > 0 and pts_b is not None and len(pts_b) > 0:
        # Downsample before building KD-tree.
        cloud_a = _prepare_cloud(pts_a)
        cloud_b = _prepare_cloud(pts_b)

        tree_b = KDTree(cloud_b)
        dists, _ = tree_b.query(cloud_a, k=1, workers=1)
        min_dist = float(dists.min())

        if min_dist >= threshold:
            return None

        # Contact region: points in cloud_a within 2×threshold of cloud_b.
        contact_mask = dists < 2.0 * threshold
        if contact_mask.any():
            contact_pts = cloud_a[contact_mask]
        else:
            contact_pts = cloud_a[np.argmin(dists):np.argmin(dists) + 1]

        contact_centroid = contact_pts.mean(axis=0)

        # Normal: from center of mass of a toward center of mass of b.
        raw_normal = com_b - com_a
        n_len = float(np.linalg.norm(raw_normal))
        if n_len > 1e-9:
            normal = raw_normal / n_len
        else:
            normal = np.array([0.0, -1.0, 0.0])

        # Area: count of contact points × voxel_size².
        area = float(len(contact_pts)) * (_VOXEL_SIZE ** 2)

        return min_dist, contact_centroid, normal, area

    else:
        # Fallback: AABB gap and midpoint geometry.
        gap = _aabb_gap(min_a, max_a,
                        *_aabb(pts_b) if pts_b is not None and len(pts_b) > 0
                        else (com_b - 0.03, com_b + 0.03))
        if gap >= threshold:
            return None

        contact_pt = (com_a + com_b) / 2.0
        raw_normal = com_b - com_a
        n_len = float(np.linalg.norm(raw_normal))
        normal = raw_normal / n_len if n_len > 1e-9 else np.array([0.0, -1.0, 0.0])
        area = 0.0

        return gap, contact_pt, normal, area


# ---------------------------------------------------------------------------
# Phase 3 — Contact classification
# ---------------------------------------------------------------------------


def _vertical_axis_index(use_world_frame: bool = False) -> int:
    """Return the axis index for 'up': 2 for world Z, 1 for camera Y.

    In camera frame Y is down, so higher objects have smaller Y values.
    In world frame Z is up.
    """
    return 2 if use_world_frame else 1


def _is_higher(com: np.ndarray, ref: np.ndarray, use_world: bool = False) -> bool:
    """Return True if com is higher than ref (in appropriate frame)."""
    if use_world:
        return com[2] > ref[2]
    # Camera frame: smaller Y = higher.
    return com[1] < ref[1]


def _vertical_diff(com_a: np.ndarray, com_b: np.ndarray,
                   use_world: bool = False) -> float:
    """Absolute vertical distance between two centers of mass."""
    ax = 2 if use_world else 1
    return abs(float(com_a[ax] - com_b[ax]))


def _contact_normal_z(normal: np.ndarray, use_world: bool = False) -> float:
    """Absolute component of contact normal along the vertical axis."""
    ax = 2 if use_world else 1
    return abs(float(normal[ax]))


def _contact_normal_horizontal(normal: np.ndarray, use_world: bool = False) -> float:
    """Magnitude of contact normal in the horizontal plane."""
    if use_world:
        return float(np.linalg.norm(normal[[0, 1]]))
    return float(np.linalg.norm(normal[[0, 2]]))


def horizontal_containment(
    upper_min: np.ndarray, upper_max: np.ndarray,
    lower_min: np.ndarray, lower_max: np.ndarray,
    use_world: bool = False,
) -> float:
    """Fraction of upper's horizontal footprint area inside lower's footprint.

    Returns a value in [0, 1].  Uses XZ axes in camera frame, XY in world.
    """
    if use_world:
        axes = [0, 1]
    else:
        axes = [0, 2]

    ax0, ax1 = axes

    upper_area = max(0.0, upper_max[ax0] - upper_min[ax0]) * \
                 max(0.0, upper_max[ax1] - upper_min[ax1])
    if upper_area < 1e-12:
        return 0.0

    inter_lo0 = max(upper_min[ax0], lower_min[ax0])
    inter_hi0 = min(upper_max[ax0], lower_max[ax0])
    inter_lo1 = max(upper_min[ax1], lower_min[ax1])
    inter_hi1 = min(upper_max[ax1], lower_max[ax1])

    inter_area = max(0.0, inter_hi0 - inter_lo0) * max(0.0, inter_hi1 - inter_lo1)
    return float(np.clip(inter_area / upper_area, 0.0, 1.0))


def _com_projects_onto_footprint(
    com: np.ndarray,
    min_surface: np.ndarray, max_surface: np.ndarray,
    use_world: bool = False,
) -> bool:
    """Check if a COM's horizontal projection falls inside a footprint AABB."""
    if use_world:
        return (min_surface[0] <= com[0] <= max_surface[0] and
                min_surface[1] <= com[1] <= max_surface[1])
    # Camera frame: X and Z are horizontal.
    return (min_surface[0] <= com[0] <= max_surface[0] and
            min_surface[2] <= com[2] <= max_surface[2])


def would_tip_without(
    obj_id: str,
    support_id: str,
    edges: List[ContactEdge],
    coms: Dict[str, np.ndarray],
    use_world: bool = False,
) -> bool:
    """Return True if obj_id would tip if support_id were removed.

    Finds remaining support contacts for obj_id excluding support_id,
    builds support polygon from their contact points, and checks whether
    the COM of obj_id projects over that polygon.
    """
    remaining_pts: List[np.ndarray] = []
    for e in edges:
        if e.contact_type != "supporting":
            continue
        if e.obj_a == support_id or e.obj_b == support_id:
            continue
        if e.obj_b == obj_id:
            remaining_pts.append(e.contact_region.point)

    if not remaining_pts:
        return True  # no remaining support → tips

    com = coms.get(obj_id)
    if com is None:
        return False  # can't determine → assume stable

    return not _com_over_support_polygon(com, remaining_pts, use_world=use_world)


def _classify_contact_phase3(
    id_a: str, id_b: str,
    min_a: np.ndarray, max_a: np.ndarray, com_a: np.ndarray,
    min_b: np.ndarray, max_b: np.ndarray, com_b: np.ndarray,
    normal: np.ndarray,
    edges: List[ContactEdge],
    coms: Dict[str, np.ndarray],
    use_world: bool = False,
) -> ContactType:
    """Full Phase-3 classification.

    Convention: id_a is the lower object (higher Y in camera frame / lower Z
    in world frame).  The caller ensures this ordering.

    Returns "supporting" when B rests on A, "nested" when B is enclosed inside
    A, and "none" otherwise.
    """
    # --- Nested check: inner object physically enclosed by outer on all axes ---
    if _nested(min_b, max_b, min_a, max_a, use_world=use_world):
        return "nested"
    if _nested(min_a, max_a, min_b, max_b, use_world=use_world):
        return "nested"

    ax = 2 if use_world else 1

    # Determine if this is a vertical (stacking) relationship by checking
    # whether B's bottom surface is near A's top surface — i.e. B rests on A.
    # This is more reliable than COM-to-COM distance, which conflates object
    # heights with actual vertical separation, causing flat surfaces (tables)
    # to appear laterally adjacent to objects sitting on them.
    #
    # Camera frame: Y is down, so "bottom of B" = max_b[1], "top of A" = min_a[1]
    # World frame:  Z is up,   so "bottom of B" = min_b[2], "top of A" = max_a[2]
    if use_world:
        b_bottom = float(min_b[ax])
        a_top    = float(max_a[ax])
    else:
        b_bottom = float(max_b[ax])   # camera Y: larger = lower
        a_top    = float(min_a[ax])   # camera Y: smaller = higher (top surface)

    # Gap between top of A and bottom of B (positive = B is above A's top face)
    surface_gap = (b_bottom - a_top) if use_world else (a_top - b_bottom)
    height_b = float(max_b[ax] - min_b[ax])
    # "Resting on" if the gap is small relative to B's own height.
    # Threshold: 40% of B's height to absorb depth noise and mask bleed.
    vertical_contact = surface_gap > -0.02 and surface_gap < max(0.04, 0.4 * height_b)

    if vertical_contact:
        # Don't gate on contact normal — KD-tree normals are unreliable for
        # noisy depth point clouds where mask boundaries cause the closest
        # points to be at the sides rather than the top/bottom faces.
        com_projects = _com_projects_onto_footprint(com_b, min_a, max_a,
                                                    use_world=use_world)
        if com_projects:
            return "supporting"
    return "none"


# ---------------------------------------------------------------------------
# Support polygon helpers
# ---------------------------------------------------------------------------


def _com_over_support_polygon(
    com: np.ndarray,
    support_pts: List[np.ndarray],
    use_world: bool = False,
) -> bool:
    """Check if COM's horizontal projection falls within support polygon.

    Handles 0, 1, 2, and ≥3 support points.
    """
    if not support_pts:
        return False

    if use_world:
        def horiz(p: np.ndarray) -> np.ndarray:
            return p[[0, 1]]
    else:
        def horiz(p: np.ndarray) -> np.ndarray:
            return p[[0, 2]]

    com_h = horiz(com)
    pts_h = np.array([horiz(p) for p in support_pts])

    if len(pts_h) == 1:
        return float(np.linalg.norm(com_h - pts_h[0])) < 0.03

    if len(pts_h) == 2:
        p0, p1 = pts_h[0], pts_h[1]
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6:
            return float(np.linalg.norm(com_h - p0)) < 0.03
        t = np.clip(np.dot(com_h - p0, seg) / (seg_len ** 2), 0.0, 1.0)
        closest = p0 + t * seg
        return float(np.linalg.norm(com_h - closest)) < 0.03

    # ≥3 points: ray-casting point-in-polygon test on the convex hull.
    # Sort by angle around centroid for convex polygon approximation.
    centroid = pts_h.mean(axis=0)
    angles = np.arctan2(pts_h[:, 1] - centroid[1], pts_h[:, 0] - centroid[0])
    order = np.argsort(angles)
    poly = pts_h[order]
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > com_h[1]) != (yj > com_h[1]) and
                com_h[0] < (xj - xi) * (com_h[1] - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _support_polygon_pts(
    obj_id: str,
    edges: List[ContactEdge],
    exclude_id: Optional[str] = None,
) -> List[np.ndarray]:
    """Collect contact points that support obj_id (excluding exclude_id).

    Returns contact points from supporting/stacked edges where obj_b == obj_id.
    """
    pts: List[np.ndarray] = []
    for e in edges:
        if exclude_id and (e.obj_a == exclude_id or e.obj_b == exclude_id):
            continue
        if e.contact_type == "supporting" and e.obj_b == obj_id:
            pts.append(e.contact_region.point)
    return pts


# ---------------------------------------------------------------------------
# Phase 5 — Removal consequences
# ---------------------------------------------------------------------------


def _compute_removal_consequences(
    obj_ids: List[str],
    coms: Dict[str, np.ndarray],
    edges: List[ContactEdge],
    use_world: bool = False,
) -> Dict[str, RemovalConsequence]:
    """Compute per-object removal consequences.

    For each object being removed:
    - Find directly supported objects.
    - If any would tip without that support → "unstable".
    - Check secondary cascade → "cascade".
    """
    consequences: Dict[str, RemovalConsequence] = {oid: "stable" for oid in obj_ids}

    for removed_id in obj_ids:
        # Directly supported objects.
        directly_supported = [
            e.obj_b for e in edges
            if e.obj_a == removed_id and e.contact_type == "supporting"
        ]
        if not directly_supported:
            continue

        unstable_set: List[str] = []
        for sup_id in directly_supported:
            remaining = _support_polygon_pts(sup_id, edges, exclude_id=removed_id)
            com = coms.get(sup_id)
            if com is None:
                continue
            if not _com_over_support_polygon(com, remaining, use_world=use_world):
                unstable_set.append(sup_id)

        if not unstable_set:
            continue

        # Check cascade: do any objects supported by the now-unstable objects
        # also become unsupported?
        cascade = False
        for newly_unstable in unstable_set:
            secondary = [
                e.obj_b for e in edges
                if e.obj_a == newly_unstable
                and e.contact_type == "supporting"
                and e.obj_b not in unstable_set
                and e.obj_b != removed_id
            ]
            for sec_id in secondary:
                rem = _support_polygon_pts(sec_id, edges, exclude_id=newly_unstable)
                com = coms.get(sec_id)
                if com is not None and not _com_over_support_polygon(
                        com, rem, use_world=use_world):
                    cascade = True
                    break
            if cascade:
                break

        consequences[removed_id] = "cascade" if cascade else "unstable"

    return consequences


# ---------------------------------------------------------------------------
# Phase 6 — Stability scores
# ---------------------------------------------------------------------------


def _margin_score(
    com: np.ndarray,
    support_pts: List[np.ndarray],
    use_world: bool = False,
) -> float:
    """Normalised COM distance to edge of support polygon.

    Computed as the ratio of the COM's distance to the nearest edge of the
    support polygon divided by the inscribed circle radius.  Clipped to [0,1].
    """
    if not support_pts:
        return 0.0

    if use_world:
        def horiz(p: np.ndarray) -> np.ndarray:
            return p[[0, 1]]
    else:
        def horiz(p: np.ndarray) -> np.ndarray:
            return p[[0, 2]]

    com_h = horiz(com)
    pts_h = np.array([horiz(p) for p in support_pts])

    if len(pts_h) < 3:
        # For 1-2 points use distance-to-nearest as a proxy.
        min_d = float(np.min(np.linalg.norm(pts_h - com_h, axis=1)))
        return float(np.clip(min_d / 0.05, 0.0, 1.0))

    try:
        hull = ConvexHull(pts_h)
    except Exception:
        return 0.0

    # Inscribed circle radius ≈ area / semi-perimeter (inradius of convex polygon).
    area = hull.volume  # In 2D ConvexHull, .volume is the area.
    perimeter = hull.area   # In 2D ConvexHull, .area is the perimeter.
    if perimeter < 1e-9:
        return 0.0
    inradius = area / (perimeter / 2.0)

    # Distance from COM to nearest hull edge.
    equations = hull.equations  # (num_facets, 3) — [normal_x, normal_y, offset]
    signed_dists = pts_h[hull.vertices].dot(equations[:, :2].T) + equations[:, 2]
    # Distance from COM to each edge.
    com_dists = equations[:, :2].dot(com_h) + equations[:, 2]
    # Negative sign convention: positive means inside.
    min_dist = float(np.min(-com_dists))  # positive = inside hull

    if inradius < 1e-9:
        return 0.0
    return float(np.clip(min_dist / inradius, 0.0, 1.0))


def _redundancy_score(
    obj_id: str,
    edges: List[ContactEdge],
    coms: Dict[str, np.ndarray],
    use_world: bool = False,
) -> float:
    """Fraction of support contacts removable while keeping obj_id stable."""
    support_edges = [
        e for e in edges
        if e.contact_type == "supporting" and e.obj_b == obj_id
    ]
    if not support_edges:
        return 1.0
    com = coms.get(obj_id)
    if com is None:
        return 1.0

    removable = 0
    for se in support_edges:
        remaining = _support_polygon_pts(obj_id, edges, exclude_id=se.obj_a)
        if _com_over_support_polygon(com, remaining, use_world=use_world):
            removable += 1
    return removable / len(support_edges)


def _compute_stability_scores(
    obj_ids: List[str],
    coms: Dict[str, np.ndarray],
    edges: List[ContactEdge],
    use_world: bool = False,
) -> Dict[str, float]:
    """Compute per-object stability scores in [0, 1].

    Objects with no support contacts above them (i.e., they are not being
    supported) receive score 1.0.  Others receive:
        0.5 * margin_score + 0.5 * redundancy_score
    """
    scores: Dict[str, float] = {}

    for obj_id in obj_ids:
        support_pts = _support_polygon_pts(obj_id, edges, exclude_id=None)
        if not support_pts:
            scores[obj_id] = 1.0
            continue
        com = coms.get(obj_id)
        if com is None:
            scores[obj_id] = 1.0
            continue
        m = _margin_score(com, support_pts, use_world=use_world)
        r = _redundancy_score(obj_id, edges, coms, use_world=use_world)
        scores[obj_id] = float(0.5 * m + 0.5 * r)

    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_contact_graph(
    objects: List[DetectedObject],
    obj_masks: Dict[str, np.ndarray],
    depth_frame: np.ndarray,
    camera_intrinsics,
    contact_threshold_m: float = 0.008,
    camera_position: Optional[np.ndarray] = None,
    camera_rotation=None,
    color_image: Optional[np.ndarray] = None,
    llm_client=None,
    llm_mode: str = "nl",
    llm_debug_image_out: Optional[List] = None,
) -> ContactGraph:
    """Compute the pairwise contact graph for a set of detected objects.

    Args:
        objects: All DetectedObject instances in the current frame.
        obj_masks: {object_id: bool ndarray (H, W)} from SAM2 segmentation.
        depth_frame: (H, W) float32 depth in metres.
        camera_intrinsics: Object with fx, fy, ppx/cx, ppy/cy attributes.
        contact_threshold_m: Max distance to consider objects in contact, metres.
            Default 0.008 m (8 mm).
        camera_position: (3,) camera origin in robot base/world frame.  When
            provided together with camera_rotation, all point clouds are
            transformed to world frame (Z-up) before classification, giving
            more reliable stacked/supporting results than camera-frame geometry.
        camera_rotation: scipy.spatial.transform.Rotation for camera-to-world.
        color_image: RGB uint8 (H, W, 3) scene image.  Required when llm_client
            is provided — used to render the annotated image sent to the model.
        llm_client: Optional LLMClient instance.  When provided, Phase-3
            geometric classifications are replaced with LLM-inferred labels.
            Requires camera_position, camera_rotation, and color_image.

    Returns:
        ContactGraph with edges, support_tree, stability_scores, and
        removal_consequences populated.
    """
    if len(objects) < 2:
        return ContactGraph(
            support_tree={o.object_id: [] for o in objects},
            stability_scores={o.object_id: 1.0 for o in objects},
            removal_consequences={o.object_id: "stable" for o in objects},
        )

    h, w = depth_frame.shape
    fx = float(getattr(camera_intrinsics, 'fx', w / 2))
    fy = float(getattr(camera_intrinsics, 'fy', h / 2))
    cx = float(getattr(camera_intrinsics, 'ppx', getattr(camera_intrinsics, 'cx', w / 2)))
    cy = float(getattr(camera_intrinsics, 'ppy', getattr(camera_intrinsics, 'cy', h / 2)))

    use_world = camera_position is not None and camera_rotation is not None

    def _to_world(pts: np.ndarray) -> np.ndarray:
        if not use_world or len(pts) == 0:
            return pts
        return camera_rotation.apply(pts) + camera_position

    # -----------------------------------------------------------------------
    # Build per-object AABBs, centroids, and point clouds
    # -----------------------------------------------------------------------
    obj_aabbs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    obj_coms: Dict[str, np.ndarray] = {}
    obj_clouds: Dict[str, Optional[np.ndarray]] = {}

    _SURFACE_KEYWORDS = ("surface", "table", "shelf", "tray", "counter", "desk", "floor")

    def _is_surface(obj: DetectedObject) -> bool:
        t = obj.object_type.lower()
        return any(kw in t for kw in _SURFACE_KEYWORDS)

    surface_ids: List[str] = [o.object_id for o in objects if _is_surface(o)]

    for obj in objects:
        mask = obj_masks.get(obj.object_id)
        if mask is not None and mask.any():
            masked_depth = np.where(mask, depth_frame, 0.0)
            pts = _to_world(_depth_to_pointcloud(masked_depth, fx, fy, cx, cy))
        else:
            pts = np.empty((0, 3), dtype=np.float32)

        if len(pts) > 0:
            # Use 5th–95th percentile extents to reject outlier depth pixels
            # at mask boundaries that would otherwise inflate the AABB.
            mn = np.percentile(pts, 5,  axis=0)
            mx = np.percentile(pts, 95, axis=0)
            obj_aabbs[obj.object_id] = (mn, mx)
            obj_coms[obj.object_id] = np.median(pts, axis=0)
            obj_clouds[obj.object_id] = pts
        elif obj.position_3d is not None:
            # Use position_3d which is already in world frame if robot_interface was set.
            c = np.asarray(obj.position_3d, dtype=float)
            obj_aabbs[obj.object_id] = (c - 0.03, c + 0.03)
            obj_coms[obj.object_id] = c
            obj_clouds[obj.object_id] = None
        else:
            logger.debug("compute_contact_graph: skipping %s — no depth data",
                         obj.object_id)
            continue

    obj_ids = list(obj_aabbs.keys())
    n = len(obj_ids)
    logger.debug("compute_contact_graph: %d objects with AABB data", n)

    if n < 2:
        return ContactGraph(
            support_tree={oid: [] for oid in obj_ids},
            stability_scores={oid: 1.0 for oid in obj_ids},
            removal_consequences={oid: "stable" for oid in obj_ids},
            room_available={oid: True for oid in surface_ids},
        )

    # -----------------------------------------------------------------------
    # Phase 1 — Broad phase
    # -----------------------------------------------------------------------
    candidates = _broad_phase_candidates(
        obj_ids, obj_aabbs, surface_ids, contact_threshold_m, use_world=use_world
    )

    # -----------------------------------------------------------------------
    # Phase 2 — Narrow phase + contact region
    # Phase 3 — Classification (partial — edges list built without would_tip)
    # -----------------------------------------------------------------------
    # We build edges in two passes:
    # Pass A: compute contact region and determine order (lower/upper).
    # Pass B: Phase-3 classification using the already-collected edges list
    #         (so would_tip_without can inspect it).

    raw_edges: List[Tuple[str, str, float, np.ndarray, np.ndarray, float]] = []
    # raw_edges entries: (id_a, id_b, min_dist, contact_pt, normal, area)
    # id_a is always the lower object.

    for id_i, id_j in candidates:
        mn_i, mx_i = obj_aabbs[id_i]
        mn_j, mx_j = obj_aabbs[id_j]
        com_i = obj_coms[id_i]
        com_j = obj_coms[id_j]
        pts_i = obj_clouds.get(id_i)
        pts_j = obj_clouds.get(id_j)

        result = _narrow_phase(
            pts_i, pts_j,
            com_i, com_j,
            mn_i, mx_i,
            contact_threshold_m,
        )
        if result is None:
            continue

        min_dist, contact_pt, normal, area = result

        # Order so that id_a is the lower object.
        # Camera frame: lower = larger Y. World frame: lower = smaller Z.
        if use_world:
            i_is_lower = com_i[2] <= com_j[2]
        else:
            i_is_lower = com_i[1] >= com_j[1]
        if i_is_lower:
            a_id, b_id = id_i, id_j
            min_a, max_a, com_a = mn_i, mx_i, com_i
            min_b, max_b, com_b = mn_j, mx_j, com_j
        else:
            a_id, b_id = id_j, id_i
            min_a, max_a, com_a = mn_j, mx_j, com_j
            min_b, max_b, com_b = mn_i, mx_i, com_i
            # Flip normal so it still points a → b.
            normal = -normal

        raw_edges.append((a_id, b_id, min_dist, contact_pt, normal, area))

    # Pass A: create placeholder edges (contact_type="adjacent") so that the
    # would_tip_without calls in Phase 3 have something to inspect.
    edges: List[ContactEdge] = []
    for a_id, b_id, _, contact_pt, normal, area in raw_edges:
        edges.append(ContactEdge(
            obj_a=a_id,
            obj_b=b_id,
            contact_type="none",  # placeholder
            contact_region=ContactRegion(point=contact_pt, normal=normal, area=area),
            removal_consequence="stable",
        ))

    # Pass B: Phase-3 full classification.
    # Optionally replaced by LLM classifications when llm_client is provided.
    from .llm_contact_classifier import LLMContactResult
    llm_result: Optional[LLMContactResult] = None
    llm_labels: Dict[Tuple[str, str], tuple] = {}
    if llm_client is not None and color_image is not None and use_world:
        try:
            from .llm_contact_classifier import (
                classify_contacts_with_llm,
                classify_contacts_nl,
            )
            _classify_fn = classify_contacts_nl if llm_mode == "nl" else classify_contacts_with_llm
            llm_result = _classify_fn(
                objects=objects,
                object_aabbs=obj_aabbs,
                color_img=color_image,
                obj_masks=obj_masks,
                llm_client=llm_client,
                surface_ids=surface_ids,
                debug_image_out=llm_debug_image_out,
            )
            llm_labels = llm_result.labels
            logger.info("LLM classification returned %d pair labels, %d room assessments",
                        len(llm_labels), len(llm_result.room_available))
        except Exception as exc:
            logger.error("LLM contact classification error — falling back to geometric: %s", exc)

    for idx, (a_id, b_id, _, contact_pt, normal, area) in enumerate(raw_edges):
        # Use LLM label if available (key is sorted pair)
        llm_key = (min(a_id, b_id), max(a_id, b_id))
        if llm_key in llm_labels:
            contact_type, llm_a, llm_b = llm_labels[llm_key]
            # If LLM reversed the direction (llm_a is b_id), swap edge endpoints
            # so obj_a is always the supporter/container.
            if contact_type == "supporting" and llm_a == b_id and llm_b == a_id:
                edges[idx].obj_a, edges[idx].obj_b = b_id, a_id
        else:
            min_a, max_a = obj_aabbs[a_id]
            min_b, max_b = obj_aabbs[b_id]
            com_a = obj_coms[a_id]
            com_b = obj_coms[b_id]

            contact_type = _classify_contact_phase3(
                a_id, b_id,
                min_a, max_a, com_a,
                min_b, max_b, com_b,
                normal,
                edges,
                obj_coms,
                use_world=use_world,
            )
        edges[idx].contact_type = contact_type

    # -----------------------------------------------------------------------
    # Phase 4 — Support tree
    # -----------------------------------------------------------------------
    support_tree: Dict[str, List[str]] = {oid: [] for oid in obj_ids}
    for e in edges:
        if e.contact_type == "supporting":
            if e.obj_a in support_tree:
                support_tree[e.obj_a].append(e.obj_b)
            else:
                support_tree[e.obj_a] = [e.obj_b]

    # -----------------------------------------------------------------------
    # Phase 5 — Removal consequences
    # -----------------------------------------------------------------------
    removal_consequences = _compute_removal_consequences(
        obj_ids, obj_coms, edges, use_world=use_world
    )

    # Back-fill removal_consequence on edges (consequence of removing obj_a).
    for e in edges:
        e.removal_consequence = removal_consequences.get(e.obj_a, "stable")

    # -----------------------------------------------------------------------
    # Phase 6 — Stability scores
    # -----------------------------------------------------------------------
    stability_scores = _compute_stability_scores(
        obj_ids, obj_coms, edges, use_world=use_world
    )

    logger.info(
        "compute_contact_graph: %d objects, %d contact edges "
        "(%d supporting, %d nested, %d none)%s",
        n,
        len(edges),
        sum(1 for e in edges if e.contact_type == "supporting"),
        sum(1 for e in edges if e.contact_type == "nested"),
        sum(1 for e in edges if e.contact_type == "none"),
        " [LLM]" if llm_labels else "",
    )

    # room_available — from LLM when available, else True for all surfaces.
    room_available: Dict[str, bool] = {}
    if llm_result is not None:
        room_available = dict(llm_result.room_available)
    if not room_available:
        room_available = {oid: True for oid in surface_ids}

    return ContactGraph(
        edges=edges,
        support_tree=support_tree,
        stability_scores=stability_scores,
        removal_consequences=removal_consequences,
        object_aabbs=dict(obj_aabbs),
        room_available=room_available,
    )
