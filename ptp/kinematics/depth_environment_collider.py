"""
Depth-image-based environment collision for PyBullet motion planning.

Converts a RealSense depth frame into per-object triangle-mesh collision bodies
plus a background body for the unmasked environment, so the planner can
selectively enable or disable individual objects during collision checking
(e.g. ignore the target object while planning a grasp).

The camera extrinsic is read from the XArmPybulletInterface FK at update time
so it automatically tracks the wrist-mounted camera as the arm moves.

Typical usage::

    collider = DepthEnvironmentCollider(planner)

    # Build one body per detected object + a background body
    collider.update(camera, masks)          # masks: {object_id: bool H×W ndarray}

    # Check trajectory against everything except the grasp target
    hit = collider.check_trajectory(trajectory, ignore={"red_cup_1"})

    # Or check only background (ignore all objects)
    hit = collider.check_trajectory(trajectory, ignore=set(collider.object_ids))
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional, Set

import numpy as np

try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False

try:
    import pybullet as p
    _PYBULLET_AVAILABLE = True
except ImportError:
    _PYBULLET_AVAILABLE = False

from ptp.camera.base_camera import BaseCamera, CameraIntrinsics

if TYPE_CHECKING:
    from ptp.kinematics.base_pybullet_interface import BasePybulletInterface

logger = logging.getLogger(__name__)

_DEPTH_MIN_M = 0.15
_DEPTH_MAX_M = 2.0
_DEPTH_STRIDE = 2              # reduced from 4 — 4× more points, better coverage of small/far objects
_VOXEL_SIZE_M = 0.003          # downsample to 3 mm grid before BPA to normalise point spacing
_BPA_RADIUS_FACTOR_OBJ = 1.5   # tighter ball for per-object meshes — prevents gap-spanning fins
_BPA_RADIUS_FACTOR_BG  = 2.0   # looser ball for background — larger surfaces tolerate more gap
_BPA_RADIUS_MAX_M = 0.04       # hard cap: ball never spans more than 4 cm
_LONG_EDGE_FACTOR = 4.0        # triangles whose longest edge > factor × mean_spacing are artifacts
_MIN_COMPONENT_TRIS_RATIO = 0.02  # min component size as fraction of total triangles (floor: 8)
_MIN_COMPONENT_TRIS_FLOOR = 8  # absolute minimum — avoids pruning everything on small objects
_OUTLIER_NB_POINTS = 16        # statistical outlier removal neighbour count
_OUTLIER_STD_RATIO = 1.5       # points beyond mean + ratio×std are removed
_COLLISION_MARGIN_M = 0.005
_WORLD_Z_MIN_M = -0.10          # discard points below this (noise/mis-projection below table)
_WORLD_Z_MAX_M = 1.0            # discard points above the workspace ceiling
# Pixels within this many pixels of any object mask edge are excluded from the
# background mesh.  Depth sensors bleed values behind objects at silhouette
# edges; eroding the exclusion zone removes those leaked pixels before BPA.
_MASK_ERODE_PX = 6
# Background points whose depth exceeds the minimum object depth at the same
# pixel (scaled by this factor) are occluded and removed.  1.0 = exact, values
# slightly above 1.0 give a small tolerance for sensor noise.
_OCCLUSION_DEPTH_FACTOR = 1.05

# Label used for the body covering all non-object depth pixels
BACKGROUND_ID = "__background__"


class DepthEnvironmentCollider:
    """Per-object + background mesh collision bodies built from a live depth image.

    Each detected object gets its own PyBullet body (masked depth pixels).
    All remaining depth pixels form a single background body.  This lets the
    planner ignore specific objects during collision checking.

    Args:
        planner: XArmPybulletInterface (or any BasePybulletInterface subclass).
        collision_margin_m: Distance (metres) below which a robot link is
            considered in collision with a mesh body.
    """

    def __init__(
        self,
        planner: BasePybulletInterface,
        collision_margin_m: float = _COLLISION_MARGIN_M,
    ) -> None:
        if not _O3D_AVAILABLE:
            raise ImportError("open3d is required: pip install open3d")
        if not _PYBULLET_AVAILABLE:
            raise ImportError("pybullet is required: pip install pybullet")

        self._planner = planner
        self._margin = float(collision_margin_m)
        # object_id (or BACKGROUND_ID) -> pybullet body ID
        self._bodies: Dict[str, int] = {}
        self._cam_pos_world: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def object_ids(self) -> list[str]:
        """Object IDs for which individual bodies exist (excludes background)."""
        return [k for k in self._bodies if k != BACKGROUND_ID]

    @property
    def background_body_id(self) -> Optional[int]:
        """PyBullet body ID of the background mesh, or None."""
        return self._bodies.get(BACKGROUND_ID)

    def body_id_for(self, object_id: str) -> Optional[int]:
        """Return the PyBullet body ID for a specific object, or None."""
        return self._bodies.get(object_id)

    def update(
        self,
        camera: BaseCamera,
        masks: Dict[str, np.ndarray],
    ) -> Dict[str, bool]:
        """Capture a depth frame and rebuild all collision bodies.

        Args:
            camera: RealSenseCamera (or any BaseCamera with depth support).
            masks: {object_id: bool H×W mask} from GSAM2 / _last_masks.

        Returns:
            Dict mapping each body label (object_id + BACKGROUND_ID) to
            True if its mesh was successfully built.
        """
        T_base_cam = self._get_camera_extrinsic()
        if T_base_cam is None:
            logger.warning("DepthEnvironmentCollider: cannot read camera extrinsic")
            return {}
        intrinsics = camera.get_camera_intrinsics()
        _, depth = camera.get_aligned_frames()
        return self._rebuild(depth, intrinsics, T_base_cam, masks)

    def update_from_depth(
        self,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        masks: Dict[str, np.ndarray],
        debug_dir: Optional[str] = None,
    ) -> Dict[str, bool]:
        """Rebuild from an already-captured depth array.

        Args:
            depth_m: (H, W) float32 depth image in metres.
            intrinsics: Camera intrinsic parameters.
            masks: {object_id: bool H×W mask}.
            debug_dir: If given, write ``depth_debug.png`` into this directory.

        Returns:
            Dict mapping each label to True if its mesh was built successfully.
        """
        T_base_cam = self._get_camera_extrinsic()
        if T_base_cam is None:
            logger.warning("DepthEnvironmentCollider: cannot read camera extrinsic")
            return {}
        return self._rebuild(depth_m, intrinsics, T_base_cam, masks, debug_dir=debug_dir)

    def check_trajectory(
        self,
        trajectory: np.ndarray,
        ignore: Optional[Set[str]] = None,
    ) -> Optional[int]:
        """Check each waypoint of a planned trajectory against the depth-mesh bodies.

        Floor and self-collision are handled by the OMPL planner's validity
        checker during path generation.  This method is used post-planning
        to verify the trajectory against depth-mesh obstacles (which may not
        have been present when the trajectory was planned, e.g. in the grasp
        sampler when ignoring the target object).

        Args:
            trajectory: (N, dof) array of joint configurations in radians.
            ignore: Set of object_id labels (or BACKGROUND_ID) to skip.

        Returns:
            Index of the first colliding waypoint, or None if collision-free.
        """
        active_bodies = {
            label: bid for label, bid in self._bodies.items()
            if (ignore is None or label not in ignore)
        }
        if not active_bodies:
            return None

        client = self._planner._physics_client
        rid = self._planner._robot_id
        movable = self._planner._movable_joints
        n_links = p.getNumJoints(rid, physicsClientId=client)
        saved_joints = self._planner._joints.copy()
        hit: Optional[int] = None

        try:
            for step_idx, joints in enumerate(trajectory):
                for j, joint_idx in enumerate(movable):
                    if j < len(joints):
                        p.resetJointState(rid, joint_idx, float(joints[j]),
                                          physicsClientId=client)

                for label, body_id in active_bodies.items():
                    for link_idx in range(-1, n_links):
                        contacts = p.getClosestPoints(
                            bodyA=rid,
                            bodyB=body_id,
                            distance=self._margin,
                            linkIndexA=link_idx,
                            physicsClientId=client,
                        )
                        if contacts:
                            logger.debug(
                                "Mesh collision at waypoint %d body=%s link=%d dist=%.4f",
                                step_idx, label, link_idx, contacts[0][8],
                            )
                            hit = step_idx
                            break
                    if hit is not None:
                        break

                if hit is not None:
                    break
        finally:
            for j, joint_idx in enumerate(movable):
                if j < len(saved_joints):
                    p.resetJointState(rid, joint_idx, float(saved_joints[j]),
                                      physicsClientId=client)

        return hit

    def remove(self, label: Optional[str] = None) -> None:
        """Remove one or all mesh bodies from PyBullet.

        Args:
            label: If given, remove only that body (object_id or BACKGROUND_ID).
                   If None, remove all bodies.
        """
        client = self._planner._physics_client
        to_remove = [label] if label is not None else list(self._bodies.keys())
        for lbl in to_remove:
            body_id = self._bodies.pop(lbl, None)
            if body_id is not None:
                try:
                    p.removeBody(body_id, physicsClientId=client)
                except Exception:
                    pass

    def __del__(self) -> None:
        try:
            self.remove()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Multi-frame accumulation API
    # ------------------------------------------------------------------

    def start_accumulation(self) -> None:
        """Begin a multi-frame point-cloud accumulation pass.

        Call once before the scan loop, then call accumulate_from_depth() for
        each viewpoint, then finalize_accumulated() to build PyBullet bodies.
        """
        self._accumulated: Dict[str, list] = {}

    def point_counts(self) -> Dict[str, int]:
        """Return the current voxel-downsampled point count per label.

        Used by scan_workspace to decide whether each label has reached the
        target density threshold.
        """
        if not hasattr(self, "_accumulated"):
            return {}
        counts: Dict[str, int] = {}
        for label, chunks in self._accumulated.items():
            if chunks:
                counts[label] = int(sum(len(c) for c in chunks))
            else:
                counts[label] = 0
        return counts

    def accumulate_from_depth(
        self,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        masks: Dict[str, np.ndarray],
        T_base_cam: np.ndarray,
        voxel_size: float = 0.005,
    ) -> None:
        """Add one depth frame to the accumulated point clouds.

        Args:
            depth_m: (H, W) float32 depth image in metres.
            intrinsics: Camera intrinsic parameters.
            masks: {object_id: bool H×W mask} from GSAM2.
            T_base_cam: 4×4 camera→base transform for this viewpoint.
            voxel_size: Voxel downsampling resolution (metres) applied after
                each frame to keep memory bounded.
        """
        if not hasattr(self, "_accumulated"):
            self.start_accumulation()

        h, w = depth_m.shape

        # Build union mask for background exclusion
        union_mask = np.zeros((h, w), dtype=bool)
        for mask in masks.values():
            if mask.shape == (h, w):
                union_mask |= mask.astype(bool)

        # Accumulate per-object clouds
        for obj_id, mask in masks.items():
            if mask.shape != (h, w):
                continue
            pts, _, _ = self._depth_to_world_points(depth_m, intrinsics, T_base_cam,
                                                     pixel_mask=mask.astype(bool))
            if pts is not None and len(pts) > 0:
                if obj_id not in self._accumulated:
                    self._accumulated[obj_id] = []
                self._accumulated[obj_id].append(pts)

        # Accumulate background
        try:
            from scipy.ndimage import binary_dilation, minimum_filter
            erode_struct = np.ones((2 * _MASK_ERODE_PX + 1, 2 * _MASK_ERODE_PX + 1), dtype=bool)
            exclusion_mask = binary_dilation(union_mask, structure=erode_struct)
            obj_depth = np.where(union_mask, depth_m, np.inf).astype(np.float32)
            min_obj_depth = minimum_filter(obj_depth, size=2 * _MASK_ERODE_PX + 1)
            background_mask = ~exclusion_mask
            occluded = (
                background_mask
                & np.isfinite(min_obj_depth)
                & (depth_m > _OCCLUSION_DEPTH_FACTOR * min_obj_depth)
            )
            background_mask &= ~occluded
        except ImportError:
            background_mask = ~union_mask

        pts_bg, _, _ = self._depth_to_world_points(depth_m, intrinsics, T_base_cam,
                                                    pixel_mask=background_mask)
        if pts_bg is not None and len(pts_bg) > 0:
            if BACKGROUND_ID not in self._accumulated:
                self._accumulated[BACKGROUND_ID] = []
            self._accumulated[BACKGROUND_ID].append(pts_bg)

        # Voxel-downsample each label's running cloud to keep memory bounded
        if voxel_size > 0:
            for label in list(self._accumulated.keys()):
                chunks = self._accumulated[label]
                if len(chunks) > 1:
                    merged = np.vstack(chunks)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(merged)
                    pcd = pcd.voxel_down_sample(voxel_size)
                    self._accumulated[label] = [np.asarray(pcd.points, dtype=np.float32)]

    def finalize_accumulated(self) -> Dict[str, bool]:
        """Build PyBullet bodies from all accumulated point clouds.

        Removes any previously built bodies first, then builds one body per
        label from the merged multi-frame cloud.

        Returns:
            Dict mapping each label to True if its mesh was built successfully.
        """
        if not hasattr(self, "_accumulated") or not self._accumulated:
            logger.warning("DepthEnvironmentCollider: finalize called with no accumulated data")
            return {}

        self.remove()
        results: Dict[str, bool] = {}
        for label, chunks in self._accumulated.items():
            if not chunks:
                results[label] = False
                continue
            pts = np.vstack(chunks).astype(np.float32)
            results[label] = self._build_body(pts, label)

        built = [lbl for lbl, ok in results.items() if ok]
        logger.info(
            "DepthEnvironmentCollider: finalized %d bodies from accumulated scan: %s",
            len(built), ", ".join(built),
        )
        del self._accumulated
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_camera_extrinsic(self) -> Optional[np.ndarray]:
        cam_pos, cam_rot = self._planner.get_camera_transform()
        if cam_pos is None or cam_rot is None:
            return None
        T = np.eye(4)
        T[:3, :3] = cam_rot.as_matrix()
        T[:3, 3] = cam_pos
        return T

    def _rebuild(
        self,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        T_base_cam: np.ndarray,
        masks: Dict[str, np.ndarray],
        debug_dir: Optional[str] = None,
    ) -> Dict[str, bool]:
        self.remove()
        self._cam_pos_world = T_base_cam[:3, 3].copy()

        h, w = depth_m.shape
        results: Dict[str, bool] = {}

        # Union of all object masks — used both for background exclusion and
        # per-pixel minimum-depth computation for occlusion culling.
        union_mask = np.zeros((h, w), dtype=bool)
        for mask in masks.values():
            if mask.shape == (h, w):
                union_mask |= mask.astype(bool)

        # Build one body per detected object.
        for obj_id, mask in masks.items():
            if mask.shape != (h, w):
                logger.warning("DepthEnvironmentCollider: mask shape mismatch for %s", obj_id)
                results[obj_id] = False
                continue
            pts, _, _ = self._depth_to_world_points(depth_m, intrinsics, T_base_cam,
                                                     pixel_mask=mask.astype(bool))
            results[obj_id] = self._build_body(pts, obj_id)

        # Build background body with two levels of occlusion filtering so the
        # mesh doesn't extend behind objects that are visible to the camera.
        #
        # 1. Erode the union mask before inverting: pixels within _MASK_ERODE_PX
        #    of any object silhouette are excluded.  Depth sensors produce
        #    bleed/interpolation artefacts at object edges; this removes them.
        #
        # 2. Ray-occlusion cull: for each background pixel, if its depth is
        #    greater than _OCCLUSION_DEPTH_FACTOR × the minimum object depth at
        #    that same pixel (or nearby), the point is behind an object and is
        #    removed.  This catches pixels that are geometrically occluded but
        #    not covered by the segmentation mask (e.g. shadow/bleed regions).
        try:
            from scipy.ndimage import binary_dilation, minimum_filter
            erode_struct = np.ones(
                (2 * _MASK_ERODE_PX + 1, 2 * _MASK_ERODE_PX + 1), dtype=bool
            )
            # Dilate the union mask so a wider border is excluded from background.
            exclusion_mask = binary_dilation(union_mask, structure=erode_struct)

            # Minimum object depth per pixel over a local neighbourhood — gives
            # a conservative foreground depth reference for occlusion testing.
            obj_depth = np.where(union_mask, depth_m, np.inf).astype(np.float32)
            min_obj_depth = minimum_filter(obj_depth, size=2 * _MASK_ERODE_PX + 1)

            background_mask = ~exclusion_mask
            # Occluded pixel: background depth > factor × nearest object depth.
            occluded = (
                background_mask
                & np.isfinite(min_obj_depth)
                & (depth_m > _OCCLUSION_DEPTH_FACTOR * min_obj_depth)
            )
            background_mask &= ~occluded
            n_culled = int(occluded.sum())
            if n_culled:
                logger.debug(
                    "DepthEnvironmentCollider: background occlusion cull removed %d px", n_culled
                )
        except ImportError:
            # scipy unavailable — fall back to simple mask inversion.
            logger.warning(
                "DepthEnvironmentCollider: scipy not available, skipping occlusion cull"
            )
            background_mask = ~union_mask

        pts_bg, _, _ = self._depth_to_world_points(depth_m, intrinsics, T_base_cam,
                                                    pixel_mask=background_mask)
        results[BACKGROUND_ID] = self._build_body(pts_bg, BACKGROUND_ID)

        built = [lbl for lbl, ok in results.items() if ok]
        logger.info(
            "DepthEnvironmentCollider: built %d bodies (%s)",
            len(built), ", ".join(built),
        )
        if debug_dir is not None:
            import os
            out_path = os.path.join(debug_dir, "depth_debug.png")
            self._save_depth_debug(depth_m, masks, out_path=out_path)
        return results

    def _save_depth_debug(
        self,
        depth_m: np.ndarray,
        masks: Optional[Dict[str, np.ndarray]] = None,
        out_path: str = "/tmp/collider_depth_debug.png",
    ) -> None:
        """Save depth debug images: plain colourmap and one with mask overlays.

        Writes two files:
          - ``out_path``              — raw depth colourmap (no masks)
          - ``<stem>_masks.<ext>``    — same colourmap with per-object mask tints
        """
        try:
            import cv2
        except ImportError:
            logger.debug("cv2 not available — skipping depth debug image")
            return

        # Auto-scale depth to the p2/p98 range of valid pixels so scene
        # contrast is maximised regardless of the fixed _DEPTH_MIN/MAX constants.
        valid = depth_m[(depth_m > _DEPTH_MIN_M) & (depth_m < _DEPTH_MAX_M)]
        if len(valid) > 0:
            d_lo = float(np.percentile(valid, 2))
            d_hi = float(np.percentile(valid, 98))
            if d_hi <= d_lo:
                d_lo, d_hi = _DEPTH_MIN_M, _DEPTH_MAX_M
        else:
            d_lo, d_hi = _DEPTH_MIN_M, _DEPTH_MAX_M
        d_clipped = np.clip(depth_m, d_lo, d_hi)
        d_scaled = ((d_clipped - d_lo) / (d_hi - d_lo) * 255).astype(np.uint8)
        vis = cv2.applyColorMap(d_scaled, cv2.COLORMAP_MAGMA)

        cv2.imwrite(out_path, vis)
        logger.info("DepthEnvironmentCollider: depth debug saved → %s", out_path)

        if not masks:
            return

        import os
        stem, ext = os.path.splitext(out_path)
        masks_path = f"{stem}_masks{ext}"

        # Distinct BGR tint colours for each mask.
        _TINT_COLOURS = [
            (80,  200,  80),
            (80,   80, 220),
            (220, 160,  40),
            (40,  220, 220),
            (200,  80, 200),
            (80,  200, 200),
        ]
        overlay = vis.copy()
        h, w = depth_m.shape
        for idx, (obj_id, mask) in enumerate(masks.items()):
            if mask.shape != (h, w):
                continue
            colour = _TINT_COLOURS[idx % len(_TINT_COLOURS)]
            tint = np.zeros_like(overlay)
            tint[mask.astype(bool)] = colour
            overlay = cv2.addWeighted(overlay, 1.0, tint, 0.45, 0)
            # Draw contour so mask edges are visible even at low opacity.
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, colour, 2)
            # Label in the centroid of the mask.
            ys, xs = np.where(mask.astype(bool))
            if len(xs):
                cx_m, cy_m = int(xs.mean()), int(ys.mean())
                cv2.putText(overlay, str(obj_id), (cx_m, cy_m),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.4, w / 1200),
                            (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(masks_path, overlay)
        logger.info("DepthEnvironmentCollider: depth debug (masks) saved → %s", masks_path)

    def _depth_to_world_points(
        self,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        T_base_cam: np.ndarray,
        pixel_mask: np.ndarray,
    ):
        """Backproject masked depth pixels to world-frame 3-D points.

        Returns:
            (pts_world, rows_kept, cols_kept) where rows/cols are the pixel
            coordinates of the surviving points (after Z clamping), or
            (None, None, None) if no valid points remain.
        """
        h, w = depth_m.shape
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.cx, intrinsics.cy

        rows = np.arange(0, h, _DEPTH_STRIDE)
        cols = np.arange(0, w, _DEPTH_STRIDE)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")

        # Subsample the pixel mask to match the stride grid
        sampled_mask = pixel_mask[rr, cc]
        d = depth_m[rr, cc]

        valid = sampled_mask & (d > _DEPTH_MIN_M) & (d < _DEPTH_MAX_M)
        if not np.any(valid):
            return None, None, None

        rows_v = rr[valid]
        cols_v = cc[valid]
        d_v = d[valid].astype(float)
        x = (cols_v.astype(float) - cx) * d_v / fx
        y = (rows_v.astype(float) - cy) * d_v / fy
        pts_cam = np.stack([x, y, d_v], axis=1)

        R = T_base_cam[:3, :3]
        t = T_base_cam[:3, 3]
        pts_world = ((R @ pts_cam.T).T + t).astype(np.float32)

        # Discard points outside the valid workspace Z range — these are
        # depth noise or mis-projected background points below the floor.
        z_valid = (pts_world[:, 2] >= _WORLD_Z_MIN_M) & (pts_world[:, 2] <= _WORLD_Z_MAX_M)
        pts_world = pts_world[z_valid]
        rows_v    = rows_v[z_valid]
        cols_v    = cols_v[z_valid]
        if len(pts_world) == 0:
            return None, None, None
        return pts_world, rows_v, cols_v

    def _build_body(self, pts: Optional[np.ndarray], label: str) -> bool:
        """Reconstruct mesh from points and add it to PyBullet. Returns success."""
        if pts is None or len(pts) < 10:
            logger.debug("DepthEnvironmentCollider: too few points for %s (%s)",
                         label, len(pts) if pts is not None else 0)
            return False

        mesh = self._points_to_mesh(pts, is_background=(label == BACKGROUND_ID))
        if mesh is None:
            logger.debug("DepthEnvironmentCollider: mesh failed for %s", label)
            return False

        body_id = self._mesh_to_pybullet(mesh, label=label)
        if body_id is None:
            return False

        self._bodies[label] = body_id
        logger.info(
            "DepthEnvironmentCollider: [%s] body=%d  triangles=%d  points=%d",
            label, body_id, len(np.asarray(mesh.triangles)), len(pts),
        )
        return True

    def _points_to_mesh(self, pts: np.ndarray, is_background: bool = False) -> Optional[object]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # Voxel downsample to a uniform 3 mm grid so BPA ball radius is consistent
        # regardless of object distance or stride density variation.
        pcd = pcd.voxel_down_sample(_VOXEL_SIZE_M)
        if len(pcd.points) < 10:
            return None

        # Remove statistical outliers — eliminates isolated noise points at depth
        # edges that BPA would otherwise bridge into phantom triangles.
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=_OUTLIER_NB_POINTS,
            std_ratio=_OUTLIER_STD_RATIO,
        )
        if len(pcd.points) < 10:
            return None

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        if self._cam_pos_world is not None:
            pcd.orient_normals_towards_camera_location(
                camera_location=self._cam_pos_world.tolist()
            )

        distances = pcd.compute_nearest_neighbor_distance()
        if len(distances) == 0:
            return None
        mean_spacing = float(np.mean(distances))
        bpa_factor = _BPA_RADIUS_FACTOR_BG if is_background else _BPA_RADIUS_FACTOR_OBJ
        radius = min(bpa_factor * mean_spacing, _BPA_RADIUS_MAX_M)

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector([radius, radius * 2]),
        )
        if len(mesh.triangles) == 0:
            return None

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()

        # Remove triangles whose longest edge exceeds the artifact threshold.
        # These are BPA fins that span depth discontinuities or large point gaps.
        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        max_edge = _LONG_EDGE_FACTOR * mean_spacing
        v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
        longest = np.maximum(
            np.linalg.norm(v1 - v0, axis=1),
            np.maximum(np.linalg.norm(v2 - v1, axis=1), np.linalg.norm(v0 - v2, axis=1)),
        )
        mesh.remove_triangles_by_index(np.where(longest > max_edge)[0].tolist())
        mesh.remove_unreferenced_vertices()
        if len(mesh.triangles) == 0:
            return None

        # Remove small disconnected components — noise islands left after edge filtering.
        # Threshold scales with total triangle count so small objects aren't over-pruned.
        triangle_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_tris = np.asarray(cluster_n_tris)
        total_tris = int(cluster_n_tris.sum())
        min_tris = max(_MIN_COMPONENT_TRIS_FLOOR, int(total_tris * _MIN_COMPONENT_TRIS_RATIO))
        keep_clusters = set(int(i) for i, n in enumerate(cluster_n_tris) if n >= min_tris)
        if not keep_clusters:
            return None
        mesh.remove_triangles_by_index(
            np.where(~np.isin(triangle_clusters, list(keep_clusters)))[0].tolist()
        )
        mesh.remove_unreferenced_vertices()
        if len(mesh.triangles) == 0:
            return None

        return mesh

    # RGBA colours cycled per object label for GUI visibility.
    _LABEL_COLOURS = [
        (1.0, 0.3, 0.3, 0.6),   # red
        (0.3, 1.0, 0.3, 0.6),   # green
        (0.3, 0.6, 1.0, 0.6),   # blue
        (1.0, 0.8, 0.2, 0.6),   # yellow
        (0.8, 0.3, 1.0, 0.6),   # purple
        (0.2, 0.9, 0.9, 0.6),   # cyan
    ]
    _BACKGROUND_COLOUR = (0.7, 0.7, 0.7, 0.35)
    _label_counter: int = 0

    def _mesh_to_pybullet(self, mesh, label: str = "") -> Optional[int]:
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        tris = np.asarray(mesh.triangles, dtype=np.int32)
        if len(verts) == 0 or len(tris) == 0:
            return None
        client = self._planner._physics_client
        try:
            col_id = p.createCollisionShape(
                p.GEOM_MESH,
                vertices=verts.tolist(),
                indices=tris.flatten().tolist(),
                physicsClientId=client,
            )
            if label == BACKGROUND_ID:
                rgba = self._BACKGROUND_COLOUR
            else:
                rgba = self._LABEL_COLOURS[self._label_counter % len(self._LABEL_COLOURS)]
                self._label_counter += 1
            vis_id = p.createVisualShape(
                p.GEOM_MESH,
                vertices=verts.tolist(),
                indices=tris.flatten().tolist(),
                rgbaColor=rgba,
                physicsClientId=client,
            )
            return p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=[0, 0, 0],
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=client,
            )
        except Exception as exc:
            logger.warning("DepthEnvironmentCollider: pybullet mesh creation failed: %s", exc)
            return None
