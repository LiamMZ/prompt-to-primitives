"""SAM3D mesh reconstruction for per-object collision bodies.

Wraps the SAM 3D Objects inference API to produce Open3D TriangleMesh objects
from a single RGB image + 2D segmentation mask.

Two modes:
  - Local:  Sam3DMeshifier(checkpoint_dir="checkpoints/hf")
  - Remote: Sam3DMeshifier(server_url="http://host:8766")

Usage::

    # local
    meshifier = Sam3DMeshifier(checkpoint_dir="checkpoints/hf")
    mesh = meshifier.reconstruct(color_rgb, mask_bool, T_base_cam)

    # remote (server running scripts/sam3d_server.py)
    meshifier = Sam3DMeshifier(server_url="http://gpu-host:8766")
    mesh = meshifier.reconstruct(color_rgb, mask_bool, T_base_cam)

    # batch
    meshes = meshifier.reconstruct_all(color_rgb, masks_dict, T_base_cam)
"""

from __future__ import annotations

import base64
import io
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Sam3DMeshifier:
    """Reconstruct per-object 3D meshes using SAM 3D Objects.

    Pass either ``checkpoint_dir`` (local inference) or one or more
    ``server_url`` values (remote servers started with
    ``scripts/sam3d_server.py``).  Multiple URLs enable parallel
    reconstruction — one object per server at a time.

    Args:
        checkpoint_dir: Path to the SAM3D checkpoint directory (local mode).
        server_url: Single server URL, or comma-separated list of URLs.
            e.g. ``"http://host:8766"`` or
            ``"http://host:8766,http://host:8767"``.
        compile: torch.compile the model (local mode, slower first call).
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/hf",
        server_url: Optional[str] = None,
        compile: bool = False,
    ) -> None:
        if server_url:
            self._server_urls: List[str] = [u.rstrip("/") for u in server_url.split(",")]
        else:
            self._server_urls = []
        self._checkpoint_dir = Path(checkpoint_dir)
        self._compile = compile
        self._inference = None  # local model, lazy-loaded

    @property
    def is_remote(self) -> bool:
        return bool(self._server_urls)

    @property
    def _server_url(self) -> Optional[str]:
        return self._server_urls[0] if self._server_urls else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the model (local mode) or health-check the server (remote mode)."""
        if self.is_remote:
            self._check_health()
            return
        if self._inference is not None:
            return
        pipeline_yaml = self._checkpoint_dir / "pipeline.yaml"
        if not pipeline_yaml.exists():
            raise FileNotFoundError(
                f"SAM3D pipeline.yaml not found at {pipeline_yaml}. "
                "Download checkpoints with: hf download facebook/sam-3d-objects"
            )
        try:
            from inference import Inference  # sam3d-objects package
        except ImportError as exc:
            raise ImportError(
                "sam3d-objects package not installed. "
                "Follow setup instructions in environments/default.yml."
            ) from exc
        logger.info("Loading SAM 3D Objects model from %s …", pipeline_yaml)
        self._inference = Inference(str(pipeline_yaml), compile=self._compile)
        logger.info("SAM 3D Objects model loaded.")

    def reconstruct(
        self,
        color_rgb: np.ndarray,
        mask: np.ndarray,
        T_base_cam: np.ndarray,
        seed: int = 42,
        depth_m: Optional[np.ndarray] = None,
        intrinsics: Optional[Any] = None,
    ) -> Optional[object]:
        """Reconstruct a single object mesh in the robot base frame.

        Args:
            color_rgb: (H, W, 3) uint8 RGB image.
            mask: (H, W) bool mask selecting the object pixels.
            T_base_cam: (4, 4) homogeneous camera-to-base transform.
            seed: Random seed for reproducibility.
            depth_m: (H, W) float32 depth image in metres. Used to scale and
                localise the mesh to match the real object size and position.
            intrinsics: Camera intrinsics (fx, fy, cx, cy). Required with depth_m.

        Returns:
            open3d.geometry.TriangleMesh in base frame, or None on failure.
        """
        try:
            return self._reconstruct_one(color_rgb, mask, T_base_cam, seed,
                                         depth_m=depth_m, intrinsics=intrinsics)
        except Exception as exc:
            logger.warning("SAM3D reconstruction failed: %s", exc)
            return None

    def reconstruct_all(
        self,
        color_rgb: np.ndarray,
        masks: Dict[str, np.ndarray],
        T_base_cam: np.ndarray,
        seed: int = 42,
        depth_m: Optional[np.ndarray] = None,
        intrinsics: Optional[Any] = None,
    ) -> Dict[str, Optional[object]]:
        """Reconstruct meshes for all objects in ``masks``.

        When multiple server URLs are configured, objects are distributed
        across servers and reconstructed in parallel.

        Args:
            depth_m: (H, W) float32 depth image in metres. Used to scale and
                localise each mesh to match the real object size and position.
            intrinsics: Camera intrinsics. Required with depth_m.

        Returns a dict mapping object_id → mesh (or None on per-object failure).
        """
        items = list(masks.items())
        n_servers = len(self._server_urls)

        if n_servers <= 1:
            results: Dict[str, Optional[object]] = {}
            for obj_id, mask in items:
                try:
                    results[obj_id] = self._reconstruct_one(
                        color_rgb, mask, T_base_cam, seed,
                        depth_m=depth_m, intrinsics=intrinsics,
                    )
                except Exception as exc:
                    logger.warning("SAM3D reconstruction failed for %r: %s", obj_id, exc)
                    results[obj_id] = None
            return results

        # Multiple servers — parallel, round-robin assignment
        def _worker(obj_id: str, mask: np.ndarray, server_idx: int):
            url = self._server_urls[server_idx % n_servers]
            meshifier = Sam3DMeshifier(server_url=url)
            try:
                return obj_id, meshifier._reconstruct_one(
                    color_rgb, mask, T_base_cam, seed,
                    depth_m=depth_m, intrinsics=intrinsics,
                )
            except Exception as exc:
                logger.warning("SAM3D reconstruction failed for %r on %s: %s", obj_id, url, exc)
                return obj_id, None

        results = {}
        with ThreadPoolExecutor(max_workers=n_servers) as ex:
            futures = {
                ex.submit(_worker, obj_id, mask, i): obj_id
                for i, (obj_id, mask) in enumerate(items)
            }
            for future in as_completed(futures):
                obj_id, mesh = future.result()
                results[obj_id] = mesh
        return results

    # ------------------------------------------------------------------
    # Internals — dispatch
    # ------------------------------------------------------------------

    def _reconstruct_one(
        self,
        color_rgb: np.ndarray,
        mask: np.ndarray,
        T_base_cam: np.ndarray,
        seed: int,
        depth_m: Optional[np.ndarray] = None,
        intrinsics: Optional[Any] = None,
    ) -> Optional[object]:
        if self.is_remote:
            mesh = self._reconstruct_remote(color_rgb, mask, seed,
                                            depth_m=depth_m, intrinsics=intrinsics)
        else:
            self.load()
            mesh = self._reconstruct_local(color_rgb, mask, seed,
                                           depth_m=depth_m, intrinsics=intrinsics)

        if mesh is None or len(mesh.triangles) == 0:
            return None

        # The server/local path applies SAM3D's own scale+rotation+translation in
        # camera frame.  We only need to transform into robot base frame here.
        mesh.transform(T_base_cam)
        return mesh

    # ------------------------------------------------------------------
    # Local inference
    # ------------------------------------------------------------------

    def _reconstruct_local(
        self,
        color_rgb: np.ndarray,
        mask: np.ndarray,
        seed: int,
        depth_m: Optional[np.ndarray] = None,
        intrinsics: Optional[Any] = None,
    ) -> Optional[object]:
        import open3d as o3d

        if depth_m is None or intrinsics is None:
            raise RuntimeError("depth_m and intrinsics are required for local SAM3D reconstruction")

        H, W = depth_m.shape
        u = np.arange(W)[None, :].repeat(H, axis=0).astype(np.float32)
        v = np.arange(H)[:, None].repeat(W, axis=1).astype(np.float32)
        fx = getattr(intrinsics, "fx", W / 2)
        fy = getattr(intrinsics, "fy", H / 2)
        cx = getattr(intrinsics, "ppx", getattr(intrinsics, "cx", W / 2))
        cy = getattr(intrinsics, "ppy", getattr(intrinsics, "cy", H / 2))
        z = depth_m.astype(np.float32)
        pointmap = np.stack([-(u - cx) / fx * z, -(v - cy) / fy * z, z], axis=-1)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        output = self._inference(
            color_rgb.astype(np.uint8), mask.astype(np.uint8),
            seed=seed, pointmap=pointmap, intrinsic=K,
        )

        mesh_glb = output.get("glb")
        if mesh_glb is None or len(mesh_glb.vertices) == 0:
            return None

        vertices = self._apply_sam3d_transform(np.asarray(mesh_glb.vertices), output)
        faces = np.asarray(mesh_glb.faces).astype(np.int32)

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d_mesh.compute_vertex_normals()
        return o3d_mesh if len(o3d_mesh.triangles) > 0 else None

    # ------------------------------------------------------------------
    # Remote inference
    # ------------------------------------------------------------------

    def _check_health(self) -> None:
        import urllib.request
        url = f"{self._server_url}/health"
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}")
        except Exception as exc:
            raise RuntimeError(
                f"SAM3D server not reachable at {self._server_url}: {exc}. "
                "Start it with: python scripts/sam3d_server.py"
            ) from exc
        logger.info("SAM3D server reachable at %s", self._server_url)

    def _reconstruct_remote(
        self,
        color_rgb: np.ndarray,
        mask: np.ndarray,
        seed: int,
        depth_m: Optional[np.ndarray] = None,
        intrinsics: Optional[Any] = None,
    ) -> Optional[object]:
        import urllib.request
        import json
        import urllib.error
        import open3d as o3d
        from PIL import Image as _PIL

        buf = io.BytesIO()
        _PIL.fromarray(color_rgb.astype(np.uint8)).save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        mask_buf = io.BytesIO()
        _PIL.fromarray((mask.astype(np.uint8) * 255)).save(mask_buf, format="PNG")
        mask_b64 = base64.b64encode(mask_buf.getvalue()).decode()

        if depth_m is None or intrinsics is None:
            raise RuntimeError("depth_m and intrinsics are required for SAM3D remote reconstruction")

        depth_buf = io.BytesIO()
        np.savez_compressed(depth_buf, depth_m=depth_m.astype(np.float32))
        depth_b64 = base64.b64encode(depth_buf.getvalue()).decode()

        fx = getattr(intrinsics, "fx", None)
        fy = getattr(intrinsics, "fy", None)
        cx = getattr(intrinsics, "ppx", getattr(intrinsics, "cx", None))
        cy = getattr(intrinsics, "ppy", getattr(intrinsics, "cy", None))
        K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]

        payload = json.dumps({
            "image_b64": image_b64,
            "mask_b64": mask_b64,
            "depth_b64": depth_b64,
            "intrinsic": K,
            "seed": seed,
        }).encode()

        req = urllib.request.Request(
            f"{self._server_url}/reconstruct",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = json.loads(r.read())
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("error", str(exc))
            except Exception:
                detail = str(exc)
            raise RuntimeError(f"SAM3D server error {exc.code}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"SAM3D server request failed: {exc}") from exc

        if "error" in body:
            raise RuntimeError(f"SAM3D server error: {body['error']}")

        ply_bytes = base64.b64decode(body["mesh_ply_b64"])
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            f.write(ply_bytes)
            tmp_path = f.name
        mesh = o3d.io.read_triangle_mesh(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)

        logger.debug("SAM3D remote: received mesh with %d triangles", len(mesh.triangles))
        return mesh if len(mesh.triangles) > 0 else None

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_sam3d_transform(vertices: np.ndarray, output: dict) -> np.ndarray:
        """Apply SAM3D's predicted scale/rotation/translation to mesh vertices.

        Replicates the transform pipeline from the SAM3D reference notebook:
            vertices → flip_z → yup_to_zup → scale/rotate/translate → pytorch3d_to_cam
        """
        import torch
        from pytorch3d.transforms import quaternion_to_matrix, Transform3d

        R_flip_z     = torch.tensor([[1,0,0],[0,1,0],[0,0,-1]], dtype=torch.float32)
        R_yup_to_zup = torch.tensor([[-1,0,0],[0,0,1],[0,1,0]], dtype=torch.float32).T
        R_p3d_to_cam = torch.tensor([[-1,0,0],[0,-1,0],[0,0,1]], dtype=torch.float32)

        verts = torch.tensor(vertices, dtype=torch.float32).unsqueeze(0)
        verts = verts @ R_flip_z
        verts = verts @ R_yup_to_zup

        S = output["scale"][0].cpu().float()
        T = output["translation"][0].cpu().float()
        R = output["rotation"].squeeze().cpu().float()
        R_mat = quaternion_to_matrix(R)

        tfm = Transform3d(dtype=torch.float32).scale(S).rotate(R_mat).translate(*T)
        verts = tfm.transform_points(verts)
        verts = verts @ R_p3d_to_cam
        return verts[0].cpu().numpy().astype(np.float32)

    @staticmethod
    def _pcd_to_mesh(pcd: Any) -> Optional[Any]:
        """Convert a Gaussian splat point cloud to a smooth mesh via Poisson reconstruction."""
        import open3d as o3d
        import numpy as np

        pcd = pcd.voxel_down_sample(0.003)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=1.5)
        if len(pcd.points) < 4:
            return None

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=7, linear_fit=False
        )
        if len(mesh.triangles) == 0:
            return None

        densities = np.asarray(densities)
        threshold = np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(densities < threshold)

        return mesh if len(mesh.triangles) > 0 else None

    @staticmethod
    def _mask_depth_points(
        depth_m: np.ndarray,
        mask: np.ndarray,
        intrinsics: Any,
    ) -> Optional[np.ndarray]:
        """Back-project masked depth pixels to 3D camera-frame points."""
        h, w = depth_m.shape
        bool_mask = mask.astype(bool)
        if bool_mask.shape != (h, w):
            return None

        ys, xs = np.where(bool_mask)
        if len(ys) == 0:
            return None

        zs = depth_m[ys, xs].astype(float)
        valid = (zs > 0.05) & np.isfinite(zs)
        ys, xs, zs = ys[valid], xs[valid], zs[valid]
        if len(zs) < 4:
            return None

        fx = getattr(intrinsics, "fx", w / 2)
        fy = getattr(intrinsics, "fy", h / 2)
        cx = getattr(intrinsics, "ppx", getattr(intrinsics, "cx", w / 2))
        cy = getattr(intrinsics, "ppy", getattr(intrinsics, "cy", h / 2))

        x = (xs.astype(float) - cx) * zs / fx
        y = (ys.astype(float) - cy) * zs / fy
        return np.stack([x, y, zs], axis=1).astype(np.float32)

    @staticmethod
    def _align_mesh_to_depth(mesh: Any, pts_cam: np.ndarray) -> None:
        """Scale and translate a SAM3D mesh to match the depth point cloud in-place.

        SAM3D returns a mesh in an arbitrary canonical coordinate frame — the shape
        is correct but the scale and position are not.  We:
          1. Centre the mesh at its own geometric centroid (remove canonical offset).
          2. Scale it uniformly so its bounding-box extents match the depth cloud.
          3. Translate it to the depth centroid in camera frame.
        """
        import open3d as o3d

        verts = np.asarray(mesh.vertices)
        if len(verts) == 0:
            return

        # 1. Centre mesh at its own centroid.
        mesh_centroid = verts.mean(axis=0)
        mesh.translate(-mesh_centroid)

        # 2. Compute per-axis extents of the mesh and the depth cloud.
        verts = np.asarray(mesh.vertices)  # re-read after translate
        mesh_extents = verts.max(axis=0) - verts.min(axis=0)

        depth_min = pts_cam.min(axis=0)
        depth_max = pts_cam.max(axis=0)
        depth_extents = depth_max - depth_min
        depth_centroid = pts_cam.mean(axis=0)

        # 3. Uniform scale: use the median ratio across axes to avoid outlier axes
        #    (e.g. Z extent is often underestimated due to occlusion).
        valid_axes = mesh_extents > 1e-4
        if not valid_axes.any():
            mesh.translate(depth_centroid)
            return

        ratios = depth_extents[valid_axes] / mesh_extents[valid_axes]
        scale = float(np.median(ratios))
        scale = max(0.01, min(scale, 5.0))  # clamp to sane range
        mesh.scale(scale, center=np.zeros(3))

        # 4. Translate to depth centroid.
        mesh.translate(depth_centroid)

        logger.debug(
            "SAM3D align: mesh_extents=%s depth_extents=%s scale=%.3f centroid=%s",
            np.round(mesh_extents, 3), np.round(depth_extents, 3),
            scale, np.round(depth_centroid, 3),
        )

    @staticmethod
    def _mask_centroid_cam_approx(mask: np.ndarray) -> Optional[np.ndarray]:
        """Fallback: image-centre normalized position when no depth is available."""
        ys, xs = np.where(mask.astype(bool))
        if len(ys) == 0:
            return None
        h, w = mask.shape[:2]
        cy, cx = float(ys.mean()), float(xs.mean())
        # Rough assumption: object is ~0.5 m away; gives a plausible camera-frame pos.
        return np.array([(cx / w - 0.5) * 0.5, (cy / h - 0.5) * 0.5, 0.5])
