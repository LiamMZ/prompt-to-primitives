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
from typing import Dict, List, Optional

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
    ) -> Optional[object]:
        """Reconstruct a single object mesh in the robot base frame.

        Args:
            color_rgb: (H, W, 3) uint8 RGB image.
            mask: (H, W) bool mask selecting the object pixels.
            T_base_cam: (4, 4) homogeneous camera-to-base transform.
            seed: Random seed for reproducibility.

        Returns:
            open3d.geometry.TriangleMesh in base frame, or None on failure.
        """
        try:
            return self._reconstruct_one(color_rgb, mask, T_base_cam, seed)
        except Exception as exc:
            logger.warning("SAM3D reconstruction failed: %s", exc)
            return None

    def reconstruct_all(
        self,
        color_rgb: np.ndarray,
        masks: Dict[str, np.ndarray],
        T_base_cam: np.ndarray,
        seed: int = 42,
    ) -> Dict[str, Optional[object]]:
        """Reconstruct meshes for all objects in ``masks``.

        When multiple server URLs are configured, objects are distributed
        across servers and reconstructed in parallel.

        Returns a dict mapping object_id → mesh (or None on per-object failure).
        """
        items = list(masks.items())
        n_servers = len(self._server_urls)

        if n_servers <= 1:
            # Single server or local — sequential
            results: Dict[str, Optional[object]] = {}
            for obj_id, mask in items:
                try:
                    results[obj_id] = self._reconstruct_one(color_rgb, mask, T_base_cam, seed)
                except Exception as exc:
                    logger.warning("SAM3D reconstruction failed for %r: %s", obj_id, exc)
                    results[obj_id] = None
            return results

        # Multiple servers — parallel, round-robin assignment
        def _worker(obj_id: str, mask: np.ndarray, server_idx: int):
            url = self._server_urls[server_idx % n_servers]
            meshifier = Sam3DMeshifier(server_url=url)
            try:
                return obj_id, meshifier._reconstruct_one(color_rgb, mask, T_base_cam, seed)
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
    ) -> Optional[object]:
        if self.is_remote:
            mesh = self._reconstruct_remote(color_rgb, mask, seed)
        else:
            self.load()
            mesh = self._reconstruct_local(color_rgb, mask, seed)

        if mesh is None or len(mesh.triangles) == 0:
            return None

        # Place mesh at the depth-derived centroid of the mask, then transform
        # into the robot base frame.
        cam_translation = self._mask_centroid_cam(color_rgb, mask)
        if cam_translation is None:
            return None
        mesh.translate(cam_translation)
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
    ) -> Optional[object]:
        from PIL import Image as _PIL

        output = self._inference(color_rgb.astype(np.uint8), mask.astype(np.uint8), seed=seed)
        return self._output_to_open3d(output)

    def _output_to_open3d(self, output: dict) -> Optional[object]:
        import open3d as o3d

        gs = output.get("gs")
        if gs is None:
            return None

        import tempfile
        if isinstance(gs, (str, Path)):
            pcd = o3d.io.read_point_cloud(str(gs))
        else:
            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
                gs_ply_path = f.name
            try:
                gs.save_ply(gs_ply_path)
                pcd = o3d.io.read_point_cloud(gs_ply_path)
            finally:
                Path(gs_ply_path).unlink(missing_ok=True)

        if len(pcd.points) < 4:
            return None

        hull, _ = pcd.compute_convex_hull()
        hull.orient_triangles()
        return hull

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
    ) -> Optional[object]:
        import urllib.request
        import json
        import open3d as o3d
        from PIL import Image as _PIL

        # Encode RGB image as PNG base64.
        buf = io.BytesIO()
        _PIL.fromarray(color_rgb.astype(np.uint8)).save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        # Encode mask as single-channel PNG base64.
        mask_buf = io.BytesIO()
        _PIL.fromarray((mask.astype(np.uint8) * 255)).save(mask_buf, format="PNG")
        mask_b64 = base64.b64encode(mask_buf.getvalue()).decode()

        payload = json.dumps({
            "image_b64": image_b64,
            "mask_b64": mask_b64,
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

    def _mask_centroid_cam(
        self,
        color_rgb: np.ndarray,
        mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        ys, xs = np.where(mask.astype(bool))
        if len(ys) == 0:
            return None
        cy, cx = float(ys.mean()), float(xs.mean())
        h, w = mask.shape[:2]
        return np.array([cx / w - 0.5, cy / h - 0.5, 0.0])
