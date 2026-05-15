"""
SAM 3D Objects inference server.

Loads the SAM 3D Objects model once and serves a /reconstruct HTTP endpoint
so the main PTP pipeline can query it without loading 32 GB of weights locally.
Deploy on any machine with sufficient VRAM (≥32 GB), including HuggingFace Spaces
(use a T4-Large or A10G Space with the sam3d-objects environment).

Usage:
    python scripts/sam3d_server.py [--port 8766] [--checkpoint checkpoints/hf]

    # Expose on LAN / HuggingFace Spaces (bind all interfaces):
    python scripts/sam3d_server.py --host 0.0.0.0 --port 7860

Endpoints:

    GET  /health
    Response: {"status": "ok"}

    POST /reconstruct
    Content-Type: application/json
    Body:
      {
        "image_b64":  "<base64-encoded PNG — RGB or RGBA>",
        "mask_b64":   "<base64-encoded single-channel PNG (0=bg, 255=obj)>",
        "seed":       42   (optional)
      }
    Response (success):
      {
        "mesh_ply_b64": "<base64-encoded binary PLY mesh>",
        "n_triangles":  <int>
      }
    Response (error):
      {"error": "<message>"}
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import sys
from pathlib import Path

import numpy as np

# inference.py lives in the notebook/ subdirectory of the sam-3d-objects repo.
# Add it to sys.path so `from inference import Inference` works regardless of
# where this script is invoked from.
_here = Path(__file__).resolve().parent
# Derive sam-3d-objects root from SAM3D_CKPT env var (checkpoints/hf → ../..)
_sam3d_ckpt = os.environ.get("SAM3D_CKPT", "")
_sam3d_home = Path(_sam3d_ckpt).parent.parent if _sam3d_ckpt else Path()
for _candidate in [
    _here / "notebook",
    _here.parent / "notebook",
    _sam3d_home / "notebook",
    Path.home() / "installs" / "sam-3d-objects" / "notebook",
]:
    if _candidate.exists():
        sys.path.insert(0, str(_candidate))
        break

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("sam3d_server")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM 3D Objects inference server")
    p.add_argument("--port", type=int, default=8766, help="TCP port to listen on")
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind address (use 0.0.0.0 to expose on LAN / HuggingFace Spaces)")
    p.add_argument(
        "--checkpoint",
        default=os.environ.get("SAM3D_CKPT", "checkpoints/hf"),
        help="Path to SAM3D checkpoint directory containing pipeline.yaml",
    )
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model (faster inference, slower first call)")
    return p.parse_args()


def _load_model(checkpoint_dir: str, compile: bool = False):
    from pathlib import Path
    pipeline_yaml = Path(checkpoint_dir) / "pipeline.yaml"
    if not pipeline_yaml.exists():
        logger.error("pipeline.yaml not found at %s", pipeline_yaml)
        sys.exit(1)

    try:
        from inference import Inference
    except ImportError:
        logger.error(
            "sam3d-objects package not installed. "
            "Follow setup instructions in environments/default.yml."
        )
        sys.exit(1)

    logger.info("Loading SAM 3D Objects from %s …", pipeline_yaml)
    model = Inference(str(pipeline_yaml), compile=compile)
    logger.info("SAM 3D Objects model ready.")
    return model


def _depth_to_pointmap(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Back-project depth to a SAM3D-convention pointmap [-x, -y, z] (H, W, 3)."""
    H, W = depth.shape
    u = np.arange(W)[None, :].repeat(H, axis=0).astype(np.float32)
    v = np.arange(H)[:, None].repeat(W, axis=1).astype(np.float32)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = depth.astype(np.float32)
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return np.stack([-x, -y, z], axis=-1)


def _apply_sam3d_transform(vertices, output: dict) -> np.ndarray:
    """Apply SAM3D's predicted scale/rotation/translation to mesh vertices.

    Replicates the transform pipeline from the reference RGBD3DReconstructor:
        vertices → flip_z → yup_to_zup → scale/rotate/translate → pytorch3d_to_cam
    """
    import torch
    from pytorch3d.transforms import quaternion_to_matrix

    def _t(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float()
        return torch.tensor(np.asarray(x), dtype=torch.float32)

    R_flip_z     = torch.tensor([[1,0,0],[0,1,0],[0,0,-1]], dtype=torch.float32)
    R_yup_to_zup = torch.tensor([[-1,0,0],[0,0,1],[0,1,0]], dtype=torch.float32).T
    R_p3d_to_cam = torch.tensor([[-1,0,0],[0,-1,0],[0,0,1]], dtype=torch.float32)

    verts = torch.tensor(vertices, dtype=torch.float32)  # (N, 3)
    verts = verts @ R_flip_z
    verts = verts @ R_yup_to_zup

    S     = _t(output["scale"]).flatten()[0]          # scalar
    T     = _t(output["translation"]).flatten()[:3]   # (3,)
    R     = _t(output["rotation"]).flatten()[:4]      # quaternion (4,)
    R_mat = quaternion_to_matrix(R)                    # (3, 3)

    verts = verts * S
    verts = verts @ R_mat
    verts = verts + T
    verts = verts @ R_p3d_to_cam
    return verts.numpy().astype(np.float32)


def _run_inference(
    model,
    image_arr: np.ndarray,
    mask_arr: np.ndarray,
    pointmap: np.ndarray,
    intrinsic: np.ndarray,
    seed: int = 42,
) -> tuple:
    """Run SAM3D inference with depth-derived pointmap and return PLY bytes."""
    import open3d as o3d
    import tempfile
    import torch

    pointmap_t = torch.from_numpy(pointmap).float()
    output = model(image_arr, mask_arr, seed=seed, pointmap=pointmap_t)

    mesh_glb = output.get("glb")
    if mesh_glb is None:
        raise ValueError("SAM3D output missing 'glb' key")

    vertices = np.asarray(mesh_glb.vertices)
    faces = np.asarray(mesh_glb.faces)

    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("SAM3D returned empty mesh")

    vertices = _apply_sam3d_transform(vertices, output)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    o3d_mesh.compute_vertex_normals()

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        tmp_path = f.name
    o3d.io.write_triangle_mesh(tmp_path, o3d_mesh, write_ascii=False)
    ply_bytes = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    return ply_bytes, len(faces)


def main() -> None:
    args = _parse_args()

    try:
        from flask import Flask, jsonify, request
    except ImportError:
        logger.error("flask is required: pip install flask")
        sys.exit(1)

    model = _load_model(args.checkpoint, compile=args.compile)
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/reconstruct", methods=["POST"])
    def reconstruct():
        data = request.get_json(force=True)
        image_b64    = data.get("image_b64", "")
        mask_b64     = data.get("mask_b64", "")
        depth_b64    = data.get("depth_b64", "")
        intrinsic    = data.get("intrinsic", None)  # [[fx,0,cx],[0,fy,cy],[0,0,1]]
        seed = int(data.get("seed", 42))

        if not image_b64 or not mask_b64:
            return jsonify({"error": "image_b64 and mask_b64 are required"}), 400
        if not depth_b64 or intrinsic is None:
            return jsonify({"error": "depth_b64 and intrinsic are required"}), 400

        try:
            from PIL import Image as _PIL
            import numpy as np

            image_arr = np.array(_PIL.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB"))
            mask_arr  = (np.array(_PIL.open(io.BytesIO(base64.b64decode(mask_b64))).convert("L")) > 127).astype(np.uint8)

            depth_bytes = base64.b64decode(depth_b64)
            depth_arr   = np.load(io.BytesIO(depth_bytes))["depth_m"].astype(np.float32)
            K           = np.array(intrinsic, dtype=np.float32)
            pointmap    = _depth_to_pointmap(depth_arr, K)
        except Exception as exc:
            return jsonify({"error": f"input decode failed: {exc}"}), 400

        try:
            ply_bytes, n_tris = _run_inference(
                model, image_arr, mask_arr, pointmap, K, seed=seed
            )
        except Exception as exc:
            logger.exception("SAM3D inference failed")
            return jsonify({"error": f"inference failed: {exc}"}), 500

        mesh_b64 = base64.b64encode(ply_bytes).decode()
        logger.info("reconstruct | n_triangles=%d  ply_bytes=%d", n_tris, len(ply_bytes))
        return jsonify({"mesh_ply_b64": mesh_b64, "n_triangles": n_tris})

    logger.info("Starting SAM3D server on %s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
