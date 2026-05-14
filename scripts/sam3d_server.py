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


def _run_inference(model, pil_image, seed: int = 42) -> bytes:
    """Run SAM3D inference and return the mesh as binary PLY bytes."""
    import open3d as o3d
    import numpy as np
    from pathlib import Path
    import tempfile

    output = model(pil_image, seed=seed)

    gs = output.get("gs")
    if gs is None:
        raise ValueError("SAM3D output missing 'gs' key")

    # Convert Gaussian splat to convex-hull mesh for collision use.
    if isinstance(gs, (str, Path)):
        pcd = o3d.io.read_point_cloud(str(gs))
    else:
        pts = np.asarray(gs.get_xyz().cpu())
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

    if len(pcd.points) < 4:
        raise ValueError(f"Point cloud too sparse: {len(pcd.points)} points")

    hull, _ = pcd.compute_convex_hull()
    hull.orient_triangles()

    if len(hull.triangles) == 0:
        raise ValueError("Convex hull produced no triangles")

    # Serialise to PLY bytes.
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        tmp_path = f.name
    o3d.io.write_triangle_mesh(tmp_path, hull, write_ascii=False)
    ply_bytes = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    return ply_bytes, len(hull.triangles)


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
        image_b64 = data.get("image_b64", "")
        mask_b64 = data.get("mask_b64", "")
        seed = int(data.get("seed", 42))

        if not image_b64 or not mask_b64:
            return jsonify({"error": "image_b64 and mask_b64 are required"}), 400

        try:
            from PIL import Image
            import numpy as np

            img_bytes = base64.b64decode(image_b64)
            pil_rgb = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            mask_bytes = base64.b64decode(mask_b64)
            mask_arr = np.array(Image.open(io.BytesIO(mask_bytes)).convert("L"))
            alpha = (mask_arr > 127).astype(np.uint8) * 255

            # Embed mask as alpha channel — SAM3D expects RGBA with alpha = object mask.
            rgba = np.dstack([np.array(pil_rgb), alpha])
            pil_rgba = Image.fromarray(rgba, mode="RGBA")
        except Exception as exc:
            return jsonify({"error": f"input decode failed: {exc}"}), 400

        try:
            ply_bytes, n_tris = _run_inference(model, pil_rgba, seed=seed)
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
