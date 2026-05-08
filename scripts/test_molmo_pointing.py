"""test_molmo_pointing.py — Interactive Molmo pointing test for primitive parameters.

Each query type is selected independently via --queries. Available queries:

  grasp        — best pick/grasp interaction point
  push         — push contact surface + surface normal (PCA)
  pull         — pull contact surface + surface normal (PCA)
  push-pull    — push AND pull surface in one pass (matches executor pipeline)
  hinge        — hinge / pivot point for pivot_pull (requires --hinge-location)
  push-aside   — best surface to push an object aside

Default is all queries. Use --queries to run a subset, e.g.:
  --queries grasp push-pull
  --queries pull hinge

Usage — live RealSense:
    python scripts/test_molmo_pointing.py --object-type "cup"
    python scripts/test_molmo_pointing.py --object-type "drawer" --queries pull hinge \\
        --hinge-location "left edge of the drawer"

Usage — saved images:
    python scripts/test_molmo_pointing.py \\
        --rgb-path /tmp/rgb.png --depth-path /tmp/depth.npy \\
        --object-type "bottle" --queries grasp push
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ptp.utils.logging_utils import configure_logging, get_structured_logger
from ptp.camera import CameraIntrinsics, REALSENSE_AVAILABLE
from ptp.perception.molmo_point_detector import MolmoPointDetector
from ptp.perception.surface_normal import compute_surface_normal
from ptp.perception.pointing_prompts import build_prompt, build_hinge_prompt

logger = get_structured_logger(__name__)

# ---------------------------------------------------------------------------
# Run output directory  (outputs/molmo_pointing/runs/YYYYMMDD_HHMMSS)
# ---------------------------------------------------------------------------

_SCRIPT_NAME    = Path(__file__).stem                        # test_molmo_pointing
_RUN_ID         = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_BASE_OUTPUT    = _REPO_ROOT / "outputs" / _SCRIPT_NAME
_DEFAULT_OUTDIR = _BASE_OUTPUT / "runs" / _RUN_ID


def _init_run_dir(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = _BASE_OUTPUT / "runs" / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.resolve())
    logger.info("Run output : %s", run_dir)
    logger.info("Latest     : %s", latest)

ALL_QUERIES = ["grasp", "push", "pull", "push-pull", "hinge", "push-aside"]

_QUERY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "grasp":       (0, 255, 0),      # green
    "push":        (0, 100, 255),    # orange
    "pull":        (0, 0, 255),      # red
    "push-pull":   (0, 180, 255),    # amber
    "hinge":       (255, 0, 255),    # magenta
    "push-aside":  (255, 100, 0),    # blue
    "normal":      (0, 220, 255),    # yellow
}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _draw_point(
    img: np.ndarray,
    x_px: float,
    y_px: float,
    label: str,
    color: Tuple[int, int, int],
    radius: int = 12,
) -> np.ndarray:
    import cv2
    out = img.copy()
    cx, cy = int(round(x_px)), int(round(y_px))
    cv2.circle(out, (cx, cy), radius, color, -1)
    cv2.circle(out, (cx, cy), radius + 2, (255, 255, 255), 2)
    cv2.putText(out, label, (cx + radius + 4, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, label, (cx + radius + 4, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return out


def _draw_normal_arrow(
    img: np.ndarray,
    x_px: float,
    y_px: float,
    normal_cam: np.ndarray,
    label: str,
    length: int = 70,
) -> np.ndarray:
    """Project camera-frame normal onto image plane and draw as arrow.

    Camera X → image right, camera Y → image down, camera Z → depth (not shown).
    """
    import cv2
    out = img.copy()
    nx, ny = float(normal_cam[0]), float(normal_cam[1])
    mag = (nx ** 2 + ny ** 2) ** 0.5
    if mag < 1e-6:
        return out
    nx, ny = nx / mag, ny / mag
    tip_x = int(round(x_px + nx * length))
    tip_y = int(round(y_px + ny * length))
    color = _QUERY_COLORS["normal"]
    cv2.arrowedLine(out, (int(round(x_px)), int(round(y_px))), (tip_x, tip_y),
                    color, 3, tipLength=0.3)
    cv2.putText(out, label, (tip_x + 4, tip_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def _save(img_bgr: np.ndarray, path: Path, tag: str) -> None:
    import cv2
    path.mkdir(parents=True, exist_ok=True)
    fname = path / f"{tag}.png"
    cv2.imwrite(str(fname), img_bgr)
    logger.info("Saved %s", fname)


def _ip_to_px(ip, img_h: int, img_w: int) -> Tuple[float, float]:
    """InteractionPoint position_2d (norm [y,x] 0-1000) → pixel (x_px, y_px)."""
    norm_y, norm_x = ip.position_2d
    return float(norm_x) / 1000.0 * img_w, float(norm_y) / 1000.0 * img_h


def _normal_for_ip(ip, depth: np.ndarray, intrinsics: CameraIntrinsics, img_h: int, img_w: int):
    """Compute PCA surface normal at an InteractionPoint's pixel location."""
    px_row = float(ip.position_2d[0]) / 1000.0 * img_h
    px_col = float(ip.position_2d[1]) / 1000.0 * img_w
    normal_cam, confidence = compute_surface_normal(
        depth=depth,
        fx=intrinsics.fx, fy=intrinsics.fy,
        cx=intrinsics.cx, cy=intrinsics.cy,
        center_yx=(px_row, px_col),
        radius_px=40.0,
        method="pca",
    )
    return normal_cam, confidence


# ---------------------------------------------------------------------------
# Camera / image loading
# ---------------------------------------------------------------------------

def _capture_realsense(warmup_frames: int = 30) -> Tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
    from ptp.camera.realsense_camera import RealSenseCamera
    logger.info("Opening RealSense camera …")
    with RealSenseCamera(width=640, height=480, fps=30) as cam:
        logger.info("Warming up — discarding %d frames for auto-exposure to settle …", warmup_frames)
        for _ in range(warmup_frames):
            cam.get_aligned_frames()
        color, depth = cam.get_aligned_frames()
        intrinsics = cam.get_camera_intrinsics()
    logger.info(
        "Captured 640x480  fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
        intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
    )
    return color, depth, intrinsics


def _load_from_disk(rgb_path: str, depth_path: str) -> Tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
    import cv2
    color_bgr = cv2.imread(rgb_path)
    if color_bgr is None:
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    depth = np.load(depth_path).astype(np.float32)
    h, w = color.shape[:2]
    intrinsics = CameraIntrinsics(fx=600.0, fy=600.0, cx=w / 2.0, cy=h / 2.0, width=w, height=h)
    logger.info("Loaded RGB %s depth %s", color.shape, depth.shape)
    return color, depth, intrinsics


# ---------------------------------------------------------------------------
# Individual query runners — each returns a result dict or None
# ---------------------------------------------------------------------------

def query_grasp(
    detector: MolmoPointDetector,
    color: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    object_type: str,
) -> Optional[dict]:
    logger.info("=== GRASP interaction point ===")
    results = detector.get_interaction_points(
        rgb_image=color, depth_frame=depth, camera_intrinsics=intrinsics,
        object_id=f"{object_type}_0", object_type=object_type,
        bounding_box_2d=None, actions={"pick"},
    )
    ip = results.get("pick")
    if ip is None:
        logger.warning("grasp: no point returned")
        return None
    h, w = color.shape[:2]
    x_px, y_px = _ip_to_px(ip, h, w)
    logger.info("grasp: pixel=(%.1f, %.1f)  3d=%s", x_px, y_px,
                [f"{v:.3f}" for v in ip.position_3d] if ip.position_3d is not None else "N/A")
    return {"query": "grasp", "ip": ip, "x_px": x_px, "y_px": y_px,
            "label": "grasp", "color": _QUERY_COLORS["grasp"]}


def query_push(
    detector: MolmoPointDetector,
    color: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    object_type: str,
) -> Optional[dict]:
    logger.info("=== PUSH surface contact + normal ===")
    prompt = f"Point to the surface of the {object_type} where I should push against it."
    results = detector.get_interaction_points(
        rgb_image=color, depth_frame=depth, camera_intrinsics=intrinsics,
        object_id=f"{object_type}_0", object_type=object_type,
        bounding_box_2d=None, actions={"_surface"},
        custom_prompts={"_surface": prompt},
    )
    ip = results.get("_surface")
    if ip is None:
        logger.warning("push: no surface point returned")
        return None
    h, w = color.shape[:2]
    x_px, y_px = _ip_to_px(ip, h, w)
    normal_cam, confidence = _normal_for_ip(ip, depth, intrinsics, h, w)
    logger.info("push: pixel=(%.1f, %.1f)  3d=%s", x_px, y_px,
                [f"{v:.3f}" for v in ip.position_3d] if ip.position_3d is not None else "N/A")
    if normal_cam is not None:
        logger.info("push: normal=[%.3f, %.3f, %.3f]  confidence=%.2f",
                    normal_cam[0], normal_cam[1], normal_cam[2], confidence)
    else:
        logger.warning("push: surface normal estimation failed")
    return {"query": "push", "ip": ip, "x_px": x_px, "y_px": y_px,
            "label": "push surface", "color": _QUERY_COLORS["push"],
            "normal_cam": normal_cam, "normal_confidence": confidence}


def query_pull(
    detector: MolmoPointDetector,
    color: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    object_type: str,
) -> Optional[dict]:
    logger.info("=== PULL surface contact + normal ===")
    prompt = f"Point to the surface of the {object_type} where I should pull from it."
    results = detector.get_interaction_points(
        rgb_image=color, depth_frame=depth, camera_intrinsics=intrinsics,
        object_id=f"{object_type}_0", object_type=object_type,
        bounding_box_2d=None, actions={"_surface"},
        custom_prompts={"_surface": prompt},
    )
    ip = results.get("_surface")
    if ip is None:
        logger.warning("pull: no surface point returned")
        return None
    h, w = color.shape[:2]
    x_px, y_px = _ip_to_px(ip, h, w)
    normal_cam, confidence = _normal_for_ip(ip, depth, intrinsics, h, w)
    logger.info("pull: pixel=(%.1f, %.1f)  3d=%s", x_px, y_px,
                [f"{v:.3f}" for v in ip.position_3d] if ip.position_3d is not None else "N/A")
    if normal_cam is not None:
        logger.info("pull: normal=[%.3f, %.3f, %.3f]  confidence=%.2f",
                    normal_cam[0], normal_cam[1], normal_cam[2], confidence)
    else:
        logger.warning("pull: surface normal estimation failed")
    return {"query": "pull", "ip": ip, "x_px": x_px, "y_px": y_px,
            "label": "pull surface", "color": _QUERY_COLORS["pull"],
            "normal_cam": normal_cam, "normal_confidence": confidence}


def query_push_pull(
    detector: MolmoPointDetector,
    color: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    object_type: str,
) -> Optional[dict]:
    """Single Molmo call asking for the push/pull contact surface — mirrors executor pipeline."""
    logger.info("=== PUSH-PULL surface contact + normal (executor pipeline) ===")
    prompt = (
        f"Point to the surface of the {object_type} where I should "
        f"push or pull against it."
    )
    results = detector.get_interaction_points(
        rgb_image=color, depth_frame=depth, camera_intrinsics=intrinsics,
        object_id=f"{object_type}_0", object_type=object_type,
        bounding_box_2d=None, actions={"_surface"},
        custom_prompts={"_surface": prompt},
    )
    ip = results.get("_surface")
    if ip is None:
        logger.warning("push-pull: no surface point returned")
        return None
    h, w = color.shape[:2]
    x_px, y_px = _ip_to_px(ip, h, w)
    normal_cam, confidence = _normal_for_ip(ip, depth, intrinsics, h, w)
    logger.info("push-pull: pixel=(%.1f, %.1f)  3d=%s", x_px, y_px,
                [f"{v:.3f}" for v in ip.position_3d] if ip.position_3d is not None else "N/A")
    if normal_cam is not None:
        logger.info("push-pull: normal=[%.3f, %.3f, %.3f]  confidence=%.2f  "
                    "push direction (negated)=[%.3f, %.3f, %.3f]",
                    normal_cam[0], normal_cam[1], normal_cam[2], confidence,
                    -normal_cam[0], -normal_cam[1], -normal_cam[2])
    else:
        logger.warning("push-pull: surface normal estimation failed")
    return {"query": "push-pull", "ip": ip, "x_px": x_px, "y_px": y_px,
            "label": "push/pull surface", "color": _QUERY_COLORS["push-pull"],
            "normal_cam": normal_cam, "normal_confidence": confidence}


def query_hinge(
    detector: MolmoPointDetector,
    color: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    object_type: str,
    hinge_location: str,
) -> Optional[dict]:
    logger.info("=== HINGE / PIVOT point ===")
    prompt = f"Point to the hinge or pivot point of the {hinge_location}."
    results = detector.get_interaction_points(
        rgb_image=color, depth_frame=depth, camera_intrinsics=intrinsics,
        object_id=f"{object_type}_0", object_type=object_type,
        bounding_box_2d=None, actions={"_hinge"},
        custom_prompts={"_hinge": prompt},
    )
    ip = results.get("_hinge")
    if ip is None:
        logger.warning("hinge: no point returned")
        return None
    h, w = color.shape[:2]
    x_px, y_px = _ip_to_px(ip, h, w)
    logger.info("hinge: pixel=(%.1f, %.1f)  3d=%s", x_px, y_px,
                [f"{v:.3f}" for v in ip.position_3d] if ip.position_3d is not None else "N/A")
    return {"query": "hinge", "ip": ip, "x_px": x_px, "y_px": y_px,
            "label": "hinge", "color": _QUERY_COLORS["hinge"]}


def query_push_aside(
    detector: MolmoPointDetector,
    color: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    object_type: str,
) -> Optional[dict]:
    logger.info("=== PUSH-ASIDE surface ===")
    results = detector.get_interaction_points(
        rgb_image=color, depth_frame=depth, camera_intrinsics=intrinsics,
        object_id=f"{object_type}_0", object_type=object_type,
        bounding_box_2d=None, actions={"push-aside"},
    )
    ip = results.get("push-aside")
    if ip is None:
        logger.warning("push-aside: no point returned")
        return None
    h, w = color.shape[:2]
    x_px, y_px = _ip_to_px(ip, h, w)
    logger.info("push-aside: pixel=(%.1f, %.1f)  3d=%s", x_px, y_px,
                [f"{v:.3f}" for v in ip.position_3d] if ip.position_3d is not None else "N/A")
    return {"query": "push-aside", "ip": ip, "x_px": x_px, "y_px": y_px,
            "label": "push aside", "color": _QUERY_COLORS["push-aside"]}


# ---------------------------------------------------------------------------
# Pivot radius (logged when both pull/push-pull and hinge results are present)
# ---------------------------------------------------------------------------

def _log_pivot_radius(surface_r: Optional[dict], hinge_r: Optional[dict]) -> Optional[float]:
    if surface_r is None or hinge_r is None:
        return None
    s3d = surface_r["ip"].position_3d
    h3d = hinge_r["ip"].position_3d
    if s3d is None or h3d is None:
        logger.info("pivot radius: cannot compute (missing 3D positions)")
        return None
    s, h = np.asarray(s3d, float), np.asarray(h3d, float)
    r3d = float(np.linalg.norm(s - h))
    rxy = float(np.linalg.norm((s - h)[:2]))
    logger.info(
        "pivot radius: 3D=%.3fm  XY=%.3fm  "
        "hinge=[%.3f,%.3f,%.3f]  contact=[%.3f,%.3f,%.3f]",
        r3d, rxy, h[0], h[1], h[2], s[0], s[1], s[2],
    )
    return rxy


# ---------------------------------------------------------------------------
# Per-query visualisation
# ---------------------------------------------------------------------------

def _visualise(
    raw_bgr: np.ndarray,
    r: dict,
    extra_points: Optional[List[dict]] = None,
) -> np.ndarray:
    import cv2
    vis = raw_bgr.copy()
    vis = _draw_point(vis, r["x_px"], r["y_px"], r["label"], r["color"])

    if r.get("normal_cam") is not None:
        conf = r.get("normal_confidence", 0.0)
        vis = _draw_normal_arrow(vis, r["x_px"], r["y_px"], r["normal_cam"],
                                 f"normal (conf={conf:.2f})")

    for ep in (extra_points or []):
        if ep is None:
            continue
        vis = _draw_point(vis, ep["x_px"], ep["y_px"], ep["label"], ep["color"])
        cv2.line(vis,
                 (int(round(r["x_px"])), int(round(r["y_px"]))),
                 (int(round(ep["x_px"])), int(round(ep["y_px"]))),
                 (200, 200, 0), 2, cv2.LINE_AA)

    return vis


def _build_composite(
    color_rgb: np.ndarray,
    results: List[Optional[dict]],
    depth: np.ndarray,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    import cv2
    vis = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    for r in results:
        if r is None:
            continue
        vis = _draw_point(vis, r["x_px"], r["y_px"], r["label"], r["color"])
        if r.get("normal_cam") is not None:
            conf = r.get("normal_confidence", 0.0)
            vis = _draw_normal_arrow(vis, r["x_px"], r["y_px"], r["normal_cam"],
                                     f"normal {conf:.2f}")

    # Depth inset bottom-right
    inset_h, inset_w = img_h // 4, img_w // 4
    d_u8 = (np.clip(depth / 2.0, 0, 1) * 255).astype(np.uint8)
    d_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
    vis[img_h - inset_h:, img_w - inset_w:] = cv2.resize(d_color, (inset_w, inset_h))
    cv2.putText(vis, "depth", (img_w - inset_w + 4, img_h - inset_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return vis


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Molmo pointing test for primitive parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available queries: {', '.join(ALL_QUERIES)}",
    )
    p.add_argument("--rgb-path", default=None,
                   help="Saved RGB image (.png/.jpg). Omit to capture from RealSense.")
    p.add_argument("--depth-path", default=None,
                   help="Saved depth .npy file (required with --rgb-path).")
    p.add_argument("--object-type", default="object",
                   help="Object class label, e.g. 'cup', 'drawer'.")
    p.add_argument("--queries", nargs="+", default=ALL_QUERIES,
                   choices=ALL_QUERIES, metavar="QUERY",
                   help=f"Which queries to run. Choices: {', '.join(ALL_QUERIES)}. "
                        "Default: all.")
    p.add_argument("--hinge-location", default=None,
                   help="Description of the hinge for the 'hinge' query, e.g. "
                        "'left edge of the drawer'. Required when 'hinge' is in --queries.")
    p.add_argument("--output-dir", default=None,
                   help="Output directory for this run. "
                        f"Defaults to outputs/{_SCRIPT_NAME}/runs/YYYYMMDD_HHMMSS.")
    p.add_argument("--checkpoint", default="allenai/Molmo2-4B",
                   help="Molmo2 HuggingFace checkpoint.")
    p.add_argument("--warmup-frames", type=int, default=30,
                   help="RealSense frames to discard before capture (auto-exposure settling).")
    p.add_argument("--fx", type=float, default=None, help="Camera fx override (--rgb-path only).")
    p.add_argument("--fy", type=float, default=None, help="Camera fy override (--rgb-path only).")
    p.add_argument("--cx", type=float, default=None, help="Camera cx override (--rgb-path only).")
    p.add_argument("--cy", type=float, default=None, help="Camera cy override (--rgb-path only).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    configure_logging()
    output_dir = Path(args.output_dir) if args.output_dir else _DEFAULT_OUTDIR
    _init_run_dir(output_dir)

    if "hinge" in args.queries and not args.hinge_location:
        logger.warning(
            "'hinge' query selected but --hinge-location not provided. "
            "Will use 'hinge of the %s' as default.", args.object_type
        )

    # ---- Image source ----
    if args.rgb_path:
        if not args.depth_path:
            logger.error("--depth-path required with --rgb-path")
            sys.exit(1)
        color, depth, intrinsics = _load_from_disk(args.rgb_path, args.depth_path)
        if args.fx is not None:
            intrinsics = CameraIntrinsics(
                fx=args.fx, fy=args.fy or args.fx,
                cx=args.cx or intrinsics.cx, cy=args.cy or intrinsics.cy,
                width=intrinsics.width, height=intrinsics.height,
            )
    else:
        if not REALSENSE_AVAILABLE:
            logger.error(
                "pyrealsense2 not installed. Pass --rgb-path / --depth-path "
                "to use saved images, or install pyrealsense2 and connect a camera."
            )
            sys.exit(1)
        color, depth, intrinsics = _capture_realsense(warmup_frames=args.warmup_frames)

    img_h, img_w = color.shape[:2]

    import cv2
    raw_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    _save(raw_bgr, output_dir, "00_raw_rgb")

    # Save depth as .npy (for re-running with --rgb-path / --depth-path)
    np.save(str(output_dir / "00_depth.npy"), depth)
    logger.info("Saved %s", output_dir / "00_depth.npy")

    # Save colourised depth for visual inspection
    d_u8 = (np.clip(depth / 2.0, 0, 1) * 255).astype(np.uint8)
    _save(cv2.applyColorMap(d_u8, cv2.COLORMAP_JET), output_dir, "00_depth_vis")

    # ---- Load Molmo ----
    detector = MolmoPointDetector(checkpoint=args.checkpoint)
    detector.load()

    # ---- Run selected queries ----
    object_type = args.object_type
    all_results: List[Optional[dict]] = []
    r_pull_or_pushpull: Optional[dict] = None  # for pivot radius calculation
    r_hinge: Optional[dict] = None

    for query in args.queries:
        r: Optional[dict] = None

        if query == "grasp":
            r = query_grasp(detector, color, depth, intrinsics, object_type)
            if r:
                _save(_visualise(raw_bgr, r), output_dir, "01_grasp")

        elif query == "push":
            r = query_push(detector, color, depth, intrinsics, object_type)
            if r:
                _save(_visualise(raw_bgr, r), output_dir, "02_push_surface")

        elif query == "pull":
            r = query_pull(detector, color, depth, intrinsics, object_type)
            if r:
                _save(_visualise(raw_bgr, r), output_dir, "03_pull_surface")
                r_pull_or_pushpull = r

        elif query == "push-pull":
            r = query_push_pull(detector, color, depth, intrinsics, object_type)
            if r:
                _save(_visualise(raw_bgr, r), output_dir, "04_push_pull_surface")
                r_pull_or_pushpull = r

        elif query == "hinge":
            hinge_desc = args.hinge_location or f"hinge of the {object_type}"
            r = query_hinge(detector, color, depth, intrinsics, object_type, hinge_desc)
            r_hinge = r
            if r:
                _save(_visualise(raw_bgr, r), output_dir, "05_hinge")

        elif query == "push-aside":
            r = query_push_aside(detector, color, depth, intrinsics, object_type)
            if r:
                _save(_visualise(raw_bgr, r), output_dir, "06_push_aside")

        all_results.append(r)

    # ---- Pivot lever-arm composite (pull/push-pull + hinge) ----
    pivot_radius: Optional[float] = None
    if r_pull_or_pushpull and r_hinge:
        pivot_radius = _log_pivot_radius(r_pull_or_pushpull, r_hinge)
        vis = _visualise(raw_bgr, r_pull_or_pushpull, extra_points=[r_hinge])
        _save(vis, output_dir, "07_pivot_lever_arm")

    # ---- Composite: all results on one image ----
    composite = _build_composite(color, all_results, depth, img_h, img_w)
    _save(composite, output_dir, "99_composite")

    # ---- Summary ----
    succeeded = [r["query"] for r in all_results if r is not None]
    failed = [q for q, r in zip(args.queries, all_results) if r is None]

    summary_lines = [
        "=== Molmo pointing summary ===",
        f"  Object      : {object_type}",
        f"  Image size  : {img_w}x{img_h}",
        f"  Intrinsics  : fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} "
        f"cx={intrinsics.cx:.1f} cy={intrinsics.cy:.1f}",
        f"  Queries run : {', '.join(args.queries)}",
        f"  Succeeded   : {', '.join(succeeded) if succeeded else 'none'}",
    ]
    if failed:
        summary_lines.append(f"  Failed      : {', '.join(failed)}")

    for r in all_results:
        if r is None:
            continue
        pos3d = r["ip"].position_3d
        pos3d_str = (f"[{pos3d[0]:.3f}, {pos3d[1]:.3f}, {pos3d[2]:.3f}]m"
                     if pos3d is not None else "N/A")
        normal_str = ""
        if r.get("normal_cam") is not None:
            n = r["normal_cam"]
            normal_str = (f"  normal=[{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}]"
                          f"  push_dir=[{-n[0]:.3f}, {-n[1]:.3f}, {-n[2]:.3f}]"
                          f"  conf={r.get('normal_confidence', 0):.2f}")
        summary_lines.append(
            f"  [{r['query']:12s}]  pixel=({r['x_px']:.1f}, {r['y_px']:.1f})"
            f"  3d={pos3d_str}{normal_str}"
        )

    if pivot_radius is not None:
        summary_lines.append(f"  Pivot radius: {pivot_radius:.3f}m (XY-plane)")

    summary_lines.append(f"  Output      : {output_dir}")

    summary_text = "\n".join(summary_lines) + "\n"
    (output_dir / "summary.txt").write_text(summary_text)
    logger.info("Saved %s", output_dir / "summary.txt")

    print("\n" + summary_text)


if __name__ == "__main__":
    main()
