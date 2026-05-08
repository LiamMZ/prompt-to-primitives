"""Helpers for loading perception-pool snapshot artifacts.

Snapshots are directories under <perception_pool_dir>/snapshots/<id>/
containing color images, depth .npz files, camera intrinsics JSON, and
robot-state JSON.  An index.json at the pool root maps object IDs to the
snapshot(s) that observed them.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SnapshotArtifacts:
    """All per-snapshot data consumed by the decomposer and executor."""

    snapshot_id: Optional[str]
    meta: Optional[Dict[str, Any]] = None
    color_bytes: Optional[bytes] = None
    depth: Optional[np.ndarray] = None
    intrinsics: Optional[Any] = None        # SimpleNamespace with fx/fy/cx/cy/…
    color_shape: Optional[Tuple[int, int]] = None  # (height, width)
    robot_state: Optional[Dict[str, Any]] = None


@dataclass
class SnapshotCache:
    """Lightweight in-process cache to avoid re-reading disk on every call."""

    index: Optional[Dict[str, Any]] = None
    artifacts: Dict[str, SnapshotArtifacts] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_snapshot_artifacts(
    world_state: Dict[str, Any],
    perception_pool_dir: Path,
    cache: Optional[SnapshotCache] = None,
    snapshot_id: Optional[str] = None,
) -> SnapshotArtifacts:
    """Load all artifacts for the specified snapshot (or latest from world_state)."""
    snapshot_id = snapshot_id or world_state.get("last_snapshot_id")
    if not snapshot_id:
        return SnapshotArtifacts(snapshot_id=None)

    if cache and snapshot_id in cache.artifacts:
        return cache.artifacts[snapshot_id]

    index = _resolve_index(world_state, perception_pool_dir, cache)
    meta  = (index.get("snapshots") or {}).get(snapshot_id) if index else None
    if meta is None:
        empty = SnapshotArtifacts(snapshot_id=None)
        if cache:
            cache.artifacts[snapshot_id] = empty
        return empty

    files = meta.get("files") or {}
    artifacts = SnapshotArtifacts(snapshot_id=snapshot_id, meta=meta)
    artifacts.color_bytes = _read_bytes(perception_pool_dir, files.get("color"))
    artifacts.depth       = _read_depth(perception_pool_dir, files.get("depth_npz"))
    artifacts.intrinsics  = _read_intrinsics(perception_pool_dir, files.get("intrinsics"))
    artifacts.robot_state = _read_json(perception_pool_dir, files.get("robot_state"))

    if artifacts.color_bytes:
        try:
            from PIL import Image
            with Image.open(io.BytesIO(artifacts.color_bytes)) as img:
                artifacts.color_shape = (img.height, img.width)
        except Exception:
            pass

    if cache:
        cache.artifacts[snapshot_id] = artifacts
    return artifacts


def latest_snapshot_for_object_ids(
    world_state: Dict[str, Any],
    perception_pool_dir: Path,
    object_ids: List[str],
    cache: Optional[SnapshotCache] = None,
) -> Optional[str]:
    """Return the newest snapshot ID that observed any of the given object IDs."""
    if not object_ids:
        return None
    index = _resolve_index(world_state, perception_pool_dir, cache)
    if not index:
        return None

    objects_map   = index.get("objects") or {}
    snapshots_meta = index.get("snapshots") or {}

    def _ts(sid: str) -> str:
        m = snapshots_meta.get(sid) or {}
        return m.get("recorded_at") or m.get("captured_at") or ""

    candidates: List[str] = []
    for oid in object_ids:
        snaps = objects_map.get(oid)
        if snaps:
            candidates.append(snaps[-1])

    return max(candidates, key=_ts) if candidates else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_index(
    world_state: Dict[str, Any],
    perception_pool_dir: Path,
    cache: Optional[SnapshotCache],
) -> Optional[Dict[str, Any]]:
    if cache and cache.index:
        return cache.index
    index = world_state.get("snapshot_index")
    if not index:
        path = Path(perception_pool_dir) / "index.json"
        if path.exists():
            try:
                index = json.loads(path.read_text())
            except Exception:
                index = None
    if cache and index:
        cache.index = index
    return index


def _read_bytes(pool: Path, rel: Optional[str]) -> Optional[bytes]:
    if not rel:
        return None
    p = Path(pool) / rel
    return p.read_bytes() if p.exists() else None


def _read_depth(pool: Path, rel: Optional[str]) -> Optional[np.ndarray]:
    if not rel:
        return None
    p = Path(pool) / rel
    if not p.exists():
        return None
    try:
        with np.load(p) as d:
            return d.get("depth_m")
    except Exception:
        return None


def _read_intrinsics(pool: Path, rel: Optional[str]) -> Optional[Any]:
    if not rel:
        return None
    p = Path(pool) / rel
    if not p.exists():
        return None
    try:
        return SimpleNamespace(**json.loads(p.read_text()))
    except Exception:
        return None


def _read_json(pool: Path, rel: Optional[str]) -> Optional[Dict[str, Any]]:
    if not rel:
        return None
    p = Path(pool) / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None
