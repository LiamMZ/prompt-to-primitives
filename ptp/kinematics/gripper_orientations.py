"""Canonical gripper preset orientations for the xArm.

Single source of truth for the quaternions used by the planner, grasp solver,
and test scripts.  All values are xyzw unit quaternions in the robot base frame.
"""

from __future__ import annotations

from typing import List

# xyzw quaternions measured empirically on the physical xArm.
# top_down: gripper Z-axis pointing straight down  (~roll=-178°)
# side:     gripper Z-axis pointing horizontally   (~pitch=-86°)
PRESET_QUATERNIONS: dict[str, List[float]] = {
    "top_down": [-0.9983, 0.0314,  0.0438,  0.0223],
    "side":     [-0.6894, 0.0305, -0.7237,  0.0033],
}

PRESET_DESCRIPTIONS: dict[str, str] = {
    "top_down": "Gripper pointing straight down (default pick/place)",
    "side":     "Gripper pointing horizontally (handles, drawers, doors)",
}


def preset_quat(name: str) -> List[float]:
    """Return the xyzw quaternion for a named preset, defaulting to top_down."""
    return list(PRESET_QUATERNIONS.get(name, PRESET_QUATERNIONS["top_down"]))
