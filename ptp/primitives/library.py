"""Primitive library — the vocabulary of atomic manipulation primitives.

Each entry is a PrimitiveSchema that the LLM decomposer and executor reference
to validate and execute plans.  New primitives can be added here without
touching any other module.
"""

from __future__ import annotations

from typing import Dict

from ptp.primitives.types import PrimitiveSchema, _positive_number_validator

PRIMITIVE_LIBRARY: Dict[str, PrimitiveSchema] = {
    # ------------------------------------------------------------------
    # move_gripper_to_pose — move EEF to a target pose
    # ------------------------------------------------------------------
    "move_gripper_to_pose": PrimitiveSchema(
        name="move_gripper_to_pose",
        optional_params=(
            "target_pixel_yx",
            "target_position",
            "pivot_point",
            "preset_orientation",
            "target_orientation",
            "is_place",
            "point_label",
            "is_top_down_grasp",
            "is_side_grasp",
            "speed_factor",
            "execute",
            "depth_offset_m",
        ),
        allowed_frames=("base", "camera"),
        description=(
            "Move the gripper end-effector to a target pose. "
            "target_pixel_yx: normalised [y, x] (0-1000) pixel for back-projection. "
            "target_position: [x, y, z] in base frame (resolved by executor). "
            "preset_orientation: 'top_down' or 'side'. "
            "is_place: True when placing (adds z clearance). "
            "speed_factor: execution speed multiplier (0.1–2.0). "
            "execute: if False, plan but do not execute."
        ),
    ),

    # ------------------------------------------------------------------
    # push_pull — constrained EEF motion along/about a surface
    # ------------------------------------------------------------------
    "push_pull": PrimitiveSchema(
        name="push_pull",
        required_params=("surface_label",),
        optional_params=(
            "force_direction",
            "is_button",
            "has_pivot",
            "hinge_location",
            "speed_factor",
            "execute",
            "metadata",
            "object_id",
        ),
        allowed_frames=("base", "camera"),
        description=(
            "Push or pull relative to a named surface or object. "
            "surface_label: name of the surface/object to interact with. "
            "force_direction: 'perpendicular' (into surface) or 'parallel' (slide). "
            "is_button: True for momentary press-and-retract. "
            "has_pivot: True for revolute (door/drawer) articulation. "
            "hinge_location: surface boundary label for the hinge axis. "
            "metadata: executor-injected dict containing surface_normal_base (set by "
            "Molmo surface grounding), surface_normal_confidence, surface_pixel_yx; "
            "and for pivot motions: pivot_point_base (Molmo-grounded hinge 3D position "
            "in base frame) and pivot_radius_m (lever-arm distance from hinge to contact point)."
        ),
    ),

    # ------------------------------------------------------------------
    # Gripper control
    # ------------------------------------------------------------------
    "open_gripper": PrimitiveSchema(
        name="open_gripper",
        optional_params=("wait", "timeout"),
        allowed_frames=("base", "camera"),
        description="Open gripper to release held objects.",
        param_validators={"timeout": _positive_number_validator("timeout")},
    ),
    "close_gripper": PrimitiveSchema(
        name="close_gripper",
        optional_params=("wait", "timeout", "simple_close"),
        allowed_frames=("base", "camera"),
        description="Close gripper to acquire a grasp.",
        param_validators={"timeout": _positive_number_validator("timeout")},
    ),
    "retract_gripper": PrimitiveSchema(
        name="retract_gripper",
        optional_params=("distance", "speed_factor", "execute"),
        allowed_frames=("base", "camera"),
        description="Return the arm to its home/neutral configuration.",
        param_validators={
            "distance": _positive_number_validator("distance"),
            "speed_factor": _positive_number_validator("speed_factor"),
        },
    ),

    # ------------------------------------------------------------------
    # twist — wrist rotation about EEF Z axis
    # ------------------------------------------------------------------
    "twist": PrimitiveSchema(
        name="twist",
        optional_params=("direction", "rotation_angle_deg", "speed_factor", "timeout"),
        allowed_frames=("base", "camera"),
        description=(
            "Rotate the wrist/final joint. "
            "direction: 'clockwise' or 'counterclockwise'. "
            "rotation_angle_deg: degrees to rotate (default 90)."
        ),
        param_validators={
            "rotation_angle_deg": _positive_number_validator("rotation_angle_deg"),
            "speed_factor": _positive_number_validator("speed_factor"),
            "timeout": _positive_number_validator("timeout"),
        },
    ),
}
