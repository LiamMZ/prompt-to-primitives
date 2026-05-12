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
    # push — move EEF into a surface (along -normal)
    # ------------------------------------------------------------------
    "push": PrimitiveSchema(
        name="push",
        required_params=("surface_label",),
        optional_params=(
            "is_button",
            "action_goal",
            "speed_factor",
            "execute",
            "metadata",
            "object_id",
        ),
        allowed_frames=("base", "camera"),
        description=(
            "Move the gripper into a surface along the negative surface normal. "
            "surface_label: target object_id or surface name. "
            "is_button: True for momentary press-and-retract. "
            "action_goal: high-level goal string passed to Molmo (e.g. 'press', 'push'). "
            "Executor injects surface_normal_base via metadata."
        ),
    ),

    # ------------------------------------------------------------------
    # pull — move EEF away from a surface (along +normal)
    # ------------------------------------------------------------------
    "pull": PrimitiveSchema(
        name="pull",
        required_params=("surface_label",),
        optional_params=(
            "has_pivot",
            "hinge_axis",
            "action_goal",
            "speed_factor",
            "execute",
            "metadata",
            "object_id",
        ),
        allowed_frames=("base", "camera"),
        description=(
            "Move the gripper away from a surface along the positive surface normal. "
            "surface_label: target object_id or surface name. "
            "has_pivot: True for revolute articulation (door/drawer) — Molmo locates the hinge automatically. "
            "action_goal: high-level goal string passed to Molmo (e.g. 'open', 'pull'). "
            "Executor injects surface_normal_base and pivot metadata."
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
        optional_params=("direction", "turn_amount", "speed_factor", "timeout"),
        allowed_frames=("base", "camera"),
        description=(
            "Rotate the wrist/final joint. "
            "direction: 'clockwise' or 'counterclockwise'. "
            "turn_amount: 'quarter_turn' (90°), 'half_turn' (180°), "
            "'three_quarter_turn' (270°), or 'full_turn' (360°). Default: full_turn."
        ),
        param_validators={
            "speed_factor": _positive_number_validator("speed_factor"),
            "timeout": _positive_number_validator("timeout"),
        },
    ),
}
