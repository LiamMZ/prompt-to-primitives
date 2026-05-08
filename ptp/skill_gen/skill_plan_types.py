"""
Primitive skill plan types for translating symbolic actions into executable robot primitives.

These structures intentionally mirror the CuRobo/xArm primitives exposed in
`xarm_curobo_interface.py` while remaining serialization-friendly for prompt I/O.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


def _json_default(value: Any) -> Any:
    """Best-effort converter to keep registry hashing stable."""
    if isinstance(value, (set, tuple)):
        return list(value)
    try:
        return dict(value)
    except Exception:
        return str(value)


def compute_registry_hash(registry: Dict[str, Any]) -> str:
    """
    Compute a deterministic hash of the registry/world slice for caching.

    Args:
        registry: World registry dictionary (JSON-serializable)

    Returns:
        Hex digest string
    """
    payload = json.dumps(registry, sort_keys=True, default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _vector_validator(expected_len: int) -> Callable[[Any], Optional[str]]:
    """Create a validator that checks for a numeric list/tuple of specific length."""
    def _validate(value: Any) -> Optional[str]:
        if not isinstance(value, (list, tuple)):
            return f"expected list/tuple length {expected_len}, got {type(value).__name__}"
        if len(value) != expected_len:
            return f"expected length {expected_len}, got {len(value)}"
        return None
    return _validate


def _positive_number_validator(field_name: str) -> Callable[[Any], Optional[str]]:
    """Validate that a value is a positive number."""
    def _validate(value: Any) -> Optional[str]:
        try:
            if float(value) <= 0:
                return f"{field_name} must be > 0"
        except Exception:
            return f"{field_name} must be numeric"
        return None
    return _validate


@dataclass
class PrimitiveSchema:
    """Schema definition for a single primitive."""

    name: str
    required_params: Tuple[str, ...] = field(default_factory=tuple)
    optional_params: Tuple[str, ...] = field(default_factory=tuple)
    allowed_frames: Tuple[str, ...] = ("base", "camera")
    description: str = ""
    default_frame: str = "base"
    param_validators: Dict[str, Callable[[Any], Optional[str]]] = field(default_factory=dict)

    def validate(self, call: "PrimitiveCall") -> List[str]:
        """
        Validate a primitive call against this schema.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: List[str] = []

        # Required parameters
        for param in self.required_params:
            if param not in call.parameters:
                errors.append(f"missing required parameter '{param}' for {self.name}")

        # Unknown parameters
        allowed = set(self.required_params) | set(self.optional_params)
        for param in call.parameters:
            if param not in allowed:
                errors.append(f"unexpected parameter '{param}' for {self.name}")

        # Frame validation
        if self.allowed_frames and call.frame not in self.allowed_frames:
            errors.append(
                f"frame '{call.frame}' not allowed for {self.name}; "
                f"expected one of {', '.join(self.allowed_frames)}"
            )

        # Run parameter-level validators
        for param_name, validator in self.param_validators.items():
            if param_name in call.parameters:
                msg = validator(call.parameters[param_name])
                if msg:
                    errors.append(f"{param_name}: {msg}")

        return errors


@dataclass
class PrimitiveCall:
    """
    Represents a single primitive invocation destined for xArm/CuRobo helpers.
    """

    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    frame: str = "base"  # Logical frame the parameters are expressed in
    references: Dict[str, str] = field(default_factory=dict)  # e.g., object_id, interaction_point_id
    metadata: Dict[str, Any] = field(default_factory=dict)  # retries, speed profiles, etc.

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "name": self.name,
            "frame": self.frame,
            "parameters": self.parameters,
            "references": self.references,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrimitiveCall":
        """Deserialize from a JSON-like dictionary."""
        params = dict(data.get("parameters") or {})
        frame = data.get("frame") or params.pop("frame", "base")
        return cls(
            name=data.get("name", ""),
            frame=frame,
            parameters=params,
            references=data.get("references") or {},
            metadata=data.get("metadata") or {},
        )

    def validate(self, schema: PrimitiveSchema) -> List[str]:
        """Validate this call against a schema."""
        return schema.validate(self)


@dataclass
class SkillPlanDiagnostics:
    """Supplemental diagnostics emitted by the decomposer."""

    assumptions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    freshness_notes: List[str] = field(default_factory=list)
    freshness: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    interaction_points: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assumptions": self.assumptions,
            "warnings": self.warnings,
            "freshness_notes": self.freshness_notes,
            "freshness": self.freshness,
            "rationale": self.rationale,
            "interaction_points": self.interaction_points,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillPlanDiagnostics":
        return cls(
            assumptions=data.get("assumptions") or [],
            warnings=data.get("warnings") or [],
            freshness_notes=data.get("freshness_notes") or [],
            freshness=data.get("freshness") or {},
            rationale=data.get("rationale") or "",
            interaction_points=data.get("interaction_points") or [],
        )


@dataclass
class SkillPlan:
    """Container for a full primitive sequence."""

    action_name: str
    primitives: List[PrimitiveCall] = field(default_factory=list)
    diagnostics: SkillPlanDiagnostics = field(default_factory=SkillPlanDiagnostics)
    registry_hash: Optional[str] = None
    source_snapshot_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_name": self.action_name,
            "primitives": [p.to_dict() for p in self.primitives],
            "diagnostics": self.diagnostics.to_dict(),
            "registry_hash": self.registry_hash,
            "source_snapshot_id": self.source_snapshot_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillPlan":
        primitives = [
            PrimitiveCall.from_dict(p) for p in data.get("primitives", [])
        ]
        diagnostics = SkillPlanDiagnostics.from_dict(data.get("diagnostics") or {})
        return cls(
            action_name=data.get("action_name", ""),
            primitives=primitives,
            diagnostics=diagnostics,
            registry_hash=data.get("registry_hash"),
            source_snapshot_id=data.get("source_snapshot_id"),
        )

    def validate(self, schema_map: Dict[str, PrimitiveSchema]) -> List[str]:
        """
        Validate all primitives using provided schemas.

        Args:
            schema_map: Mapping from primitive name to PrimitiveSchema

        Returns:
            List of validation errors
        """
        errors: List[str] = []
        for idx, primitive in enumerate(self.primitives):
            schema = schema_map.get(primitive.name)
            if not schema:
                errors.append(f"[{idx}] unknown primitive '{primitive.name}'")
                continue
            for msg in primitive.validate(schema):
                errors.append(f"[{idx}] {msg}")
        return errors


PRIMITIVE_LIBRARY: Dict[str, PrimitiveSchema] = {
    # ------------------------------------------------------------------
    # move_gripper_to_pose  — move EEF to a target pose
    #   Distinguished from navigate_to_pose (base motion, added later).
    # ------------------------------------------------------------------
    "move_gripper_to_pose": PrimitiveSchema(
        name="move_gripper_to_pose",
        optional_params=("target_pixel_yx", "target_position", "pivot_point",
                         "preset_orientation", "target_orientation", "is_place",
                         "point_label", "is_top_down_grasp", "is_side_grasp",
                         "speed_factor", "execute", "depth_offset_m"),
        allowed_frames=("base", "camera"),
        description=(
            "Move the gripper end-effector to a target pose. "
            "target_pixel_yx: normalized [y, x] (0-1000) pixel for back-projection. "
            "target_position: [x, y, z] in base frame (set by executor after back-projection). "
            "preset_orientation: 'top_down' or 'side'. "
            "is_place: True when placing (adds z clearance). "
            "speed_factor: execution speed multiplier (0.1–2.0). "
            "execute: if False, plan but do not execute. "
            "Distinguished from navigate_to_pose (base/mobile platform motion)."
        ),
    ),
    # ------------------------------------------------------------------
    # push_pull  — constrained EEF motion along/about a surface
    # ------------------------------------------------------------------
    "push_pull": PrimitiveSchema(
        name="push_pull",
        required_params=("surface_label",),
        optional_params=("force_direction", "is_button", "has_pivot", "hinge_location"),
        allowed_frames=("base", "camera"),
        description=(
            "Push or pull relative to a Surface Map label. "
            "surface_label: name of the surface or object to interact with. "
            "force_direction: 'perpendicular' (push into surface) or 'parallel' (slide). "
            "is_button: True for momentary press-and-retract. "
            "has_pivot: True for revolute (door/drawer) articulation. "
            "hinge_location: surface boundary label for the hinge axis."
        ),
    ),
    # ------------------------------------------------------------------
    # open_gripper / close_gripper / retract_gripper
    # ------------------------------------------------------------------
    "open_gripper": PrimitiveSchema(
        name="open_gripper",
        optional_params=("wait", "timeout"),
        allowed_frames=("base", "camera"),
        description="Open gripper to release held objects.",
        param_validators={
            "timeout": _positive_number_validator("timeout"),
        },
    ),
    "close_gripper": PrimitiveSchema(
        name="close_gripper",
        optional_params=("wait", "timeout", "simple_close"),
        allowed_frames=("base", "camera"),
        description="Close gripper to acquire a grasp.",
        param_validators={
            "timeout": _positive_number_validator("timeout"),
        },
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
    # twist  — wrist rotation about the end-effector Z axis
    # ------------------------------------------------------------------
    "twist": PrimitiveSchema(
        name="twist",
        optional_params=("direction", "rotation_angle_deg", "speed_factor", "timeout"),
        allowed_frames=("base", "camera"),
        description=(
            "Rotate the wrist/final joint.  "
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
