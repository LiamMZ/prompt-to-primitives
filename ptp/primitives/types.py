"""Core data types for primitive skill plans.

These structures are serialization-friendly (JSON-roundtrippable) and intentionally
decoupled from any specific robot backend or planning system.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, tuple)):
        return list(value)
    try:
        return dict(value)
    except Exception:
        return str(value)


def compute_registry_hash(registry: Dict[str, Any]) -> str:
    """Deterministic SHA-256 hash of a registry/world slice for cache keying."""
    payload = json.dumps(registry, sort_keys=True, default=_json_default)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Validators (used by PrimitiveSchema)
# ---------------------------------------------------------------------------

def _vector_validator(expected_len: int) -> Callable[[Any], Optional[str]]:
    def _validate(value: Any) -> Optional[str]:
        if not isinstance(value, (list, tuple)):
            return f"expected list/tuple of length {expected_len}, got {type(value).__name__}"
        if len(value) != expected_len:
            return f"expected length {expected_len}, got {len(value)}"
        return None
    return _validate


def _positive_number_validator(field_name: str) -> Callable[[Any], Optional[str]]:
    def _validate(value: Any) -> Optional[str]:
        try:
            if float(value) <= 0:
                return f"{field_name} must be > 0"
        except Exception:
            return f"{field_name} must be numeric"
        return None
    return _validate


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class PrimitiveSchema:
    """Schema definition for a single atomic manipulation primitive."""

    name: str
    required_params: Tuple[str, ...] = field(default_factory=tuple)
    optional_params: Tuple[str, ...] = field(default_factory=tuple)
    allowed_frames: Tuple[str, ...] = ("base", "camera")
    description: str = ""
    default_frame: str = "base"
    param_validators: Dict[str, Callable[[Any], Optional[str]]] = field(default_factory=dict)

    def validate(self, call: "PrimitiveCall") -> List[str]:
        """Return a list of validation errors (empty = valid)."""
        errors: List[str] = []
        for param in self.required_params:
            if param not in call.parameters:
                errors.append(f"missing required parameter '{param}' for {self.name}")
        allowed = set(self.required_params) | set(self.optional_params)
        for param in call.parameters:
            if param not in allowed:
                errors.append(f"unexpected parameter '{param}' for {self.name}")
        if self.allowed_frames and call.frame not in self.allowed_frames:
            errors.append(
                f"frame '{call.frame}' not allowed for {self.name}; "
                f"expected one of {', '.join(self.allowed_frames)}"
            )
        for param_name, validator in self.param_validators.items():
            if param_name in call.parameters:
                msg = validator(call.parameters[param_name])
                if msg:
                    errors.append(f"{param_name}: {msg}")
        return errors


# ---------------------------------------------------------------------------
# Primitive call
# ---------------------------------------------------------------------------

@dataclass
class PrimitiveCall:
    """A single primitive invocation with typed parameters."""

    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    frame: str = "base"
    references: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "frame": self.frame,
            "parameters": self.parameters,
            "references": self.references,
            "metadata": {k: v for k, v in self.metadata.items() if not isinstance(v, (bytes, bytearray))},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrimitiveCall":
        params = dict(data.get("parameters") or {})
        frame  = data.get("frame") or params.pop("frame", "base")
        return cls(
            name=data.get("name", ""),
            frame=frame,
            parameters=params,
            references=data.get("references") or {},
            metadata=data.get("metadata") or {},
        )

    def validate(self, schema: PrimitiveSchema) -> List[str]:
        return schema.validate(self)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Skill plan
# ---------------------------------------------------------------------------

@dataclass
class SkillPlan:
    """An ordered sequence of PrimitiveCalls for one symbolic action."""

    action_name: str
    primitives: List[PrimitiveCall] = field(default_factory=list)
    diagnostics: SkillPlanDiagnostics = field(default_factory=SkillPlanDiagnostics)
    registry_hash: Optional[str] = None
    source_snapshot_id: Optional[str] = None
    raw_llm_response: Optional[str] = None
    high_level_action: str = ""          # e.g. "open", "switch_on" from TaskParser
    target_object_id: Optional[str] = None  # resolved object_id from TaskParser

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_name": self.action_name,
            "high_level_action": self.high_level_action,
            "target_object_id": self.target_object_id,
            "primitives": [p.to_dict() for p in self.primitives],
            "diagnostics": self.diagnostics.to_dict(),
            "registry_hash": self.registry_hash,
            "source_snapshot_id": self.source_snapshot_id,
            "raw_llm_response": self.raw_llm_response,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillPlan":
        return cls(
            action_name=data.get("action_name", ""),
            high_level_action=data.get("high_level_action", ""),
            target_object_id=data.get("target_object_id"),
            primitives=[PrimitiveCall.from_dict(p) for p in data.get("primitives", [])],
            diagnostics=SkillPlanDiagnostics.from_dict(data.get("diagnostics") or {}),
            registry_hash=data.get("registry_hash"),
            source_snapshot_id=data.get("source_snapshot_id"),
        )

    def validate(self, schema_map: Dict[str, PrimitiveSchema]) -> List[str]:
        """Validate all primitives against the provided schema map."""
        errors: List[str] = []
        for idx, prim in enumerate(self.primitives):
            schema = schema_map.get(prim.name)
            if not schema:
                errors.append(f"[{idx}] unknown primitive '{prim.name}'")
                continue
            for msg in prim.validate(schema):
                errors.append(f"[{idx}] {msg}")
        return errors
