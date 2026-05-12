"""Thread-safe registry for detected objects and their interaction points."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np


@dataclass
class InteractionPoint:
    """A Molmo-grounded interaction point for a specific robot action."""

    position_2d: List[int]                      # [y, x] in 0-1000 normalised scale
    position_3d: Optional[np.ndarray] = None    # [x, y, z] in metres (world frame)
    alternative_points: List[Dict[str, Any]] = field(default_factory=list)
    approach_orientation: Optional[str] = None  # "top_down", "side", or encoded vector
    approach_vector: Optional[np.ndarray] = None
    input_image_bytes: Optional[bytes] = None   # PNG sent to Molmo (for debug output)


@dataclass
class DetectedObject:
    """A single detected object with geometry, affordances, and interaction points."""

    object_type: str
    object_id: str
    affordances: Set[str] = field(default_factory=set)
    interaction_points: Dict[str, InteractionPoint] = field(default_factory=dict)
    position_2d: Optional[List[int]] = None      # [y, x] in 0-1000 scale
    position_3d: Optional[np.ndarray] = None     # [x, y, z] in metres
    bounding_box_2d: Optional[List[int]] = None  # [y1, x1, y2, x2]
    timestamp: float = field(default_factory=time.time)
    clearance_profile: Optional[Any] = None      # ClearanceProfile when available


class DetectedObjectRegistry:
    """Thread-safe store for detected objects.

    Example:
        >>> registry = DetectedObjectRegistry()
        >>> registry.add_object(obj)
        >>> registry.get_object("red_cup_1")
    """

    def __init__(self) -> None:
        self._objects: Dict[str, DetectedObject] = {}
        self._predicates: Set[str] = set()
        self._lock = threading.RLock()
        self.contact_graph: Optional[Any] = None
        self.occlusion_map: Optional[Any] = None

    # ------------------------------------------------------------------
    # Object CRUD
    # ------------------------------------------------------------------

    def add_object(self, obj: DetectedObject) -> None:
        with self._lock:
            self._objects[obj.object_id] = obj

    def get_object(self, object_id: str) -> Optional[DetectedObject]:
        with self._lock:
            return self._objects.get(object_id)

    def get_all_objects(self) -> List[DetectedObject]:
        with self._lock:
            return list(self._objects.values())

    def get_objects_by_type(self, object_type: str) -> List[DetectedObject]:
        with self._lock:
            return [o for o in self._objects.values() if o.object_type == object_type]

    def get_objects_with_affordance(self, affordance: str) -> List[DetectedObject]:
        with self._lock:
            return [o for o in self._objects.values() if affordance in o.affordances]

    def update_object(self, object_id: str, obj: DetectedObject) -> bool:
        with self._lock:
            if object_id in self._objects:
                self._objects[object_id] = obj
                return True
            return False

    def remove_object(self, object_id: str) -> bool:
        with self._lock:
            return bool(self._objects.pop(object_id, None) is not None)

    def clear(self) -> None:
        with self._lock:
            self._objects.clear()

    def count(self) -> int:
        with self._lock:
            return len(self._objects)

    def contains(self, object_id: str) -> bool:
        with self._lock:
            return object_id in self._objects

    def generate_unique_id(self, object_name: str, object_type: str) -> str:
        with self._lock:
            base_id = object_name.replace(" ", "_").lower()
            if base_id not in self._objects:
                return base_id
            existing_count = sum(
                1 for o in self._objects.values() if o.object_type == object_type
            )
            return f"{base_id}_{existing_count + 1}"

    # ------------------------------------------------------------------
    # Predicate store
    # ------------------------------------------------------------------

    def add_predicate(self, predicate: str) -> None:
        with self._lock:
            self._predicates.add(predicate)

    def add_predicates(self, predicates: List[str]) -> None:
        with self._lock:
            self._predicates.update(predicates)

    def remove_predicate(self, predicate: str) -> None:
        with self._lock:
            self._predicates.discard(predicate)

    def get_all_predicates(self) -> List[str]:
        with self._lock:
            return sorted(self._predicates)

    def clear_predicates(self) -> None:
        with self._lock:
            self._predicates.clear()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        with self._lock:
            objects_data = []
            for obj in self._objects.values():
                d: Dict[str, Any] = {
                    "object_type": obj.object_type,
                    "object_id": obj.object_id,
                    "affordances": list(obj.affordances),
                    "timestamp": obj.timestamp,
                }
                if obj.position_2d is not None:
                    d["position_2d"] = obj.position_2d
                if obj.position_3d is not None:
                    d["position_3d"] = (
                        obj.position_3d.tolist()
                        if hasattr(obj.position_3d, "tolist")
                        else obj.position_3d
                    )
                if obj.bounding_box_2d is not None:
                    d["bounding_box_2d"] = obj.bounding_box_2d
                objects_data.append(d)
        return {
            "num_objects": len(objects_data),
            "snapshot_timestamp": datetime.now().isoformat(),
            "predicates": self.get_all_predicates(),
            "objects": objects_data,
        }

    def save_to_json(self, output_path: str, include_timestamp: bool = True) -> str:
        path = Path(output_path)
        if include_timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = path.parent / f"{path.stem}_{ts}{path.suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return str(path)

    def load_from_json(self, input_path: str) -> List[DetectedObject]:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Registry file not found: {path}")
        with open(path) as f:
            data = json.load(f)
        loaded: List[DetectedObject] = []
        with self._lock:
            for d in data.get("objects", []):
                obj = DetectedObject(
                    object_type=d["object_type"],
                    object_id=d["object_id"],
                    affordances=set(d.get("affordances", [])),
                    timestamp=d.get("timestamp", time.time()),
                )
                self._objects[obj.object_id] = obj
                loaded.append(obj)
        return loaded

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.count()

    def __contains__(self, object_id: str) -> bool:
        return self.contains(object_id)

    def __repr__(self) -> str:
        with self._lock:
            return f"DetectedObjectRegistry(n={len(self._objects)})"
