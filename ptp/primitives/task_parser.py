"""task_parser.py — Translate raw natural language tasks into grounded action sequences.

This runs before SkillDecomposer. It takes the user's free-form task string and
the detected object registry, asks an LLM (with the scene image) to resolve which
objects are being referred to and what high-level actions are needed, and returns
a list of ParsedAction objects.

Each ParsedAction feeds into SkillDecomposer as a separate action_name + object_id,
replacing the previous approach of passing the raw task string directly.

Usage::

    parser = TaskParser(llm_client=my_client)
    actions = parser.parse(
        task="open the cabinet with the blue tape on the handle",
        registry=detected_registry,
        image_bytes=color_png_bytes,
    )
    # → [ParsedAction(action="open", object_id="blue_handle_2", description="...")]
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ptp.llm_interface.base import GenerateConfig, ImagePart, LLMClient

_DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parents[2] / "config" / "task_parser_prompts.yaml"
_DEFAULT_ROBOT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "robot.yaml"

logger = logging.getLogger(__name__)


@dataclass
class ParsedAction:
    """A single high-level action resolved against the detected object registry."""
    action: str
    object_id: Optional[str]
    description: str
    secondary_object_id: Optional[str] = None  # destination/recipient for place, pour, hand_over etc.


@dataclass
class TaskParseResult:
    """Full output from TaskParser.parse()."""
    actions: List[ParsedAction]
    rationale: str
    raw_llm_response: str


class TaskParser:
    """Translate a natural language task into grounded (action, object_id) pairs.

    Args:
        llm_client: Any ptp.llm_interface.LLMClient implementation.
        prompts_config_path: Override path to the YAML prompt template + schema.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompts_config_path: Optional[Path] = None,
        robot_config_path: Optional[Path] = None,
    ) -> None:
        self._llm_client = llm_client
        self.prompts_config_path = Path(prompts_config_path or _DEFAULT_PROMPTS_PATH)
        self._robot_config_path = Path(robot_config_path or _DEFAULT_ROBOT_CONFIG_PATH)
        self._prompts_cache: Tuple[float, Optional[Dict[str, Any]]] = (0.0, None)
        self._robot_config: Optional[Dict[str, Any]] = None

    def parse(
        self,
        task: str,
        registry: Any,
        image_bytes: Optional[bytes] = None,
        temperature: float = 0.1,
    ) -> TaskParseResult:
        """Parse a task string into an ordered list of grounded actions.

        Args:
            task: Raw natural language task from the user.
            registry: DetectedObjectRegistry (or dict from registry.to_dict()).
            image_bytes: PNG bytes of the RGB snapshot to send alongside the prompt.
            temperature: LLM sampling temperature.

        Returns:
            TaskParseResult with a list of ParsedAction objects.
        """
        objects = self._extract_objects(registry)
        prompts = self._load_prompts()
        robot_cfg = self._load_robot_config()
        prompt = self._build_prompt(task, objects, prompts["template"], registry=registry, robot_config=robot_cfg)
        media: List[Any] = [ImagePart(data=image_bytes, mime_type="image/png")] if image_bytes else []

        config = GenerateConfig(
            temperature=temperature,
            top_p=0.8,
            max_output_tokens=1024,
            response_mime_type="application/json",
            response_json_schema=prompts["response_schema"],
        )
        logger.info("[TaskParser] Prompt:\n%s", prompt)
        contents = media + [prompt] if media else prompt
        response = self._llm_client.generate(contents, config=config)
        raw = response.text
        logger.info("[TaskParser] LLM response:\n%s", raw)

        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_objects(self, registry: Any) -> List[Dict[str, Any]]:
        """Return a flat list of object dicts regardless of registry type."""
        if hasattr(registry, "to_dict"):
            registry = registry.to_dict()
        if isinstance(registry, dict):
            return registry.get("objects") or []
        return []

    def _load_robot_config(self) -> Dict[str, Any]:
        if self._robot_config is not None:
            return self._robot_config
        if self._robot_config_path.exists():
            self._robot_config = yaml.safe_load(self._robot_config_path.read_text()) or {}
        else:
            self._robot_config = {}
        return self._robot_config

    def _build_prompt(
        self,
        task: str,
        objects: List[Dict[str, Any]],
        template: str,
        registry: Any = None,
        robot_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        lines = []
        for obj in objects:
            pos = obj.get("position_2d") or obj.get("latest_position_2d")
            pos_str = f"  pos={pos}" if pos else ""
            lines.append(f"  - {obj.get('object_id')} ({obj.get('object_type')}){pos_str}")
        object_list = "\n".join(lines) if lines else "  (none detected)"

        # Build spatial relations from contact graph support_tree if available.
        spatial_lines = []
        cg = getattr(registry, "contact_graph", None) if registry is not None else None
        if cg is not None:
            support_tree = getattr(cg, "support_tree", {}) or {}
            for lower_id, upper_ids in support_tree.items():
                for upper_id in upper_ids:
                    spatial_lines.append(f"  - {lower_id} supports {upper_id}  (i.e. {upper_id} is on top of {lower_id})")
        spatial_relations = "\n".join(spatial_lines) if spatial_lines else "  (none detected)"

        # Build robot constraints as a rule paragraph injected into Rule 5.
        rc = robot_config or {}
        caps = rc.get("capabilities", {})
        max_held = caps.get("max_held_objects", 1)
        bimanual = caps.get("bimanual", False)
        constraint_lines = [f"Robot constraints ({rc.get('name', 'unknown')}):"]
        if max_held == 1:
            constraint_lines.append(
                "     ⚠ SINGLE GRIPPER: the robot can hold at most 1 object at a time. "
                "The gripper MUST be empty before grasping a new object. "
                "Any pick_up action must be preceded by a place action if the gripper is already holding something. "
                "Exception: tool use (e.g. holding a tool to interact with another object without releasing the tool)."
            )
        else:
            constraint_lines.append(f"     Max objects held simultaneously: {max_held}.")
        if not bimanual:
            constraint_lines.append(
                "     ⚠ NO BIMANUAL: all actions use a single arm — no simultaneous two-handed operations."
            )
        robot_constraints = "\n".join(constraint_lines)

        return (
            template
            .replace("{task}", task)
            .replace("{object_list}", object_list)
            .replace("{spatial_relations}", spatial_relations)
            .replace("{robot_constraints}", robot_constraints)
        )

    def _parse_response(self, raw: str) -> TaskParseResult:
        data = json.loads(raw)
        # Unwrap JSON Schema envelope that some models emit.
        if data.get("type") == "object" and "properties" in data and "actions" not in data:
            data = data["properties"]
        actions = [
            ParsedAction(
                action=a["action"],
                object_id=a.get("object_id"),
                description=a.get("description", ""),
                secondary_object_id=a.get("secondary_object_id"),
            )
            for a in data.get("actions", [])
        ]
        return TaskParseResult(
            actions=actions,
            rationale=data.get("rationale", ""),
            raw_llm_response=raw,
        )

    def _load_prompts(self) -> Dict[str, Any]:
        path = self.prompts_config_path
        if not path.exists():
            raise FileNotFoundError(f"Task parser prompt config not found: {path}")
        mtime = path.stat().st_mtime
        cached_mtime, cached = self._prompts_cache
        if mtime == cached_mtime and cached:
            return cached
        data = yaml.safe_load(path.read_text()) or {}
        template = data.get("template")
        schema = data.get("response_schema")
        if not template or not isinstance(schema, dict):
            raise ValueError(f"Task parser config {path} must define 'template' and 'response_schema'")
        result = {"template": template, "response_schema": json.loads(json.dumps(schema))}
        self._prompts_cache = (mtime, result)
        return result
