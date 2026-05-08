"""Skill decomposer: translates symbolic actions into validated primitive plans.

The decomposer builds a structured prompt from the world state (object registry,
latest perception snapshot, robot state) and sends it to an LLM.  The LLM
returns a JSON primitive sequence that is validated against the PRIMITIVE_LIBRARY
before being returned as a SkillPlan.

The decomposer is deliberately decoupled from any orchestrator or PDDL planner —
it only needs:
  - An LLMClient (ptp.llm_interface)
  - A perception_pool_dir on disk for snapshot lookup
  - An optional state_dir for registry.json / state.json fallbacks

World state can also be supplied directly as a dict, enabling use without any
file-system state at all.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    from google import genai as _genai
    from google.genai import types as _genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _genai = None
    _genai_types = None
    _GENAI_AVAILABLE = False

from ptp.llm_interface.base import GenerateConfig, ImagePart, LLMClient
from ptp.primitives.library import PRIMITIVE_LIBRARY
from ptp.primitives.snapshot_utils import (
    SnapshotArtifacts,
    SnapshotCache,
    latest_snapshot_for_object_ids,
    load_snapshot_artifacts,
)
from ptp.primitives.types import (
    PrimitiveCall,
    SkillPlan,
    SkillPlanDiagnostics,
    compute_registry_hash,
)

_DEFAULT_PROMPTS_PATH    = Path(__file__).resolve().parents[2] / "config" / "skill_decomposer_prompts.yaml"
_DEFAULT_CATALOG_PATH    = Path(__file__).resolve().parents[2] / "config" / "primitive_descriptions.md"

logger = logging.getLogger(__name__)


class SkillDecomposer:
    """LLM-backed decomposer that maps symbolic actions to executable primitives.

    Args:
        llm_client: Any ptp.llm_interface.LLMClient implementation (required
                    unless a legacy Gemini api_key + model_name is provided).
        perception_pool_dir: Root of the perception-pool directory tree.
        state_dir: Optional directory containing registry.json and state.json
                   for disk-based world-state fallback.
        orchestrator: Optional orchestrator object; used only to read live world
                      state when present.  Pass None for standalone use.
        primitive_catalog_path: Override path to the primitive descriptions Markdown.
        prompts_config_path: Override path to the YAML prompt template + schema.
        api_key: Legacy Gemini API key (used when llm_client is None).
        model_name: Legacy Gemini model name.

    Example:
        >>> decomposer = SkillDecomposer(llm_client=my_client,
        ...                              perception_pool_dir=Path("outputs/pool"))
        >>> plan = decomposer.plan("pick", {"objects": ["rubber_duck_1"]})
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        perception_pool_dir: Optional[Path] = None,
        state_dir: Optional[Path] = None,
        orchestrator: Optional[Any] = None,
        primitive_catalog_path: Optional[Path] = None,
        prompts_config_path: Optional[Path] = None,
        # Legacy Gemini parameters
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        client: Optional[Any] = None,
        llm_config_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._llm_client  = llm_client
        self.orchestrator = orchestrator

        root = Path(__file__).resolve().parents[2]
        self._state_dir          = Path(state_dir) if state_dir else root / "outputs" / "orchestrator_state"
        self._perception_pool_dir = Path(perception_pool_dir) if perception_pool_dir else self._state_dir / "perception_pool"

        self.primitive_catalog_path = Path(primitive_catalog_path or _DEFAULT_CATALOG_PATH)
        self.prompts_config_path    = Path(prompts_config_path    or _DEFAULT_PROMPTS_PATH)

        self._prompts_cache: Tuple[float, Optional[Dict[str, Any]]] = (0.0, None)
        self._snapshot_cache = SnapshotCache()

        # Legacy Gemini path (used when no llm_client provided)
        if llm_client is None:
            if not _GENAI_AVAILABLE:
                raise ImportError(
                    "google-genai is required when llm_client is not provided. "
                    "Install it with: pip install google-genai"
                )
            self._model_name = model_name
            self._genai_client = client or _genai.Client(api_key=api_key)
            _supports_thinking = "gemini-2.5" in model_name
            default_cfg: Dict[str, Any] = {
                "top_p": 0.8,
                "max_output_tokens": 4096,
                "response_mime_type": "application/json",
            }
            if _supports_thinking:
                default_cfg["thinking_config"] = _genai_types.ThinkingConfig(thinking_budget=-1)
            self._llm_config_kwargs: Dict[str, Any] = {**default_cfg, **(llm_config_kwargs or {})}
        else:
            self._model_name       = ""
            self._genai_client     = None
            self._llm_config_kwargs = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        action_name: str,
        parameters: Dict[str, Any],
        world_hint: Optional[Dict[str, Any]] = None,
        temperature: float = 0.1,
    ) -> SkillPlan:
        """Decompose a symbolic action into a validated SkillPlan.

        Args:
            action_name: Symbolic action name (e.g., "pick", "place", "displace").
            parameters: Action parameters (e.g., {"objects": ["rubber_duck_1"]}).
            world_hint: Optional world-state dict override; merged over live state.
            temperature: LLM temperature.

        Returns:
            Validated SkillPlan ready for PrimitiveExecutor.
        """
        world_state   = self._prepare_world_state(world_hint, parameters)
        registry_hash = compute_registry_hash(world_state.get("registry", {}))

        catalog_text     = self._read_catalog()
        target_ids       = self._extract_object_ids(parameters)
        snapshot_id      = (
            latest_snapshot_for_object_ids(
                world_state, self._perception_pool_dir, target_ids, cache=self._snapshot_cache
            ) or world_state.get("last_snapshot_id")
        )
        artifacts  = load_snapshot_artifacts(
            world_state, self._perception_pool_dir, cache=self._snapshot_cache, snapshot_id=snapshot_id
        )
        prompts    = self._load_prompts()
        prompt     = self._build_prompt(action_name, parameters, world_state,
                                        catalog_text, artifacts, prompts["template"])
        media      = self._build_media_parts(artifacts, world_state.get("latest_detections") or [])
        raw        = self._call_llm(prompt, temperature=temperature,
                                    media_parts=media, response_schema=prompts["response_schema"])

        plan = self._parse_plan(raw, action_name=action_name, registry_hash=registry_hash)
        plan.source_snapshot_id = artifacts.snapshot_id
        self._post_process_plan(plan, world_state, artifacts)
        return plan

    # ------------------------------------------------------------------
    # World state
    # ------------------------------------------------------------------

    def _prepare_world_state(
        self,
        world_hint: Optional[Dict[str, Any]],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        world_state: Dict[str, Any] = {}

        # 1. Live orchestrator state
        if self.orchestrator is not None:
            fn = getattr(self.orchestrator, "get_world_state_snapshot", None)
            if fn:
                world_state = fn() or {}

        # 2. Disk fallback
        if not world_state.get("registry"):
            rpath = self._state_dir / "registry.json"
            if rpath.exists():
                try:
                    world_state["registry"] = json.loads(rpath.read_text())
                except Exception:
                    pass
        if "last_snapshot_id" not in world_state:
            spath = self._state_dir / "state.json"
            if spath.exists():
                try:
                    world_state["last_snapshot_id"] = json.loads(spath.read_text()).get("last_snapshot_id")
                except Exception:
                    pass

        # 3. User hint overrides
        if world_hint:
            world_state.update(world_hint)

        # Annotate staleness
        registry = world_state.get("registry", {})
        now = time.time()
        for obj in registry.get("objects", []):
            ts = obj.get("timestamp")
            if isinstance(ts, (int, float)):
                obj["staleness_seconds"] = max(0.0, now - ts)

        # Merge snapshot detections
        snap_id = world_state.get("last_snapshot_id")
        detections = self._load_snapshot_detections(snap_id)
        world_state["latest_detections"] = detections
        det_map = {d.get("object_id"): d for d in detections}
        merged: List[Dict[str, Any]] = []
        for obj in registry.get("objects", []):
            det = det_map.get(obj.get("object_id")) or {}
            m   = dict(obj)
            if det:
                m.setdefault("interaction_points", det.get("interaction_points"))
                m.setdefault("latest_observation", snap_id)
                for field in ("bounding_box_2d", "position_2d", "position_3d"):
                    if det.get(field) is not None:
                        m.setdefault(f"latest_{field}", det[field])
            merged.append(m)
        registry["objects"] = merged
        world_state["registry"] = registry
        world_state["relevant_objects"] = self._filter_relevant_objects(merged, parameters)
        return world_state

    def _extract_object_ids(self, parameters: Dict[str, Any]) -> List[str]:
        ids: List[str] = []
        for key in ("object_id", "objects", "object_ids"):
            val = parameters.get(key)
            if isinstance(val, str):
                ids.append(val)
            elif isinstance(val, (list, tuple)):
                ids.extend(str(v) for v in val if v)
        return [i for i in ids if i]

    def _filter_relevant_objects(
        self, objects: List[Dict[str, Any]], parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        tokens: List[str] = []
        for v in parameters.values():
            if isinstance(v, str):
                tokens.append(v.lower())
            elif isinstance(v, (list, tuple)):
                tokens.extend(str(x).lower() for x in v if isinstance(x, (str, int)))
        if not tokens:
            return objects[:5]
        relevant = [
            o for o in objects
            if any(t in (str(o.get("object_id", "")).lower(),
                         str(o.get("object_type", "")).lower())
                   for t in tokens)
        ]
        return relevant or objects[:5]

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        action_name: str,
        parameters: Dict[str, Any],
        world_state: Dict[str, Any],
        catalog_text: str,
        artifacts: SnapshotArtifacts,
        template: str,
    ) -> str:
        registry  = world_state.get("registry", {})
        relevant  = world_state.get("relevant_objects") or registry.get("objects", [])
        det_map   = {d.get("object_id"): d for d in world_state.get("latest_detections") or []}
        perception_ctx = self._format_perception_context(world_state, artifacts)

        def _fmt_obj(o: Dict[str, Any]) -> str:
            det   = det_map.get(o.get("object_id")) or {}
            stale = o.get("staleness_seconds")
            stale_note = f"{stale:.1f}s old" if stale is not None else "freshness: unknown"
            pos3d = o.get("position_3d")
            pos_note = f" position_3d={[round(v, 3) for v in pos3d]}" if pos3d else ""
            return (
                f"- {o.get('object_type')} ({o.get('object_id')}):"
                f"{pos_note}"
                f" latest_snapshot={det.get('snapshot_id') or o.get('latest_observation')}"
                f" {stale_note}"
            )

        object_section = "\n".join(_fmt_obj(o) for o in relevant[:10]) or "none"
        action_schema  = self._resolve_action_schema(action_name)
        role_section   = self._resolve_role_assignments(action_name, parameters, action_schema)

        substitutions = {
            "{primitive_catalog}":   catalog_text.strip(),
            "{action_name}":         action_name,
            "{action_parameters}":   action_schema["parameters"],
            "{action_description}":  action_schema["description"],
            "{parameters}":          json.dumps(parameters, ensure_ascii=True),
            "{role_assignments}":    role_section,
            "{object_section}":      object_section,
            "{last_snapshot_id}":    str(artifacts.snapshot_id),
            "{perception_context}":  perception_ctx,
        }
        result = template
        for k, v in substitutions.items():
            result = result.replace(k, v)
        return result

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        prompt: str,
        temperature: float,
        media_parts: Optional[List[Any]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self._llm_client is not None:
            config = GenerateConfig(
                temperature=temperature,
                top_p=0.8,
                max_output_tokens=4096,
                response_mime_type="application/json",
                response_json_schema=response_schema,
            )
            contents: List[Any] = list(media_parts or []) + [prompt]
            response = self._llm_client.generate(
                contents if len(contents) > 1 else prompt, config=config
            )
            logger.debug("[decomposer] LLM response (%d chars)", len(response.text))
            return response.text

        # Legacy Gemini path
        cfg_kwargs: Dict[str, Any] = {**self._llm_config_kwargs, "temperature": temperature}
        if response_schema:
            cfg_kwargs["response_json_schema"] = response_schema
        cfg_gemini = _genai_types.GenerateContentConfig(**cfg_kwargs)
        contents_g = list(media_parts or []) + [prompt]
        response   = self._genai_client.models.generate_content(
            model=self._model_name, contents=contents_g, config=cfg_gemini
        )
        text = getattr(response, "text", None)
        if text is None:
            raise ValueError("LLM returned no text payload")
        return text if isinstance(text, str) else str(text)

    # ------------------------------------------------------------------
    # Parse + post-process
    # ------------------------------------------------------------------

    def _parse_plan(
        self, response_text: str, action_name: str, registry_hash: Optional[str]
    ) -> SkillPlan:
        data = json.loads(response_text)
        diagnostics_block = data.get("diagnostics") or {}
        plan = SkillPlan(
            action_name=action_name,
            primitives=[PrimitiveCall.from_dict(p) for p in data.get("primitives", [])],
            diagnostics=SkillPlanDiagnostics(
                assumptions=data.get("assumptions") or [],
                warnings=diagnostics_block.get("warnings") or [],
                freshness_notes=diagnostics_block.get("freshness_notes") or [],
                freshness=diagnostics_block.get("freshness", {}),
                rationale=diagnostics_block.get("rationale", ""),
                interaction_points=data.get("interaction_points") or [],
            ),
            registry_hash=registry_hash,
        )
        return plan

    def _post_process_plan(
        self,
        plan: SkillPlan,
        world_state: Dict[str, Any],
        artifacts: SnapshotArtifacts,
    ) -> None:
        objects  = world_state.get("registry", {}).get("objects", [])
        indexed  = {o.get("object_id"): o for o in objects}
        det_map  = {d.get("object_id"): d for d in world_state.get("latest_detections") or []}

        for idx, prim in enumerate(plan.primitives):
            ref_id = prim.references.get("object_id")
            if ref_id and ref_id not in indexed:
                plan.diagnostics.warnings.append(
                    f"[{idx}] reference object '{ref_id}' not found in registry"
                )
                continue
            ip_label = prim.references.get("interaction_point")
            if ref_id and ip_label:
                det = det_map.get(ref_id, {})
                ip  = (det.get("interaction_points") or {}).get(ip_label)
                if not ip:
                    plan.diagnostics.warnings.append(
                        f"[{idx}] missing interaction point '{ip_label}' on {ref_id}"
                    )
                else:
                    prim.metadata.setdefault("resolved_interaction_point", ip)
            if ref_id and ref_id in indexed:
                det = det_map.get(ref_id, {})
                if not (det.get("interaction_points") or {}):
                    plan.diagnostics.warnings.append(
                        f"[{idx}] object '{ref_id}' has no interaction points in snapshot"
                    )

        # Defer Molmo grounding to executor; set point_label fallback now.
        for idx, prim in enumerate(plan.primitives):
            if prim.name != "move_gripper_to_pose":
                continue
            if not prim.metadata.get("pointing_guidance"):
                continue
            ref_id = prim.references.get("object_id")
            if ref_id:
                prim.parameters.setdefault("point_label", ref_id)
                plan.diagnostics.warnings.append(
                    f"[{idx}] pointing_guidance deferred to execution; point_label={ref_id!r} set as fallback"
                )

        for obj in world_state.get("relevant_objects", []):
            stale = obj.get("staleness_seconds")
            if stale is not None and stale > 90:
                note = f"Object {obj.get('object_id')} observation is {stale:.1f}s old"
                plan.diagnostics.warnings.append(note)
                plan.diagnostics.freshness_notes.append(note)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _resolve_action_schema(self, action_name: str) -> Dict[str, str]:
        """Return {"description": str, "parameters": str} for an action name.

        Checks the live orchestrator task_analysis first; falls back to empty
        strings — no hard dependency on clutter_module or any PDDL layer.
        """
        def _extract(action: Dict) -> Dict[str, str]:
            desc   = action.get("description", "")
            params = action.get("parameters", "")
            if isinstance(params, list):
                params = "(" + " ".join(params) + ")"
            return {"description": desc, "parameters": str(params)}

        if self.orchestrator is not None:
            task_analysis = getattr(self.orchestrator, "task_analysis", None)
            if task_analysis is not None:
                fn = getattr(task_analysis, "action_context", None)
                if fn:
                    for action in fn():
                        if action.get("name") == action_name:
                            result = _extract(action)
                            if result["description"] or result["parameters"]:
                                return result

        return {"description": "", "parameters": ""}

    def _resolve_role_assignments(
        self,
        action_name: str,
        parameters: Dict[str, Any],
        action_schema: Dict[str, str],
    ) -> str:
        objects: List[str] = parameters.get("objects") or []
        if not objects:
            return ""
        var_names = re.findall(r"\?(\w+)", action_schema.get("parameters", ""))
        lines = ["Explicit role assignments for this invocation:"]
        for i, obj_id in enumerate(objects):
            role = f"?{var_names[i]}" if i < len(var_names) else f"?arg{i}"
            hint = ""
            if "blocker" in role:
                hint = "  ← THIS is the object to physically move/push aside"
            elif "target" in role:
                hint = "  ← this is the goal object being unblocked (do NOT move it)"
            elif i == 0 and action_name in ("displace", "push-aside", "clear-obstruction"):
                hint = "  ← THIS is the object to physically move"
            lines.append(f"  {role} = {obj_id}{hint}")
        return "\n".join(lines)

    def _format_perception_context(
        self, world_state: Dict[str, Any], artifacts: SnapshotArtifacts
    ) -> str:
        snap_id  = artifacts.snapshot_id or world_state.get("last_snapshot_id")
        meta     = artifacts.meta
        if meta is None and snap_id:
            meta = (world_state.get("snapshot_index") or {}).get("snapshots", {}).get(snap_id)

        def _fmt_snap(m: Optional[Dict]) -> str:
            if not m:
                return "none"
            files = m.get("files", {})
            return (f"id={snap_id}, captured_at={m.get('captured_at')}, "
                    f"color={files.get('color')}, depth={files.get('depth_npz')}")

        robot_state = world_state.get("robot_state") or {}
        registry    = world_state.get("registry", {})
        return (
            f"registry: objects={registry.get('num_objects', '?')}, "
            f"ts={registry.get('detection_timestamp')}; "
            f"snapshot: {_fmt_snap(meta)}; "
            f"robot: provider={robot_state.get('provider', 'unknown')}, "
            f"stamp={robot_state.get('stamp')}"
        )

    def _build_media_parts(
        self,
        artifacts: SnapshotArtifacts,
        detections: List[Dict[str, Any]],
    ) -> List[Any]:
        if not artifacts.color_bytes:
            return []
        img_bytes = self._build_labeled_image(artifacts.color_bytes, detections)
        if self._llm_client is not None:
            return [ImagePart(data=img_bytes, mime_type="image/png")]
        return [_genai_types.Part.from_bytes(data=img_bytes, mime_type="image/png")]

    def _build_labeled_image(
        self, color_bytes: bytes, detections: List[Dict[str, Any]]
    ) -> bytes:
        """Overlay bounding boxes + object ID labels on the RGB snapshot."""
        import hashlib
        import io as _io
        from PIL import Image, ImageDraw, ImageFont

        _PALETTE = [
            (255, 80, 80), (80, 200, 80), (80, 160, 255), (255, 200, 40),
            (200, 80, 255), (40, 220, 220), (255, 140, 0), (180, 255, 80),
        ]
        try:
            img  = Image.open(_io.BytesIO(color_bytes)).convert("RGB")
            draw = ImageDraw.Draw(img, "RGBA")
            fsz  = max(14, img.width // 50)
            try:
                font = ImageFont.load_default(size=fsz)
            except TypeError:
                font = ImageFont.load_default()

            for det in detections:
                bbox   = det.get("bounding_box_2d")
                obj_id = det.get("object_id") or det.get("object_type") or "?"
                if not bbox or len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                if x1 >= x2 or y1 >= y2:
                    continue
                c   = _PALETTE[int(hashlib.md5(obj_id.encode()).hexdigest(), 16) % len(_PALETTE)]
                draw.rectangle([x1, y1, x2, y2], outline=c + (220,), width=3, fill=c + (40,))
                try:
                    tb  = font.getbbox(obj_id)
                    tw, th = tb[2] - tb[0], tb[3] - tb[1]
                except Exception:
                    tw, th = len(obj_id) * fsz // 2, fsz
                pad = 4
                lx, ly = x1, max(0, y1 - th - pad * 2)
                draw.rectangle([lx, ly, lx + tw + pad * 2, ly + th + pad * 2], fill=c + (220,))
                draw.text((lx + pad, ly + pad), obj_id, fill=(255, 255, 255), font=font)

            buf = _io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as exc:
            logger.warning("_build_labeled_image failed: %s", exc)
            return color_bytes

    def _load_snapshot_detections(self, snapshot_id: Optional[str]) -> List[Dict[str, Any]]:
        if not snapshot_id:
            return []
        path = self._perception_pool_dir / "snapshots" / snapshot_id / "detections.json"
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return []
        detections = payload.get("objects") or []
        for d in detections:
            d.setdefault("snapshot_id", snapshot_id)
        return detections

    def _read_catalog(self) -> str:
        if self.primitive_catalog_path.exists():
            return self.primitive_catalog_path.read_text()
        # Fallback: generate catalog from PRIMITIVE_LIBRARY
        lines = ["# Primitive Library\n"]
        for name, schema in PRIMITIVE_LIBRARY.items():
            lines.append(f"## {name}\n{schema.description}\n")
        return "\n".join(lines)

    def _load_prompts(self) -> Dict[str, Any]:
        path  = self.prompts_config_path
        if not path.exists():
            raise FileNotFoundError(f"Prompt config not found: {path}")
        mtime = path.stat().st_mtime
        cached_mtime, cached = self._prompts_cache
        if mtime == cached_mtime and cached:
            return cached
        data = yaml.safe_load(path.read_text()) or {}
        template = data.get("template")
        schema   = data.get("response_schema")
        if not template or not isinstance(schema, dict):
            raise ValueError(f"Prompt config {path} must define 'template' and 'response_schema'")
        result = {"template": template, "response_schema": json.loads(json.dumps(schema))}
        self._prompts_cache = (mtime, result)
        return result
