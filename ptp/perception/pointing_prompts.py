"""Pointing prompt generation for Molmo interaction-point queries.

Prompts describe *what the robot is trying to accomplish* for a given
(object_type, action) pair, rather than the mechanical operation.  This
makes Molmo ground the semantically correct surface — e.g. for "open" on a
"bottle cap" the prompt asks for the surface whose normal points in the
removal direction, not just "where to push".

Usage::

    from ptp.perception.pointing_prompts import build_prompt, build_hinge_prompt

    prompt = build_prompt("push_pull", "bottle cap", action_goal="open")
    # → "Point to the surface of the bottle cap whose normal points in the
    #    direction a robot should push to open it (i.e. the cap's top face)."

    prompt = build_prompt("grasp", "cup")
    # → "Point to the best place to grasp the cup with a parallel-jaw gripper."

    hinge = build_hinge_prompt("drawer", hinge_location="left side")
    # → "Point to the hinge or pivot axis on the left side of the drawer."
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Surface / normal prompts  (push_pull primitive)
# ---------------------------------------------------------------------------

# Templates keyed by (query_type, optional action_goal).
# {object_type} and {action_goal} are format slots.
# query_type: "push" | "pull" | "push_pull"

_SURFACE_TEMPLATES: dict[tuple[str, str | None], str] = {
    # Generic — no goal specified
    ("push",      None): (
        "Point to the surface of the {object_type} whose outward normal points in the "
        "direction a robot should push it."
    ),
    ("pull",      None): (
        "Point to the surface of the {object_type} that a robot gripper should contact "
        "to pull it, so that the surface normal faces toward the robot."
    ),
    ("push_pull", None): (
        "Point to the surface of the {object_type} that a robot should push against or "
        "pull from — the surface whose normal aligns with the intended force direction."
    ),

    # Goal-aware overrides — these are much more semantically precise
    ("push",   "open"): (
        "Point to the surface of the {object_type} whose outward normal corresponds to "
        "the direction a robot must push to open it."
    ),
    ("pull",   "open"): (
        "Point to the surface of the {object_type} that a robot gripper should contact "
        "and pull toward itself to open it — the surface normal should face the robot."
    ),
    ("push_pull", "open"): (
        "Point to the surface of the {object_type} whose outward normal points in the "
        "direction a robot needs to apply force to open it."
    ),

    ("push",   "close"): (
        "Point to the surface of the {object_type} a robot should push against to close it."
    ),
    ("pull",   "close"): (
        "Point to the surface of the {object_type} a robot gripper should pull to close it."
    ),
    ("push_pull", "close"): (
        "Point to the surface of the {object_type} whose normal aligns with the direction "
        "needed to close it."
    ),

    ("push",   "press"): (
        "Point to the surface of the {object_type} a robot finger should press down on."
    ),
    ("push_pull", "press"): (
        "Point to the pressable surface of the {object_type} — the face a robot should "
        "push perpendicular to in order to actuate it."
    ),

    ("push",   "slide"): (
        "Point to the surface of the {object_type} a robot should push laterally to slide it."
    ),
    ("push_pull", "slide"): (
        "Point to the surface of the {object_type} whose normal is perpendicular to the "
        "sliding direction — the face the robot should contact to push it sideways."
    ),

    ("push",   "unscrew"): (
        "Point to the surface of the {object_type} whose outward normal aligns with the "
        "axis a robot should push to unscrew it."
    ),
    ("pull",   "unscrew"): (
        "Point to the surface of the {object_type} a robot should pull along the "
        "unscrewing axis."
    ),
    ("push_pull", "unscrew"): (
        "Point to the surface of the {object_type} whose normal corresponds to the "
        "unscrewing removal direction."
    ),

    ("push",   "pour"): (
        "Point to the surface of the {object_type} a robot should push to tilt it for pouring."
    ),
    ("push_pull", "pour"): (
        "Point to the surface of the {object_type} a robot contacts to tilt or tip it "
        "for pouring."
    ),

    ("push",   "knock over"): (
        "Point to the surface of the {object_type} a robot should push to knock it over."
    ),
    ("push_pull", "knock over"): (
        "Point to the surface of the {object_type} the robot pushes against to topple it."
    ),
}

# ---------------------------------------------------------------------------
# Grasp / interaction point prompts
# ---------------------------------------------------------------------------

_GRASP_TEMPLATES: dict[str | None, str] = {
    None:       "Point to the best place to grasp the {object_type} with a parallel-jaw gripper.",
    "pick":     "Point to the best place to pick up the {object_type} with a robot gripper.",
    "pour":     "Point to the best place to grasp the {object_type} so a robot can pour from it.",
    "open":     "Point to the best place to grasp the {object_type} to open it.",
    "close":    "Point to the best place to grasp the {object_type} to close it.",
    "unscrew":  "Point to the best place to grasp the {object_type} to unscrew it.",
    "handover": "Point to the best place to grasp the {object_type} to hand it to a person.",
    "place":    "Point to the best place to grasp the {object_type} so it can be placed precisely.",
    "displace": "Point to the best place to grasp the {object_type} to move it out of the way.",
}

# ---------------------------------------------------------------------------
# Push-aside prompt
# ---------------------------------------------------------------------------

_PUSH_ASIDE_TEMPLATES: dict[str | None, str] = {
    None:   "Point to the best surface to push the {object_type} aside with a robot.",
    "clear": "Point to the surface of the {object_type} a robot should push to clear it from the workspace.",
}

# ---------------------------------------------------------------------------
# Hinge / pivot prompts
# ---------------------------------------------------------------------------

_HINGE_TEMPLATES: dict[str | None, str] = {
    None:   "Point to the hinge or pivot axis of the {object_type}.",
    "open": "Point to the hinge or pivot axis the {object_type} rotates around when opened.",
    "close": "Point to the hinge or pivot axis the {object_type} rotates around when closed.",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_prompt(
    query_type: str,
    object_type: str,
    action_goal: Optional[str] = None,
) -> str:
    """Build a semantically grounded Molmo pointing prompt.

    Args:
        query_type:  One of ``"grasp"``, ``"push"``, ``"pull"``, ``"push_pull"``,
                     ``"push_aside"``, ``"hinge"``.
        object_type: Human-readable object label, e.g. ``"bottle cap"``, ``"drawer"``.
        action_goal: High-level goal the primitive serves, e.g. ``"open"``, ``"press"``,
                     ``"slide"``.  When ``None`` a generic prompt is used.

    Returns:
        Formatted prompt string ready to pass to ``MolmoPointDetector``.
    """
    obj = object_type.replace("_", " ")
    goal = action_goal.lower().strip() if action_goal else None

    if query_type == "grasp":
        template = _GRASP_TEMPLATES.get(goal) or _GRASP_TEMPLATES[None]

    elif query_type in ("push", "pull", "push_pull"):
        template = (
            _SURFACE_TEMPLATES.get((query_type, goal))
            or _SURFACE_TEMPLATES.get((query_type, None))
            or _SURFACE_TEMPLATES[("push_pull", None)]
        )

    elif query_type == "push_aside":
        template = _PUSH_ASIDE_TEMPLATES.get(goal) or _PUSH_ASIDE_TEMPLATES[None]

    elif query_type == "hinge":
        template = _HINGE_TEMPLATES.get(goal) or _HINGE_TEMPLATES[None]

    else:
        template = "Point to the best place to interact with the {object_type} for a robot."

    return template.format(object_type=obj, action_goal=goal or "")


def build_hinge_prompt(
    object_type: str,
    hinge_location: Optional[str] = None,
    action_goal: Optional[str] = None,
) -> str:
    """Build a hinge/pivot pointing prompt, optionally qualified by location.

    Args:
        object_type:    Object label, e.g. ``"drawer"``.
        hinge_location: Spatial description, e.g. ``"left side"``.
                        When provided it is appended to the prompt.
        action_goal:    Optional goal, e.g. ``"open"``.

    Returns:
        Formatted hinge prompt string.
    """
    goal = action_goal.lower().strip() if action_goal else None
    template = _HINGE_TEMPLATES.get(goal) or _HINGE_TEMPLATES[None]
    base = template.format(object_type=object_type.replace("_", " "))
    if hinge_location:
        base = base.rstrip(".") + f", located at the {hinge_location}."
    return base
