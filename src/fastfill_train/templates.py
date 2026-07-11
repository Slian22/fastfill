"""Prompt templates for the FastFill planner (two training targets).

Template A ("direct"):   user = instruction + input codec,
                         assistant = layout codec.
Template B ("plan"):     assistant = one short structured plan line, then
                         the layout codec — the OptiScene ablation showed a
                         semantic-reasoning step buys +5pp usability
                         (28%→33%), but FastFill is token-budgeted, so the
                         plan is a compact single line, never long chain of
                         thought.

The plan line for training labels is DERIVED deterministically from the
ground-truth layout (decoded from the record's output codec), so no extra
annotation pass is needed. ``split_completion`` inverts the format at eval
time: it strips the plan line so the codec decoder sees only layout text.
"""

from __future__ import annotations

from dataclasses import dataclass

import fastfill_train  # noqa: F401  (vendor path bootstrap)
from scenesmith.growing_world.fastfill.codec import decode_floor_layout
from scenesmith.growing_world.fastfill.schema import Anchor, FloorLayout

TEMPLATES = ("direct", "plan")
PLAN_PREFIX = "PLAN:"
LAYOUT_PREFIX = "LAYOUT:"

_ZONE_BY_KEYWORD: tuple[tuple[str, str], ...] = (
    ("bed", "sleeping"),
    ("wardrobe", "storage"),
    ("dresser", "storage"),
    ("cabinet", "storage"),
    ("shelf", "storage"),
    ("bookcase", "storage"),
    ("desk", "work"),
    ("sofa", "lounge"),
    ("armchair", "lounge"),
    ("coffee_table", "lounge"),
    ("tv", "lounge"),
    ("dining", "dining"),
    ("counter", "kitchen"),
    ("stove", "kitchen"),
    ("fridge", "kitchen"),
    ("kitchen", "kitchen"),
    ("sink", "washing"),
    ("toilet", "bathroom"),
    ("bathtub", "bathroom"),
    ("shower", "bathroom"),
)


def _zone_for(category: str) -> str | None:
    lowered = category.lower()
    for keyword, zone in _ZONE_BY_KEYWORD:
        if keyword in lowered:
            return zone
    return None


def derive_plan_line(layout: FloorLayout) -> str:
    """One-line structured plan from a ground-truth floor layout.

    Format: ``PLAN: zones=a,b; anchors=cat:wall,cat2:free; n=<count>`` —
    deterministic, order-stable, and short (typically < 120 chars).
    """
    zones: list[str] = []
    for obj in layout.objects:
        zone = _zone_for(obj.category)
        if zone and zone not in zones:
            zones.append(zone)
    anchors: list[str] = []
    for obj in layout.objects:
        tag = "wall" if obj.anchor in (Anchor.WALL, Anchor.CORNER) else "free"
        entry = f"{obj.category.lower().replace(' ', '_')}:{tag}"
        if entry not in anchors:
            anchors.append(entry)
    zones_part = ",".join(zones) if zones else "general"
    anchors_part = ",".join(anchors[:8])
    return (
        f"{PLAN_PREFIX} zones={zones_part}; anchors={anchors_part}; "
        f"n={len(layout.objects)}"
    )


@dataclass(frozen=True)
class RenderedExample:
    """One SFT example in chat-messages form (matches TRL SFTTrainer)."""

    user: str
    assistant: str

    def to_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "user", "content": self.user},
            {"role": "assistant", "content": self.assistant},
        ]


def render_sft_example(
    record: dict, template: str, *, room_id: str = "train"
) -> RenderedExample:
    """Render one export_sft record ({instruction,input,output}) per template.

    Surface records always use the direct format (their outputs are short);
    the plan step applies to FLOOR records only, mirroring OptiScene's
    room-level reasoning.
    """
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r}; expected {TEMPLATES}")
    user = f"{record['instruction']}\n{record['input']}"
    output = record["output"]
    if template == "direct" or _is_surface_record(record):
        return RenderedExample(user=user, assistant=output)
    layout = decode_floor_layout(output, room_id)
    plan = derive_plan_line(layout)
    return RenderedExample(
        user=user, assistant=f"{plan}\n{LAYOUT_PREFIX}\n{output}"
    )


def _is_surface_record(record: dict) -> bool:
    return "#" in str(record.get("uid", ""))  # surface uids: <sample>#<group>


def split_completion(text: str) -> tuple[str, str]:
    """Invert the template at eval time: ``(plan_line, layout_codec_text)``.

    Handles both formats: direct completions return an empty plan. Robust to
    the model echoing extra whitespace, but never rewrites layout content.
    """
    stripped = text.strip()
    if not stripped.startswith(PLAN_PREFIX):
        return "", stripped
    plan_line, _, rest = stripped.partition("\n")
    rest = rest.strip()
    if rest.startswith(LAYOUT_PREFIX):
        rest = rest[len(LAYOUT_PREFIX) :].lstrip("\n").lstrip()
    return plan_line.strip(), rest
