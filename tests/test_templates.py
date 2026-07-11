"""Tests for prompt templates: render / split round-trip and plan derivation."""

from __future__ import annotations

import fastfill_train  # noqa: F401  (vendor path bootstrap)
import pytest
from scenesmith.growing_world.fastfill.codec import encode_floor_layout
from scenesmith.growing_world.fastfill.schema import (
    Anchor,
    FloorLayout,
    FloorObjectSpec,
)

from fastfill_train.templates import (
    LAYOUT_PREFIX,
    PLAN_PREFIX,
    derive_plan_line,
    render_sft_example,
    split_completion,
)


def _layout() -> FloorLayout:
    return FloorLayout(
        room_id="r0",
        objects=(
            FloorObjectSpec(
                object_id="bed_0",
                category="bed",
                dimensions=(1.6, 2.0, 0.5),
                position_xy=(0.0, 0.0),
                yaw_deg=0.0,
                anchor=Anchor.WALL,
            ),
            FloorObjectSpec(
                object_id="desk_0",
                category="desk",
                dimensions=(1.2, 0.6, 0.75),
                position_xy=(1.0, 1.0),
                yaw_deg=90.0,
                anchor=Anchor.FREE,
            ),
        ),
    )


def _floor_record(uid: str = "s0") -> dict:
    return {
        "uid": uid,
        "split_key": "house0",
        "source_dataset": "test_ds",
        "instruction": "Place floor-standing furniture.",
        "input": "room bedroom id=r0",
        "output": encode_floor_layout(_layout()),
    }


def _surface_record() -> dict:
    return {
        "uid": "s0#g0",  # '#' marks a surface record
        "split_key": "house0",
        "source_dataset": "test_ds",
        "instruction": "Place small objects on the surface.",
        "input": "surface surf0 kind=top",
        "output": "group g0 surface=surf0 pattern=free anchor=-\n"
        "plate|25,25,3|0,0|0|-",
    }


# ---------------------------------------------------------- render_sft_example


def test_render_direct_is_instruction_input_output() -> None:
    record = _floor_record()
    rendered = render_sft_example(record, "direct")
    assert rendered.user == f"{record['instruction']}\n{record['input']}"
    assert rendered.assistant == record["output"]


def test_render_plan_prepends_plan_line_on_floor_records() -> None:
    record = _floor_record()
    rendered = render_sft_example(record, "plan")
    assert rendered.user == f"{record['instruction']}\n{record['input']}"
    assert rendered.assistant.startswith(PLAN_PREFIX)
    plan_line, _, rest = rendered.assistant.partition("\n")
    assert plan_line == derive_plan_line(_layout())
    assert rest == f"{LAYOUT_PREFIX}\n{record['output']}"


def test_render_plan_on_surface_record_stays_direct() -> None:
    record = _surface_record()
    rendered = render_sft_example(record, "plan")
    assert rendered.assistant == record["output"]
    assert PLAN_PREFIX not in rendered.assistant


def test_render_unknown_template_raises() -> None:
    with pytest.raises(ValueError):
        render_sft_example(_floor_record(), "chain_of_thought")


def test_to_messages_chat_shape() -> None:
    messages = render_sft_example(_floor_record(), "direct").to_messages()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert all(set(m) == {"role", "content"} for m in messages)


# ------------------------------------------------------------ split_completion


def test_split_inverts_direct_format() -> None:
    output = _floor_record()["output"]
    rendered = render_sft_example(_floor_record(), "direct")
    plan, layout_text = split_completion(rendered.assistant)
    assert plan == ""
    assert layout_text == output


def test_split_inverts_plan_format() -> None:
    record = _floor_record()
    rendered = render_sft_example(record, "plan")
    plan, layout_text = split_completion(rendered.assistant)
    assert plan == derive_plan_line(_layout())
    assert layout_text == record["output"]


def test_split_tolerates_extra_whitespace() -> None:
    record = _floor_record()
    rendered = render_sft_example(record, "plan")
    plan, layout_text = split_completion(f"\n  {rendered.assistant}\n\n")
    assert plan == derive_plan_line(_layout())
    assert layout_text == record["output"]


# ------------------------------------------------------------ derive_plan_line


def test_derive_plan_line_zones_anchors_count() -> None:
    line = derive_plan_line(_layout())
    assert line == f"{PLAN_PREFIX} zones=sleeping,work; anchors=bed:wall,desk:free; n=2"


def test_derive_plan_line_deterministic_and_short() -> None:
    assert derive_plan_line(_layout()) == derive_plan_line(_layout())
    assert len(derive_plan_line(_layout())) < 200
    assert "\n" not in derive_plan_line(_layout())


def test_derive_plan_line_unknown_category_falls_back_to_general() -> None:
    layout = FloorLayout(
        room_id="r0",
        objects=(
            FloorObjectSpec(
                object_id="sculpture_0",
                category="sculpture",
                dimensions=(0.5, 0.5, 1.5),
                position_xy=(0.0, 0.0),
                anchor=Anchor.CORNER,  # corner counts as wall-anchored
            ),
        ),
    )
    line = derive_plan_line(layout)
    assert "zones=general" in line
    assert "sculpture:wall" in line
    assert line.endswith("n=1")
