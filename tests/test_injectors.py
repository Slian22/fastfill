"""Tests for the DPO stage-2 violation injectors.

Builds synthetic FastFillSamples in code (vendored schema) and checks, for
every injector, that the perturbed layout fails the vendored validator with
an expected code while the clean layout produces none of those codes.
No torch / trl / datasets required.
"""

from __future__ import annotations

import pytest

import fastfill_train  # noqa: F401  (vendor path bootstrap)
from scenesmith.growing_world.fastfill.schema import (
    Anchor,
    DoorSpec,
    FastFillSample,
    FloorLayout,
    FloorObjectSpec,
    LicenseTag,
    ProvenanceMeta,
    RelationKind,
    RoomContentLayout,
    RoomContext,
    SemanticRelation,
    SupportSurfaceSpec,
    SurfaceKind,
    SurfaceObjectGroup,
    SurfaceObjectSpec,
)
from scenesmith.growing_world.fastfill.validator import validate

from fastfill_train.injectors import (
    ALL_INJECTORS,
    run_injector,
    verify_injection,
)


def make_clean_sample(
    sample_id: str = "s0", split_key: str = "houseA"
) -> FastFillSample:
    """A validator-clean bedroom sample on which every injector applies.

    6x6 m room centered at the origin, a south door, one facing relation
    (chair -> desk), two support surfaces on the desk (big top, tiny shelf)
    and one book on the desk top.
    """
    room_id = f"room_{sample_id}"
    context = RoomContext(
        room_id=room_id,
        room_type="bedroom",
        floor_polygon=((-3.0, -3.0), (3.0, -3.0), (3.0, 3.0), (-3.0, 3.0)),
        doors=(
            DoorSpec(
                door_id="d0",
                center_xy=(0.0, -3.0),
                width_m=0.9,
                clearance_depth_m=0.9,
                wall="south",
            ),
        ),
        expected_furniture=("bed",),
    )
    bed = FloorObjectSpec(
        object_id="bed_0",
        category="bed",
        dimensions=(2.0, 1.6, 0.5),
        position_xy=(-1.5, 2.0),
        anchor=Anchor.WALL,
    )
    desk = FloorObjectSpec(
        object_id="desk_0",
        category="desk",
        dimensions=(1.2, 0.6, 0.75),
        position_xy=(1.8, 2.4),
        anchor=Anchor.WALL,
        wants_surface_fill=(SurfaceKind.TOP,),
    )
    chair = FloorObjectSpec(
        object_id="chair_0",
        category="chair",
        dimensions=(0.5, 0.5, 0.9),
        position_xy=(1.8, 1.2),  # yaw 0 -> faces +Y, toward the desk
    )
    desk_top = SupportSurfaceSpec(
        surface_id="desk_0_top",
        parent_object_id="desk_0",
        kind=SurfaceKind.TOP,
        height_m=0.75,
        polygon_local=((-0.6, -0.3), (0.6, -0.3), (0.6, 0.3), (-0.6, 0.3)),
        capacity_max_objects=2,
    )
    desk_shelf = SupportSurfaceSpec(
        surface_id="desk_0_shelf",
        parent_object_id="desk_0",
        kind=SurfaceKind.SHELF,
        height_m=1.2,
        polygon_local=(
            (-0.05, -0.05),
            (0.05, -0.05),
            (0.05, 0.05),
            (-0.05, 0.05),
        ),
        capacity_max_objects=2,
    )
    book = SurfaceObjectSpec(
        object_id="g0/book_0",
        category="book",
        dimensions=(0.2, 0.15, 0.03),
        position_local=(0.2, 0.0),
    )
    layout = RoomContentLayout(
        room_id=room_id,
        floor_layout=FloorLayout(
            room_id=room_id,
            objects=(bed, desk, chair),
            support_surfaces=(desk_top, desk_shelf),
        ),
        surface_groups=(
            SurfaceObjectGroup(
                group_id="g0", surface_id="desk_0_top", objects=(book,)
            ),
        ),
        semantic_relations=(
            SemanticRelation(
                subject_id="chair_0",
                relation=RelationKind.FACING,
                object_id="desk_0",
            ),
        ),
    )
    provenance = ProvenanceMeta(
        source_dataset="synthetic",
        split_key=split_key,
        license_tag=LicenseTag.PERMISSIVE,
        notes="sanitized=v1",  # stage2 hard-gates on the sanitizer marker
    )
    return FastFillSample(
        sample_id=sample_id,
        room_context=context,
        layout=layout,
        provenance=provenance,
    )


def test_clean_sample_passes_validation() -> None:
    sample = make_clean_sample()

    report = validate(sample.layout, sample.room_context)

    assert report.passed, [v.code for v in report.violations]


@pytest.mark.parametrize("name", sorted(ALL_INJECTORS))
def test_injector_produces_expected_code_absent_from_clean_layout(
    name: str,
) -> None:
    # Arrange
    sample = make_clean_sample()

    # Act
    injection = run_injector(name, sample)

    # Assert
    assert injection is not None, f"{name} not applicable to the clean sample"
    expected = set(injection.expected_codes)
    assert expected, f"{name} returned no expected codes"
    bad_codes = {
        v.code
        for v in validate(injection.bad_layout, sample.room_context).violations
    }
    clean_codes = {
        v.code for v in validate(sample.layout, sample.room_context).violations
    }
    assert expected & bad_codes, f"{name}: none of {expected} in {bad_codes}"
    assert not (expected & clean_codes), f"{name}: clean layout already fails"


@pytest.mark.parametrize("name", sorted(ALL_INJECTORS))
def test_injector_never_mutates_the_input_sample(name: str) -> None:
    sample = make_clean_sample()
    before = sample.model_copy(deep=True)

    run_injector(name, sample)

    assert sample == before


def test_blocked_door_inapplicable_without_doors() -> None:
    sample = make_clean_sample()
    context = sample.room_context.model_copy(update={"doors": ()})
    sample = sample.model_copy(update={"room_context": context})

    assert run_injector("blocked_door", sample) is None


def test_wrong_facing_skips_samples_without_facing_relations() -> None:
    sample = make_clean_sample()
    layout = sample.layout.model_copy(update={"semantic_relations": ()})
    sample = sample.model_copy(update={"layout": layout})

    assert run_injector("wrong_facing", sample) is None


def test_collision_pair_needs_two_floor_objects() -> None:
    sample = make_clean_sample()
    floor = sample.layout.floor_layout
    one_object = floor.model_copy(
        update={"objects": floor.objects[:1], "support_surfaces": ()}
    )
    layout = sample.layout.model_copy(
        update={"floor_layout": one_object, "surface_groups": ()}
    )
    sample = sample.model_copy(update={"layout": layout})

    assert run_injector("collision_pair", sample) is None


def test_missing_required_needs_expected_furniture() -> None:
    sample = make_clean_sample()
    context = sample.room_context.model_copy(update={"expected_furniture": ()})
    sample = sample.model_copy(update={"room_context": context})

    assert run_injector("missing_required", sample) is None


def test_wrong_parent_needs_two_surfaces() -> None:
    sample = make_clean_sample()
    floor = sample.layout.floor_layout
    single_surface = floor.model_copy(
        update={"support_surfaces": floor.support_surfaces[:1]}
    )
    layout = sample.layout.model_copy(update={"floor_layout": single_surface})
    sample = sample.model_copy(update={"layout": layout})

    assert run_injector("wrong_parent", sample) is None


def test_surface_injectors_inapplicable_without_surface_objects() -> None:
    sample = make_clean_sample()
    layout = sample.layout.model_copy(update={"surface_groups": ()})
    sample = sample.model_copy(update={"layout": layout})

    for name in (
        "floating_surface_object",
        "surface_off_polygon",
        "wrong_parent",
        "surface_overflow",
    ):
        assert run_injector(name, sample) is None, name


def test_run_injector_rejects_pairs_when_original_already_violates() -> None:
    # Arrange: bed already out of bounds in the ORIGINAL layout.
    sample = make_clean_sample()
    bed = sample.layout.floor_layout.objects[0]
    dirty_bed = bed.model_copy(update={"position_xy": (6.0, 2.0)})
    objects = (dirty_bed,) + sample.layout.floor_layout.objects[1:]
    floor = sample.layout.floor_layout.model_copy(update={"objects": objects})
    layout = sample.layout.model_copy(update={"floor_layout": floor})
    sample = sample.model_copy(update={"layout": layout})

    # Act / Assert: the pair would be noise, so it is refused.
    assert run_injector("out_of_bounds", sample) is None


def test_verify_injection_rejects_wrong_expected_codes() -> None:
    sample = make_clean_sample()
    injection = ALL_INJECTORS["out_of_bounds"](sample)
    assert injection is not None
    from dataclasses import replace

    mislabeled = replace(injection, expected_codes=("L1_FLOATING",))

    assert verify_injection(sample, injection)
    assert not verify_injection(sample, mislabeled)


def test_run_injector_unknown_name_raises() -> None:
    with pytest.raises(KeyError):
        run_injector("no_such_injector", make_clean_sample())
