"""FastFill schema — the frozen data contract for one-pass room content generation.

Every model call, converter, validator and repair step in FastFill speaks the
types in this file and nothing else. The coordinate contract (Z-up, meters,
degrees, `dimensions=[width, depth, height]` full extents, furniture in the
room frame, small objects in support-surface local frames) is documented in
`README.md` next to this file and is normative.

All models are immutable (`frozen=True`) and reject unknown fields
(`extra="forbid"`), so a parsed layout is a value object: validators and
repair return *new* layouts instead of mutating.

JSON Schema for any model: ``Model.model_json_schema()``.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]


class _Frozen(BaseModel):
    """Base: immutable, strict-field pydantic model."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------- enums


class GroupPattern(str, Enum):
    """Structured placement primitive for a surface-object group (v1.4 §2.1).

    Geometry for non-FREE patterns is produced by the deterministic pattern
    expander, not by the model — the model only declares the primitive.
    """

    FREE = "free"
    PAIRED = "paired"
    MATRIX = "matrix"
    CIRCULAR = "circular"


class OverflowPolicy(str, Enum):
    """What to do when generation would exceed the output budget (§2)."""

    FAIL = "fail"
    TRUNCATE_OPTIONAL_ONLY = "truncate_optional_only"
    FALLBACK_AGENTIC = "fallback_agentic"
    DEFER_NON_TASK_SURFACES = "defer_non_task_surfaces"


class ViolationLayer(str, Enum):
    L0 = "L0"  # structural validity
    L1 = "L1"  # geometry & support
    L2 = "L2"  # function & semantics
    L3 = "L3"  # reachability & task evidence


class Severity(str, Enum):
    FATAL = "fatal"  # unusable layout (schema break, cyclic support, ...)
    ERROR = "error"  # must be repaired before the layout may pass
    WARNING = "warning"  # recorded, does not fail the layout


class SurfaceKind(str, Enum):
    TOP = "top"
    SHELF = "shelf"
    SEAT = "seat"
    COUNTER = "counter"
    INSET = "inset"  # e.g. sink basin — placement usually forbidden


class Anchor(str, Enum):
    WALL = "wall"
    CORNER = "corner"
    FREE = "free"


class RelationKind(str, Enum):
    ON = "on"
    NEAR = "near"
    FACING = "facing"
    INSIDE = "inside"
    ALIGNED_WITH = "aligned_with"
    PAIRED_WITH = "paired_with"


class EvidenceKind(str, Enum):
    OBJECT_PRESENT = "object_present"
    RELATION_HOLDS = "relation_holds"
    CLEARANCE_AVAILABLE = "clearance_available"
    REACHABLE = "reachable"


class LicenseTag(str, Enum):
    """铁律 2: HSSD-derived (CC BY-NC) content is flagged and excluded from
    training by default; unknown licenses stay pending until resolved."""

    PERMISSIVE = "permissive"  # apache/mit/cc-by
    CC_BY_NC = "cc_by_nc"
    LICENSE_PENDING = "license_pending"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------- provenance


class ProvenanceMeta(_Frozen):
    """铁律 4 lineage fields — mandatory on every persisted sample.

    ``split_key`` is house-first: samples from the same source house must
    never straddle a train/val/test split.
    """

    source_dataset: str
    source_house_id: str = ""
    source_room_id: str = ""
    source_asset_ids: tuple[str, ...] = ()
    upstream_dataset: str = ""  # e.g. "3D-FRONT" behind 3D-SynthPlace/IL3D
    geometry_hash: str = ""  # cross-source near-dup key (provenance.py)
    layout_hash: str = ""  # within-source exact identity (with yaw)
    parent_sample_id: str = ""  # for derived/augmented samples
    license_tag: LicenseTag = LicenseTag.UNKNOWN
    split_key: str = ""
    notes: str = ""  # converter caveats, e.g. "floor=synthesized_bbox"


# --------------------------------------------------------------- room context


class DoorSpec(_Frozen):
    """A doorway in the room, with the clearance zone that must stay open."""

    door_id: str
    center_xy: Vec2  # room frame, on the wall line
    width_m: float = Field(gt=0)
    clearance_depth_m: float = Field(default=0.9, ge=0)
    wall: Literal["north", "south", "east", "west", ""] = ""
    connects_room_id: str = ""


class WindowSpec(_Frozen):
    window_id: str
    center_xy: Vec2
    width_m: float = Field(gt=0)
    sill_height_m: float = Field(default=0.9, ge=0)
    height_m: float = Field(default=1.2, gt=0)
    wall: Literal["north", "south", "east", "west", ""] = ""
    keep_clear: bool = False  # True → treat like a door clearance zone


class PortalSpec(_Frozen):
    """A frontier opening to not-yet-generated space (growing-world only)."""

    portal_id: str
    center_xy: Vec2
    width_m: float = Field(gt=0)
    wall: Literal["north", "south", "east", "west", ""] = ""
    status: Literal["frontier", "connected"] = "frontier"


class NeighborSummary(_Frozen):
    room_id: str
    room_type: str
    direction: Literal["north", "south", "east", "west", ""] = ""
    via_door_id: str = ""


class ForbiddenRegion(_Frozen):
    """A floor (or surface-local) polygon nothing may be placed in."""

    region_id: str
    polygon: tuple[Vec2, ...] = Field(min_length=3)
    reason: str = ""


class OutputBudget(_Frozen):
    """Hard output limits (§2). Silent batch-splitting is forbidden; overflow
    follows ``overflow_policy`` with the pruning priority: task-required
    surfaces > expected-manipuland surfaces > functional completeness >
    decorative clutter. Task evidence is never pruned."""

    max_support_surfaces_per_room: int = Field(default=12, gt=0)
    max_surface_objects_per_surface: int = Field(default=12, gt=0)
    max_surface_objects_total: int = Field(default=60, gt=0)
    max_output_tokens_floor: int = Field(default=1200, gt=0)
    max_output_tokens_surface: int = Field(default=2400, gt=0)
    overflow_policy: OverflowPolicy = OverflowPolicy.TRUNCATE_OPTIONAL_ONLY


class RoomContext(_Frozen):
    """Everything call 1 (floor generation) is conditioned on.

    Produced by a backend (v1: growing_world hook adapter; v2: worldedge slot
    materializer) — both compile to this same schema. ``floor_polygon`` is in
    the room frame with its centroid at the origin (converters re-center).
    """

    room_id: str
    room_type: str
    floor_polygon: tuple[Vec2, ...] = Field(min_length=3)
    ceiling_height_m: float = Field(default=2.5, gt=0)
    doors: tuple[DoorSpec, ...] = ()
    windows: tuple[WindowSpec, ...] = ()
    portals: tuple[PortalSpec, ...] = ()
    neighbors: tuple[NeighborSummary, ...] = ()
    forbidden_regions: tuple[ForbiddenRegion, ...] = ()
    task: str = ""  # the actionable task sentence, "" when none
    expected_furniture: tuple[str, ...] = ()
    expected_manipulands: tuple[str, ...] = ()
    style_hint: str = ""
    budget: OutputBudget = OutputBudget()
    meta: Optional[ProvenanceMeta] = None


# ---------------------------------------------------------------- floor layer


class FloorObjectSpec(_Frozen):
    """One floor-standing furniture piece (call-1 output row).

    ``asset_query`` is a retrieval/generation-neutral description, so
    retrieval and SAM3D backends stay interchangeable; ``dimensions`` is the
    requested target size — the resolver replaces it with the bound asset's
    real size downstream.
    """

    object_id: str
    category: str
    asset_query: str = ""
    dimensions: Vec3  # [width, depth, height] full extents, meters
    position_xy: Vec2  # room frame, footprint center
    z: float = 0.0  # 0 for floor-standing
    yaw_deg: float = 0.0
    anchor: Anchor = Anchor.FREE
    functional_role: str = ""
    required_by_task: bool = False
    wants_surface_fill: tuple[SurfaceKind, ...] = ()  # surfaces call 2 should fill


class SupportSurfaceSpec(_Frozen):
    """A real (or canonical-proxy) support surface on a placed furniture piece.

    Geometry comes from the deterministic asset resolver, never from the
    model. ``polygon_local`` is in the parent furniture's local frame
    (pre-yaw), centered on the surface centroid contract of README.md.
    """

    surface_id: str
    parent_object_id: str
    kind: SurfaceKind = SurfaceKind.TOP
    height_m: float = Field(ge=0)  # surface plane above the room floor
    polygon_local: tuple[Vec2, ...] = Field(min_length=3)
    surface_normal: Vec3 = (0.0, 0.0, 1.0)
    forbidden_regions_local: tuple[ForbiddenRegion, ...] = ()
    capacity_max_objects: int = Field(default=12, gt=0)
    source: Literal["canonical", "asset"] = "canonical"


class FloorLayout(_Frozen):
    """Call-1 output: floor furniture + the surface inventory to fill.

    'Floor' = floor-standing furniture layer, NOT an architectural floor
    plan (the shell comes from the WorldEdge side as RoomContext).
    """

    room_id: str
    objects: tuple[FloorObjectSpec, ...]
    support_surfaces: tuple[SupportSurfaceSpec, ...] = ()  # resolver-filled


# -------------------------------------------------------------- surface layer


class SupportContext(_Frozen):
    """Per-surface conditioning for call 2 (sent as a batch array)."""

    surface: SupportSurfaceSpec
    parent_category: str
    parent_dimensions: Vec3
    room_type: str
    task: str = ""
    expected_manipulands_here: tuple[str, ...] = ()
    neighbor_objects: tuple[str, ...] = ()  # nearby floor-object categories
    desired_groups: tuple[str, ...] = ()  # functional groups to realize


class SurfaceObjectSpec(_Frozen):
    """One small object, in its support surface's local frame."""

    object_id: str
    category: str
    asset_query: str = ""
    dimensions: Vec3  # [width, depth, height] full extents, meters
    position_local: Vec2  # surface-local frame (see README contract)
    z_local: float = 0.0  # 0 = resting on the surface plane
    yaw_deg_local: float = 0.0  # relative to parent yaw
    functional_role: str = ""
    required_by_task: bool = False


class PatternParams(_Frozen):
    """Deterministic expansion parameters for non-FREE group patterns."""

    rows: int = Field(default=0, ge=0)
    cols: int = Field(default=0, ge=0)
    spacing_x_m: float = Field(default=0.0, ge=0)
    spacing_y_m: float = Field(default=0.0, ge=0)
    radius_m: float = Field(default=0.0, ge=0)


class SemanticRelation(_Frozen):
    subject_id: str
    relation: RelationKind
    object_id: str


class SurfaceObjectGroup(_Frozen):
    """Call-2 output for one support surface: a functional group of small
    objects, optionally structured by a pattern primitive (v1.4 §2.1)."""

    group_id: str
    surface_id: str
    pattern: GroupPattern = GroupPattern.FREE
    pattern_params: PatternParams = PatternParams()
    anchor_object_id: str = ""  # pattern anchor, when pattern != FREE
    objects: tuple[SurfaceObjectSpec, ...] = ()
    intra_relations: tuple[SemanticRelation, ...] = ()


# ----------------------------------------------------------- aggregate layout


class TaskEvidenceClaim(_Frozen):
    """Generator-side claim ONLY (ids + references). The validator rebuilds
    evidence independently; self-reported claims never count toward pass."""

    claim_id: str
    task_requirement_id: str
    kind: EvidenceKind
    object_ids: tuple[str, ...] = ()
    relation: Optional[SemanticRelation] = None
    note: str = ""


class RoomContentLayout(_Frozen):
    """The complete FastFill product for one room."""

    room_id: str
    floor_layout: FloorLayout
    surface_groups: tuple[SurfaceObjectGroup, ...] = ()
    semantic_relations: tuple[SemanticRelation, ...] = ()
    task_evidence_claims: tuple[TaskEvidenceClaim, ...] = ()
    meta: Optional[ProvenanceMeta] = None


# ------------------------------------------------------------------ reporting


class Violation(_Frozen):
    """One validator finding, localized so repair can target it."""

    code: str  # e.g. "L1_FLOOR_OUT_OF_BOUNDS"
    layer: ViolationLayer
    severity: Severity
    message: str
    object_ids: tuple[str, ...] = ()
    surface_id: str = ""  # set for surface-localized violations
    details: str = ""


class ValidatedTaskEvidence(_Frozen):
    """Validator-rebuilt evidence (existence, containment, relations,
    reachability) — the only evidence that counts (§2.1)."""

    claim_id: str
    task_requirement_id: str
    verified: bool
    method: str  # which check produced this verdict
    detail: str = ""
    object_ids: tuple[str, ...] = ()


class ValidationReport(_Frozen):
    room_id: str
    passed: bool
    violations: tuple[Violation, ...] = ()
    validated_task_evidence: tuple[ValidatedTaskEvidence, ...] = ()
    checks_run: int = 0
    wall_time_ms: float = 0.0

    def by_layer(self, layer: ViolationLayer) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.layer is layer)

    def errors(self) -> tuple[Violation, ...]:
        return tuple(
            v for v in self.violations if v.severity in (Severity.ERROR, Severity.FATAL)
        )


# -------------------------------------------------------------------- dataset


class FastFillSample(_Frozen):
    """One normalized training/eval record emitted by the data converters.

    ``room_context`` is the model input side; ``layout`` is the label side.
    Surface groups may be empty (floor-only sources like 3D-SynthPlace).
    """

    sample_id: str
    room_context: RoomContext
    layout: RoomContentLayout
    provenance: ProvenanceMeta


ALL_SCHEMAS: tuple[type[BaseModel], ...] = (
    RoomContext,
    FloorLayout,
    SupportContext,
    SurfaceObjectGroup,
    RoomContentLayout,
    Violation,
    ValidationReport,
    ValidatedTaskEvidence,
    FastFillSample,
)
