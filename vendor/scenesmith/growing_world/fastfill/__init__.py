"""FastFill: task-conditioned one-pass room content generation (WP1).

See ``README.md`` in this package for the coordinate contract and module map.
Import surface is intentionally small; heavy pieces (generator, validator)
are imported lazily by callers to keep the growth loop light.
"""

from scenesmith.growing_world.fastfill.schema import (  # noqa: F401
    FastFillSample,
    FloorLayout,
    FloorObjectSpec,
    OutputBudget,
    ProvenanceMeta,
    RoomContentLayout,
    RoomContext,
    SupportContext,
    SupportSurfaceSpec,
    SurfaceObjectGroup,
    SurfaceObjectSpec,
    ValidatedTaskEvidence,
    ValidationReport,
    Violation,
)
