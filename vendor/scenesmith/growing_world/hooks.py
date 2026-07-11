"""Pluggable content-generation hooks (v1 placeholders).

v1 generates empty-shell rooms only. The furniture / wall-mounted /
ceiling-mounted / manipuland stages are intentionally left as no-op hooks so
they can later be wired back to scenesmith's agents without restructuring the
growth loop.

To re-enable a stage later, implement ``populate_room`` to build a
``RoomScene`` for the room and run the corresponding scenesmith agent, mirroring
``experiments/indoor_scene_generation.py::_generate_room``:

    furniture_agent = BaseExperiment.build_furniture_agent(...)
    asyncio.run(furniture_agent.add_furniture(scene=scene))
    # then wall_agent / ceiling_agent / manipuland_agent ...

The growth loop calls ``populate_room`` right after a room's shell geometry is
generated.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scenesmith.agent_utils.house import RoomGeometry

console_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PopulateContext:
    """Room-shell context handed to content hooks (T1.1).

    Everything the growth loop already knows about a room at populate time,
    so content backends (e.g. fastfill) can compile door/window coordinates,
    task and neighbours into their conditioning. ``placed_room`` is a
    ``PlacedRoom`` but typed ``Any`` so importing this module stays light
    (no Drake-bound import chain).
    """

    task: str | None = None
    motivation: str | None = None
    parent_room_id: str | None = None
    placed_room: Any = None  # PlacedRoom: walls carry door/window openings
    neighbors: tuple[tuple[str, str, str], ...] = field(default_factory=tuple)
    """(room_id, room_type, direction) per occupied adjacent cell."""


class ContentHooks:
    """No-op content hooks for v1. Subclass / replace to add real content."""

    def populate_room(
        self,
        *,
        room_id: str,
        room_type: str,
        room_geometry: "RoomGeometry",
        prompt: str,
        context: PopulateContext | None = None,
    ) -> None:
        """Hook invoked after a room's shell geometry is built.

        ``context`` (T1.1) is optional and defaults to ``None`` so existing
        implementations keep working; fastfill compiles it into RoomContext.

        v1: no-op. Later: build a RoomScene and run scenesmith furniture/wall/
        ceiling/manipuland agents here.
        """
        console_logger.info(
            "[hook] populate_room(%s, type=%s) — v1 no-op (content stage skipped)",
            room_id,
            room_type,
        )
