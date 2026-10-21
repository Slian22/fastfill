"""Bounded unknown-size comparison protocol; no training or inference side effects."""

from .contract import bind_room, messages, validate_output

__all__ = ["bind_room", "messages", "validate_output"]
