from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Message:
    """A message instance; field values are held in canonical string form."""

    type: str
    fields: dict[str, str] = field(default_factory=dict)
