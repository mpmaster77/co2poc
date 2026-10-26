"""Character-oriented, slash-delimited message text.

Grammar (a synthetic format, not USMTF/ADatP-3):

    TYPE/NAME=value/NAME=value//

Line breaks are ignored. Whitespace around names and values is trimmed.
Inside values, "/" "=" "~" are escaped as "~S" "~E" "~T".
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .catalogue import Catalogue
from .model import Message

_ESC = {"/": "~S", "=": "~E", "~": "~T"}
_UNESC = {"S": "/", "E": "=", "T": "~"}


@dataclass
class Issue:
    code: str
    field: str | None
    detail: str


@dataclass
class ParseResult:
    message: Message | None
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.message is not None and not self.issues


def escape(value: str) -> str:
    return "".join(_ESC.get(c, c) for c in value)


def unescape(value: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(value):
        c = value[i]
        if c == "~":
            if i + 1 >= len(value) or value[i + 1] not in _UNESC:
                raise ValueError("bad escape sequence")
            out.append(_UNESC[value[i + 1]])
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def render(cat: Catalogue, msg: Message) -> str:
    mdef = cat.message(msg.type)
    parts = [msg.type]
    for f in mdef.fields:
        if f.name in msg.fields:
            parts.append(f"{f.name}={escape(msg.fields[f.name])}")
    return "/".join(parts) + "//"


def parse(cat: Catalogue, text: str) -> ParseResult:
    """Parse text; never raises. Returns a partial message plus issues where possible."""
    issues: list[Issue] = []
    t = text.replace("\r", "").replace("\n", "").strip()
    if t.endswith("//"):
        t = t[:-2]
    else:
        issues.append(Issue("TRUNCATED", None, "missing '//' terminator"))
    parts = t.split("/")
    msg_type = parts[0].strip()
    if msg_type not in cat.messages:
        issues.append(Issue("UNKNOWN_TYPE", None, f"unknown message type {msg_type!r}"))
        return ParseResult(None, issues)
    mdef = cat.message(msg_type)
    fields: dict[str, str] = {}
    for seg in parts[1:]:
        if not seg.strip():
            issues.append(Issue("EMPTY_SEGMENT", None, "empty segment"))
            continue
        name, eq, raw = seg.partition("=")
        name, raw = name.strip(), raw.strip()
        if not eq:
            issues.append(Issue("MALFORMED_FIELD", name, "expected NAME=value"))
            continue
        fdef = mdef.field(name)
        if fdef is None:
            issues.append(Issue("UNKNOWN_FIELD", name, "not in catalogue"))
            continue
        if name in fields:
            issues.append(Issue("DUPLICATE_FIELD", name, "repeated; first value kept"))
            continue
        try:
            fields[name] = fdef.normalize(unescape(raw))
        except ValueError as exc:
            issues.append(Issue("INVALID_VALUE", name, str(exc)))
    for f in mdef.fields:
        if f.required and f.name not in fields:
            issues.append(Issue("MISSING_REQUIRED", f.name, "required field absent or invalid"))
    return ParseResult(Message(msg_type, fields), issues)
