"""Synthetic update streams: tracks that send a baseline report and then periodic changes."""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from decimal import Decimal

from .catalogue import MONTHS, Catalogue, Field
from .generator import make_message
from .link import Item
from .model import Message

_KEEP = {"token", "text"}


def _advance_dtg(value: str, minutes: int) -> str:
    t = datetime(2000 + int(value[10:12]), MONTHS.index(value[7:10]) + 1, int(value[0:2]),
                 int(value[2:4]), int(value[4:6])) + timedelta(minutes=minutes)
    return f"{t.day:02d}{t.hour:02d}{t.minute:02d}Z{MONTHS[t.month - 1]}{t.year - 2000:02d}"


def _jitter(f: Field, value: str, rng: random.Random) -> str:
    if f.type == "enum":
        return rng.choice([v for v in f.values if v != value] or [value])
    if f.type in ("int", "decimal"):
        step = Decimal(1).scaleb(-f.scale) * rng.randint(1, 30)
        new = Decimal(value) + step * rng.choice((-1, 1))
        if f.minimum is not None:
            new = max(new, f.minimum)
        if f.maximum is not None:
            new = min(new, f.maximum)
        return f.normalize(format(new, "f"))
    if f.type == "mgrs":
        return _shift_mgrs(value, rng)
    return value


def _shift_mgrs(value: str, rng: random.Random) -> str:
    i = len(value)
    while value[i - 1].isdigit():
        i -= 1
    digits = value[i:]
    half = len(digits) // 2
    e = min(10 ** half - 1, max(0, int(digits[:half]) + rng.randint(-5, 5)))
    n = min(10 ** half - 1, max(0, int(digits[half:]) + rng.randint(-5, 5)))
    return f"{value[:i]}{e:0{half}d}{n:0{half}d}"


def update(cat: Catalogue, msg: Message, rng: random.Random, minutes: int = 1) -> Message:
    """A follow-up report: the time moves on and one or two non-identity fields change."""
    mdef = cat.message(msg.type)
    fields = dict(msg.fields)
    if "DTG" in fields:
        fields["DTG"] = _advance_dtg(fields["DTG"], minutes)
    candidates = [f for f in mdef.fields if f.name in fields and f.type not in _KEEP and f.name != "DTG"]
    for f in rng.sample(candidates, k=min(len(candidates), rng.randint(1, 2))):
        fields[f.name] = _jitter(f, fields[f.name], rng)
    return Message(msg.type, {k: mdef.field(k).normalize(v) for k, v in fields.items()})


def make_stream(cat: Catalogue, seed: int, tracks: int = 12, updates: int = 8, interval_s: float = 45.0) -> list[Item]:
    rng = random.Random(seed)
    types = sorted(cat.messages)
    items: list[Item] = []
    for track in range(tracks):
        msg = make_message(cat, rng, types[track % len(types)], p_optional=0.8)
        t = rng.uniform(0, interval_s)
        for _ in range(updates + 1):
            items.append(Item(track, t, msg))
            msg = update(cat, msg, rng)
            t += interval_s * rng.uniform(0.8, 1.2)
    return items
