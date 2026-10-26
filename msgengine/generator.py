"""Seeded synthetic message generator and text corruption, driven by the catalogue."""
from __future__ import annotations

import random
import string
from decimal import Decimal

from .catalogue import MONTHS, Catalogue, Field
from .model import Message

NOISE_KINDS = ("truncate", "drop_terminator", "substitute", "extra_whitespace", "shuffle_fields",
               "drop_segment", "duplicate_segment")


def _value(f: Field, rng: random.Random) -> str:
    if f.type in ("token", "text"):
        if f.examples:
            return rng.choice(f.examples)
        if f.pattern:
            raise ValueError(f"{f.name}: custom pattern needs examples")
        return "".join(rng.choices(string.ascii_uppercase + string.digits, k=rng.randint(3, 8)))
    if f.type == "enum":
        return rng.choice(f.values)
    if f.type == "int":
        lo = int(f.minimum) if f.minimum is not None else 0
        hi = int(f.maximum) if f.maximum is not None else 1000
        return str(rng.randint(lo, hi))
    if f.type == "decimal":
        lo = int(f.minimum.scaleb(f.scale)) if f.minimum is not None else 0
        hi = int(f.maximum.scaleb(f.scale)) if f.maximum is not None else 10 ** (f.scale + 3)
        return format(Decimal(rng.randint(lo, hi)).scaleb(-f.scale).quantize(Decimal(1).scaleb(-f.scale)), "f")
    if f.type == "dtg":
        return f"{rng.randint(1, 28):02d}{rng.randint(0, 23):02d}{rng.randint(0, 59):02d}Z{rng.choice(MONTHS)}26"
    zone = rng.randint(10, 19)
    band, col, row = rng.choice("STU"), rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ"), rng.choice("ABCDEFGHJKLMNPQRSTUV")
    digits = rng.choice((6, 8, 10))
    return f"{zone}{band}{col}{row}{rng.randrange(10 ** digits):0{digits}d}"


def make_message(cat: Catalogue, rng: random.Random, msg_type: str | None = None, p_optional: float = 0.6) -> Message:
    mdef = cat.message(msg_type or rng.choice(sorted(cat.messages)))
    fields: dict[str, str] = {}
    for f in mdef.fields:
        if f.required or rng.random() < p_optional:
            fields[f.name] = f.normalize(_value(f, rng))
    return Message(mdef.id, fields)


def corrupt(text: str, kind: str, rng: random.Random) -> str:
    segs = text[:-2].split("/") if text.endswith("//") else text.split("/")
    if kind == "truncate":
        return text[:rng.randint(len(text) // 4, max(len(text) // 4, len(text) - 3))]
    if kind == "drop_terminator":
        return text[:-2]
    if kind == "substitute":
        i = rng.choice([k for k, c in enumerate(text) if c not in "/="])
        new = rng.choice([c for c in string.ascii_uppercase + string.digits if c != text[i]])
        return text[:i] + new + text[i + 1:]
    if kind == "extra_whitespace":
        body = text[:-2] if text.endswith("//") else text
        return body.replace("/", " / ").replace("=", " = ") + ("//" if text.endswith("//") else "")
    if kind == "shuffle_fields":
        body = segs[1:]
        rng.shuffle(body)
        return "/".join([segs[0], *body]) + "//"
    if kind == "drop_segment" and len(segs) > 1:
        del segs[rng.randrange(1, len(segs))]
        return "/".join(segs) + "//"
    if kind == "duplicate_segment" and len(segs) > 1:
        i = rng.randrange(1, len(segs))
        segs.insert(i, segs[i])
        return "/".join(segs) + "//"
    raise ValueError(f"unknown noise kind {kind}")

