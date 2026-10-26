"""Delta frames: send only what changed against a baseline both ends already hold.

Frame: [catalogue tag][code | 0x80][baseline id 2B][changed bitmap][removed bitmap][changed values][crc16]
The baseline id is the 16-bit check of the baseline's canonical text. The receiver rebuilds the
full message, re-validates it, and verifies the check of the result.
"""
from __future__ import annotations

from .catalogue import Catalogue
from .codec_bin import BinaryError, _crc, _dec_field, _enc_field, _Reader
from .model import Message

DELTA_FLAG = 0x80


def is_delta(data: bytes) -> bool:
    return len(data) > 1 and bool(data[1] & DELTA_FLAG)


def baseline_id(cat: Catalogue, baseline: Message) -> bytes:
    return _crc(cat, baseline)


def frame_baseline_id(data: bytes) -> bytes:
    if not is_delta(data) or len(data) < 4:
        raise BinaryError("not a delta frame")
    return data[2:4]


def encode_delta(cat: Catalogue, baseline: Message, new: Message) -> bytes:
    if baseline.type != new.type:
        raise ValueError("baseline and new message differ in type")
    mdef = cat.message(new.type)
    for f in mdef.fields:
        if f.required and f.name not in new.fields:
            raise ValueError(f"new message lacks required field {f.name}")
    changed = removed = 0
    values = bytearray()
    for i, f in enumerate(mdef.fields):
        if f.name in new.fields:
            if baseline.fields.get(f.name) != new.fields[f.name]:
                changed |= 1 << i
                values += _enc_field(f, new.fields[f.name])
        elif f.name in baseline.fields:
            removed |= 1 << i
    nbytes = (len(mdef.fields) + 7) // 8
    return (bytes([cat.tag, mdef.code | DELTA_FLAG]) + baseline_id(cat, baseline)
            + changed.to_bytes(nbytes, "big") + removed.to_bytes(nbytes, "big") + bytes(values) + _crc(cat, new))


def decode_delta(cat: Catalogue, data: bytes, baseline: Message) -> Message:
    if len(data) < 6 or not is_delta(data):
        raise BinaryError("not a delta frame")
    body, crc = data[:-2], data[-2:]
    r = _Reader(body)
    if r.take(1)[0] != cat.tag:
        raise BinaryError("catalogue tag mismatch")
    mdef = cat.by_code(r.take(1)[0] & ~DELTA_FLAG)
    if mdef is None:
        raise BinaryError("unknown message code")
    if mdef.id != baseline.type:
        raise BinaryError("baseline type mismatch")
    if r.take(2) != baseline_id(cat, baseline):
        raise BinaryError("baseline mismatch")
    nbytes = (len(mdef.fields) + 7) // 8
    changed = int.from_bytes(r.take(nbytes), "big")
    removed = int.from_bytes(r.take(nbytes), "big")
    fields = dict(baseline.fields)
    for i, f in enumerate(mdef.fields):
        if removed >> i & 1:
            fields.pop(f.name, None)
        if changed >> i & 1:
            try:
                fields[f.name] = f.normalize(_dec_field(f, r))
            except ValueError as exc:
                raise BinaryError(f"{f.name}: {exc}") from exc
    if r.i != len(body):
        raise BinaryError("trailing bytes")
    for f in mdef.fields:
        if f.required and f.name not in fields:
            raise BinaryError(f"missing required field {f.name}")
    msg = Message(mdef.id, fields)
    if _crc(cat, msg) != crc:
        raise BinaryError("integrity check failed")
    return msg
