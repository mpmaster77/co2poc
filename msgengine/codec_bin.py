"""Catalogue-aware compact binary frame with a reconstruct-and-verify check.

Frame: [catalogue tag][message code][presence bitmap for optional fields][fields...][crc16]
The CRC covers the canonical character rendering, so the receiver rebuilds the full
message, re-validates every field and checks it matches what the sender encoded.
"""
from __future__ import annotations

import zlib
from decimal import Decimal

from .catalogue import MONTHS, Catalogue, Field
from .codec_char import render
from .model import Message

_BAND = "CDEFGHJKLMNPQRSTUVWX"
_COL = "ABCDEFGHJKLMNPQRSTUVWXYZ"
_ROW = "ABCDEFGHJKLMNPQRSTUV"


class BinaryError(ValueError):
    pass


def _uvarint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _zigzag(n: int) -> int:
    return (n << 1) ^ (n >> 63) if n >= 0 else ((-n) << 1) - 1


def _unzigzag(n: int) -> int:
    return (n >> 1) if not n & 1 else -((n + 1) >> 1)


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.d, self.i = data, 0

    def take(self, n: int) -> bytes:
        if self.i + n > len(self.d):
            raise BinaryError("truncated frame")
        chunk = self.d[self.i:self.i + n]
        self.i += n
        return chunk

    def uvarint(self) -> int:
        shift = result = 0
        for _ in range(10):
            b = self.take(1)[0]
            result |= (b & 0x7F) << shift
            if not b & 0x80:
                return result
            shift += 7
        raise BinaryError("varint too long")


def _crc(cat: Catalogue, msg: Message) -> bytes:
    return (zlib.crc32(render(cat, msg).encode("utf-8")) & 0xFFFF).to_bytes(2, "big")


def _enc_field(f: Field, value: str) -> bytes:
    if f.type in ("token", "text"):
        raw = value.encode("utf-8")
        return _uvarint(len(raw)) + raw
    if f.type == "enum":
        return _uvarint(f.values.index(value))
    if f.type == "int":
        n = int(value)
        return _uvarint(n - int(f.minimum)) if f.minimum is not None else _uvarint(_zigzag(n))
    if f.type == "decimal":
        scaled = int(Decimal(value).scaleb(f.scale))
        if f.minimum is not None:
            return _uvarint(scaled - int(f.minimum.scaleb(f.scale)))
        return _uvarint(_zigzag(scaled))
    if f.type == "dtg":
        day, hour, minute = int(value[0:2]), int(value[2:4]), int(value[4:6])
        mon, yy = MONTHS.index(value[7:10]), int(value[10:12])
        packed = (((yy * 12 + mon) * 32 + day) * 32 + hour) * 64 + minute
        return packed.to_bytes(4, "big")
    zone_len = 2 if value[1].isdigit() else 1
    zone, band, col, row = int(value[:zone_len]), value[zone_len], value[zone_len + 1], value[zone_len + 2]
    digits = value[zone_len + 3:]
    letters = (_BAND.index(band) * 24 + _COL.index(col)) * 20 + _ROW.index(row)
    head = ((zone * 8 + len(digits) // 2) * 9600 + letters).to_bytes(3, "big")
    return head + _uvarint(int(digits))


def _dec_field(f: Field, r: _Reader) -> str:
    if f.type in ("token", "text"):
        try:
            return r.take(r.uvarint()).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BinaryError("bad utf-8") from exc
    if f.type == "enum":
        idx = r.uvarint()
        if idx >= len(f.values):
            raise BinaryError("enum index out of range")
        return f.values[idx]
    if f.type == "int":
        v = r.uvarint()
        return str(v + int(f.minimum)) if f.minimum is not None else str(_unzigzag(v))
    if f.type == "decimal":
        v = r.uvarint()
        scaled = v + int(f.minimum.scaleb(f.scale)) if f.minimum is not None else _unzigzag(v)
        return format(Decimal(scaled).scaleb(-f.scale).quantize(Decimal(1).scaleb(-f.scale)), "f")
    if f.type == "dtg":
        n = int.from_bytes(r.take(4), "big")
        n, minute = divmod(n, 64)
        n, hour = divmod(n, 32)
        n, day = divmod(n, 32)
        yy, mon = divmod(n, 12)
        return f"{day:02d}{hour:02d}{minute:02d}Z{MONTHS[mon]}{yy:02d}"
    n = int.from_bytes(r.take(3), "big")
    zl, letters = divmod(n, 9600)
    zone, dcount = divmod(zl, 8)
    rest, row = divmod(letters, 20)
    band, col = divmod(rest, 24)
    if band >= len(_BAND) or not 1 <= dcount <= 5:
        raise BinaryError("bad MGRS")
    num = r.uvarint()
    return f"{zone}{_BAND[band]}{_COL[col]}{_ROW[row]}{num:0{dcount * 2}d}"


def encode(cat: Catalogue, msg: Message) -> bytes:
    mdef = cat.message(msg.type)
    optional = [f for f in mdef.fields if not f.required]
    bitmap = 0
    for i, f in enumerate(optional):
        if f.name in msg.fields:
            bitmap |= 1 << i
    out = bytearray([cat.tag, mdef.code])
    out += bitmap.to_bytes((len(optional) + 7) // 8, "big")
    for f in mdef.fields:
        if f.required or f.name in msg.fields:
            out += _enc_field(f, msg.fields[f.name])
    return bytes(out) + _crc(cat, msg)


def decode(cat: Catalogue, data: bytes) -> Message:
    """Rebuild the message, re-validate every field, and verify the integrity check."""
    if len(data) < 4:
        raise BinaryError("frame too short")
    body, crc = data[:-2], data[-2:]
    r = _Reader(body)
    if r.take(1)[0] != cat.tag:
        raise BinaryError("catalogue tag mismatch")
    mdef = cat.by_code(r.take(1)[0])
    if mdef is None:
        raise BinaryError("unknown message code")
    optional = [f for f in mdef.fields if not f.required]
    bitmap = int.from_bytes(r.take((len(optional) + 7) // 8), "big")
    present = {f.name for i, f in enumerate(optional) if bitmap >> i & 1}
    fields: dict[str, str] = {}
    for f in mdef.fields:
        if f.required or f.name in present:
            try:
                fields[f.name] = f.normalize(_dec_field(f, r))
            except ValueError as exc:
                raise BinaryError(f"{f.name}: {exc}") from exc
    if r.i != len(body):
        raise BinaryError("trailing bytes")
    msg = Message(mdef.id, fields)
    if _crc(cat, msg) != crc:
        raise BinaryError("integrity check failed")
    return msg
