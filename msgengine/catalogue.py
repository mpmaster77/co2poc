"""Catalogue: message formats defined as data.

All patterns are XSD-compatible and matched against the whole value (no ^ or $).
"""
from __future__ import annotations

import calendar
import hashlib
import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any

MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
DTG_PATTERN = r"(0[1-9]|[12][0-9]|3[01])([01][0-9]|2[0-3])[0-5][0-9]Z(" + "|".join(MONTHS) + r")[0-9]{2}"
MGRS_PATTERN = r"[0-9]{1,2}[C-HJ-NP-X][A-HJ-NP-Z][A-HJ-NP-V]([0-9]{2}|[0-9]{4}|[0-9]{6}|[0-9]{8}|[0-9]{10})"
TOKEN_DEFAULT_PATTERN = r"[A-Za-z0-9_.\-]+"
NAME_RE = re.compile(r"[A-Z][A-Z0-9_]*")
TYPES = ("token", "text", "enum", "int", "decimal", "dtg", "mgrs")
_FIELD_KEYS = {"name", "type", "required", "max_len", "pattern", "values", "min", "max", "scale", "examples",
               "cues", "synonyms", "units", "hemis"}
_MSG_KEYS = {"id", "code", "title", "domain", "service", "fields", "keywords", "fusion"}
_FUSION_KEYS = {"identity", "id_any", "spatial", "radius_m", "time_window_s", "must_match", "soft_match",
                "vote_fields", "supersede", "default_speed_mps", "course_field", "speed_field", "time_resolution_s"}


class CatalogueError(ValueError):
    pass


@lru_cache(maxsize=256)
def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


@dataclass(frozen=True)
class Field:
    name: str
    type: str
    required: bool = False
    max_len: int | None = None
    pattern: str | None = None
    values: tuple[str, ...] = ()
    minimum: Decimal | None = None
    maximum: Decimal | None = None
    scale: int = 0
    examples: tuple[str, ...] = ()
    cues: tuple[str, ...] = ()
    synonyms: tuple[tuple[str, tuple[str, ...]], ...] = ()
    units: tuple[str, ...] = ()
    hemis: str = ""

    def normalize(self, value: str) -> str:
        """Return the canonical form of value, or raise ValueError."""
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        t = self.type
        if t == "token":
            pattern = self.pattern or TOKEN_DEFAULT_PATTERN
            if not value or len(value) > (self.max_len or 32):
                raise ValueError("bad length")
            if not _rx(pattern).fullmatch(value):
                raise ValueError("does not match pattern")
            return value
        if t == "text":
            if not value or len(value) > (self.max_len or 120):
                raise ValueError("bad length")
            if value != value.strip():
                raise ValueError("leading or trailing whitespace")
            if not value.isprintable():
                raise ValueError("control characters")
            return value
        if t == "enum":
            if value not in self.values:
                raise ValueError("not an allowed value")
            return value
        if t == "int":
            if not _rx(r"-?[0-9]+").fullmatch(value):
                raise ValueError("not an integer")
            n = int(value)
            self._check_range(Decimal(n))
            return str(n)
        if t == "decimal":
            if not _rx(r"-?[0-9]+(\.[0-9]+)?").fullmatch(value):
                raise ValueError("not a decimal")
            if "." in value and len(value.split(".")[1]) > self.scale:
                raise ValueError("too many decimal places")
            try:
                d = Decimal(value)
            except InvalidOperation as exc:
                raise ValueError("not a decimal") from exc
            self._check_range(d)
            d = d.quantize(Decimal(1).scaleb(-self.scale))
            if d == 0:
                d = abs(d)
            return format(d, "f")
        if t == "dtg":
            m = _rx(DTG_PATTERN).fullmatch(value)
            if not m:
                raise ValueError("not a DTG (DDHHMMZMONYY)")
            day, mon, yy = int(value[0:2]), MONTHS.index(value[7:10]) + 1, int(value[10:12])
            if day > calendar.monthrange(2000 + yy, mon)[1]:
                raise ValueError("day not valid for month")
            return value
        if t == "mgrs":
            if not _rx(MGRS_PATTERN).fullmatch(value):
                raise ValueError("not an MGRS reference")
            zone = int(re.match(r"[0-9]+", value).group())
            if not 1 <= zone <= 60:
                raise ValueError("zone out of range")
            return value
        raise ValueError(f"unknown type {t}")

    def _check_range(self, d: Decimal) -> None:
        if self.minimum is not None and d < self.minimum:
            raise ValueError("below minimum")
        if self.maximum is not None and d > self.maximum:
            raise ValueError("above maximum")


@dataclass(frozen=True)
class MessageDef:
    id: str
    code: int
    title: str
    domain: str
    service: str
    fields: tuple[Field, ...]
    keywords: tuple[str, ...] = ()
    fusion: dict | None = None

    def field(self, name: str) -> Field | None:
        for f in self.fields:
            if f.name == name:
                return f
        return None


@dataclass
class Catalogue:
    name: str
    version: str
    namespace: str
    messages: dict[str, MessageDef]
    tag: int
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def message(self, msg_id: str) -> MessageDef:
        try:
            return self.messages[msg_id]
        except KeyError:
            raise CatalogueError(f"unknown message type {msg_id!r}") from None

    def by_code(self, code: int) -> MessageDef | None:
        for m in self.messages.values():
            if m.code == code:
                return m
        return None

    @classmethod
    def load(cls, path: str | Path) -> "Catalogue":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Catalogue":
        for key in ("name", "version", "namespace", "messages"):
            if key not in data:
                raise CatalogueError(f"missing top-level key {key!r}")
        messages: dict[str, MessageDef] = {}
        codes: set[int] = set()
        for md in data["messages"]:
            unknown = set(md) - _MSG_KEYS
            if unknown:
                raise CatalogueError(f"unknown message keys {sorted(unknown)}")
            mid, code = md.get("id", ""), md.get("code")
            if not NAME_RE.fullmatch(mid):
                raise CatalogueError(f"bad message id {mid!r}")
            if mid in messages:
                raise CatalogueError(f"duplicate message id {mid}")
            if not isinstance(code, int) or not 1 <= code <= 127 or code in codes:
                raise CatalogueError(f"{mid}: code must be a unique integer 1-127")
            codes.add(code)
            fields = tuple(_parse_field(mid, fd) for fd in md.get("fields", ()))
            if not fields:
                raise CatalogueError(f"{mid}: no fields")
            names = [f.name for f in fields]
            if len(set(names)) != len(names):
                raise CatalogueError(f"{mid}: duplicate field names")
            messages[mid] = MessageDef(mid, code, md.get("title", mid), md.get("domain", ""), md.get("service", ""), fields,
                                       _str_list(f"{mid}.keywords", md.get("keywords", ())),
                                       _check_fusion(mid, md.get("fusion"), fields))
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        tag = hashlib.sha256(canonical).digest()[0]
        return cls(data["name"], data["version"], data["namespace"], messages, tag)


def _check_fusion(mid: str, cfg: dict | None, fields: tuple[Field, ...]) -> dict | None:
    if cfg is None:
        return None
    where = f"{mid}.fusion"
    unknown = set(cfg) - _FUSION_KEYS
    if unknown:
        raise CatalogueError(f"{where}: unknown keys {sorted(unknown)}")
    by_name = {f.name: f for f in fields}
    listed = [n for key in ("identity", "id_any", "must_match", "soft_match", "vote_fields") for n in cfg.get(key, ())]
    listed += [cfg[k] for k in ("course_field", "speed_field") if k in cfg]
    spatial = cfg.get("spatial")
    if spatial is not None:
        if set(spatial) == {"grid"}:
            if by_name.get(spatial["grid"]) is None or by_name[spatial["grid"]].type != "mgrs":
                raise CatalogueError(f"{where}: spatial.grid must name an mgrs field")
        elif set(spatial) == {"lat", "lon"}:
            for k in ("lat", "lon"):
                if by_name.get(spatial[k]) is None or by_name[spatial[k]].type != "decimal":
                    raise CatalogueError(f"{where}: spatial.{k} must name a decimal field")
        else:
            raise CatalogueError(f"{where}: spatial needs either 'grid' or 'lat' and 'lon'")
        for key in ("radius_m", "time_window_s"):
            if not isinstance(cfg.get(key), (int, float)) or cfg[key] <= 0:
                raise CatalogueError(f"{where}: spatial matching needs a positive {key}")
    missing = [n for n in listed if n not in by_name]
    if missing:
        raise CatalogueError(f"{where}: unknown fields {missing}")
    if not (cfg.get("identity") or cfg.get("id_any") or spatial):
        raise CatalogueError(f"{where}: needs identity, id_any or spatial rules")
    return cfg


def _str_list(where: str, items: object) -> tuple[str, ...]:
    if not isinstance(items, (list, tuple)) or not all(isinstance(s, str) and s.strip() for s in items):
        raise CatalogueError(f"{where}: must be a list of non-empty strings")
    return tuple(items)


def _parse_field(mid: str, fd: dict[str, Any]) -> Field:
    unknown = set(fd) - _FIELD_KEYS
    if unknown:
        raise CatalogueError(f"{mid}: unknown field keys {sorted(unknown)}")
    name, ftype = fd.get("name", ""), fd.get("type")
    where = f"{mid}.{name}"
    if not NAME_RE.fullmatch(name):
        raise CatalogueError(f"{mid}: bad field name {name!r}")
    if ftype not in TYPES:
        raise CatalogueError(f"{where}: unknown type {ftype!r}")
    pattern = fd.get("pattern")
    if pattern is not None:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise CatalogueError(f"{where}: bad pattern: {exc}") from exc
    max_len = fd.get("max_len")
    if max_len is not None and (not isinstance(max_len, int) or max_len < 1):
        raise CatalogueError(f"{where}: max_len must be a positive integer")
    values = tuple(fd.get("values", ()))
    if ftype == "enum":
        if not values or len(set(values)) != len(values) or not all(NAME_RE.fullmatch(v) or re.fullmatch(r"[IVX]+", v) for v in values):
            raise CatalogueError(f"{where}: enum needs unique values matching [A-Z][A-Z0-9_]*")
    minimum = Decimal(str(fd["min"])) if "min" in fd else None
    maximum = Decimal(str(fd["max"])) if "max" in fd else None
    if minimum is not None and maximum is not None and minimum > maximum:
        raise CatalogueError(f"{where}: min > max")
    scale = fd.get("scale", 0)
    if ftype == "decimal" and not (isinstance(scale, int) and 1 <= scale <= 6):
        raise CatalogueError(f"{where}: decimal needs scale 1-6")
    syn_raw = fd.get("synonyms", {})
    if syn_raw and ftype != "enum":
        raise CatalogueError(f"{where}: synonyms only apply to enum fields")
    if not isinstance(syn_raw, dict) or any(k not in values for k in syn_raw):
        raise CatalogueError(f"{where}: synonyms must map allowed enum values to phrase lists")
    synonyms = tuple((k, _str_list(f"{where}.synonyms.{k}", v)) for k, v in syn_raw.items())
    hemis = fd.get("hemis", "")
    if hemis and (ftype != "decimal" or hemis not in ("NS", "EW")):
        raise CatalogueError(f"{where}: hemis must be 'NS' or 'EW' on a decimal field")
    f = Field(name, ftype, bool(fd.get("required", False)), max_len, pattern, values, minimum, maximum, scale,
              tuple(fd.get("examples", ())), _str_list(f"{where}.cues", fd.get("cues", ())), synonyms,
              _str_list(f"{where}.units", fd.get("units", ())), hemis)
    for ex in f.examples:
        try:
            f.normalize(ex)
        except ValueError as exc:
            raise CatalogueError(f"{where}: example {ex!r} invalid: {exc}") from exc
    return f
