"""Small geodesy helpers for fusion: MGRS grid distance (same zone) and great-circle distance."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

_COLS = {1: "ABCDEFGH", 2: "JKLMNPQR", 0: "STUVWXYZ"}
_ROWS = "ABCDEFGHJKLMNPQRSTUV"
_MGRS = re.compile(r"([0-9]{1,2})([C-X])([A-Z])([A-Z])([0-9]+)")
_EARTH_M = 6371008.8


@dataclass(frozen=True)
class GridPoint:
    zone: int
    easting: float
    northing_mod: float
    resolution: float


def parse_mgrs(value: str) -> GridPoint:
    m = _MGRS.fullmatch(value)
    if not m or len(m.group(5)) % 2:
        raise ValueError(f"bad MGRS {value!r}")
    zone, col, row, digits = int(m.group(1)), m.group(3), m.group(4), m.group(5)
    if col not in _COLS[zone % 3] or row not in _ROWS:
        raise ValueError(f"bad MGRS square {value!r}")
    half = len(digits) // 2
    scale = 10 ** (5 - half)
    easting = (_COLS[zone % 3].index(col) + 1) * 100000 + int(digits[:half]) * scale
    row_idx = (_ROWS.index(row) - (5 if zone % 2 == 0 else 0)) % 20
    return GridPoint(zone, easting, row_idx * 100000 + int(digits[half:]) * scale, float(scale))


def grid_distance_m(a: GridPoint, b: GridPoint) -> float | None:
    """Metres between two grid points in the same zone; None across zones. Northing wraps every 2,000 km."""
    if a.zone != b.zone:
        return None
    d_n = (b.northing_mod - a.northing_mod + 1_000_000) % 2_000_000 - 1_000_000
    return math.hypot(b.easting - a.easting, d_n)


def make_mgrs(zone: int, band: str, easting: float, northing: float, digits: int = 10) -> str:
    half = digits // 2
    col = _COLS[zone % 3][int(easting // 100000) - 1]
    row_idx = int((northing % 2_000_000) // 100000) % 20
    row = _ROWS[(row_idx + 5) % 20 if zone % 2 == 0 else row_idx]
    scale = 10 ** (5 - half)
    return f"{zone}{band}{col}{row}{int(easting % 100000) // scale:0{half}d}{int(northing % 100000) // scale:0{half}d}"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * _EARTH_M * math.asin(math.sqrt(a))


def offset_latlon(lat: float, lon: float, east_m: float, north_m: float) -> tuple[float, float]:
    return (lat + math.degrees(north_m / _EARTH_M),
            lon + math.degrees(east_m / (_EARTH_M * math.cos(math.radians(lat)))))
