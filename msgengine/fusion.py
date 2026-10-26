"""Fusion: resolve reports to entities, corroborate, flag conflicts, and keep full provenance.

Rules per message family come from the catalogue ("fusion" key). The engine never overwrites silently:
every fused field names the report it came from and lists the values it did not choose. Where it is not
sure two reports describe the same thing it keeps them apart and says so.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from .catalogue import MONTHS, Catalogue
from .geo import GridPoint, grid_distance_m, haversine_m, offset_latlon, parse_mgrs
from .model import Message

KNOT_MPS = 0.514444


def dtg_seconds(value: str) -> float:
    return datetime(2000 + int(value[10:12]), MONTHS.index(value[7:10]) + 1, int(value[0:2]),
                    int(value[2:4]), int(value[4:6]), tzinfo=timezone.utc).timestamp()


@dataclass
class Report:
    id: str
    source: str
    msg: Message
    received: float = 0.0
    confidence: float = 1.0


@dataclass
class Obs:
    report_id: str
    source: str
    time: float
    weight: float
    value: str
    seq: int


@dataclass
class FusedField:
    value: str
    report_id: str
    source: str
    status: str = "ok"
    alternatives: tuple[tuple[str, str, str], ...] = ()


@dataclass
class Outcome:
    report_id: str
    entity_id: str
    action: str
    score: float = 1.0
    notes: list[str] = field(default_factory=list)


@dataclass
class Entity:
    id: str
    type: str
    obs: dict[str, list[Obs]] = field(default_factory=dict)
    reports: list[tuple[str, str]] = field(default_factory=list)
    sources: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    ids: dict[str, set[str]] = field(default_factory=dict)
    track: list[tuple[float, object, float | None, float | None]] = field(default_factory=list)
    latest_time: float = 0.0

    @property
    def confidence(self) -> float:
        miss = 1.0
        for c in self.sources.values():
            miss *= 1.0 - max(0.0, min(1.0, c))
        return round(1.0 - miss, 3)


def _pos(cfg: dict, msg: Message):
    sp = cfg.get("spatial")
    if not sp:
        return None
    try:
        if "grid" in sp:
            v = msg.fields.get(sp["grid"])
            return parse_mgrs(v) if v else None
        lat, lon = msg.fields.get(sp["lat"]), msg.fields.get(sp["lon"])
        return (float(lat), float(lon)) if lat and lon else None
    except ValueError:
        return None


def _distance(a, b) -> tuple[float, float] | None:
    """(metres, combined position resolution) or None if not comparable."""
    if isinstance(a, GridPoint) and isinstance(b, GridPoint):
        d = grid_distance_m(a, b)
        return None if d is None else (d, a.resolution + b.resolution)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return haversine_m(a[0], a[1], b[0], b[1]), 22.0
    return None


def _moved(pos, speed_mps: float, course_deg: float, dt: float):
    de, dn = speed_mps * math.sin(math.radians(course_deg)) * dt, speed_mps * math.cos(math.radians(course_deg)) * dt
    if isinstance(pos, GridPoint):
        return replace(pos, easting=pos.easting + de, northing_mod=pos.northing_mod + dn)
    return offset_latlon(pos[0], pos[1], de, dn)


class FusionEngine:
    def __init__(self, cat: Catalogue) -> None:
        self.cat = cat
        self._entities: dict[str, Entity] = {}
        self._seen: dict[tuple, str] = {}
        self._seq = 0
        self._n = 0

    def entities(self) -> list[Entity]:
        return list(self._entities.values())

    def ingest(self, rep: Report) -> Outcome:
        msg = rep.msg
        mdef = self.cat.message(msg.type)
        cfg = mdef.fusion or {}
        sig = (rep.source, msg.type, tuple(sorted(msg.fields.items())))
        if sig in self._seen:
            e = self._entities[self._seen[sig]]
            e.reports.append((rep.id, "duplicate_ignored"))
            return Outcome(rep.id, e.id, "duplicate_ignored", 1.0, ["identical report from the same source"])
        t = dtg_seconds(msg.fields["DTG"]) if "DTG" in msg.fields else rep.received
        pos = _pos(cfg, msg)

        cands, suggestions = self._candidates(cfg, msg, t, pos)
        notes: list[str] = []
        if cands:
            cands.sort(key=lambda c: c[0], reverse=True)
            score, entity, reason, cand_notes = cands[0]
            notes += cand_notes
            if len(cands) > 1 and score - cands[1][0] < 0.1:
                entity.flags.append(f"ambiguous merge of {rep.id}: also fits {cands[1][1].id}")
                notes.append(f"also fits {cands[1][1].id}")
            before = self._fuse(entity, cfg)
            late = t < entity.latest_time - 60
            new_source = rep.source not in entity.sources
            self._add(entity, rep, t, pos, cfg)
            after = self._fuse(entity, cfg)
            new_conflicts = {n for n, f in after.items() if f.status == "conflict"} - {n for n, f in before.items() if f.status == "conflict"}
            changed = [n for n, f in after.items() if n not in before or before[n].value != f.value]
            if late:
                notes.append("arrived after newer information")
            if new_conflicts:
                action = "conflict"
                notes.append("sources disagree on " + ", ".join(sorted(new_conflicts)))
            elif changed:
                action = "updated"
            elif new_source:
                action = "corroborated"
            else:
                action = "repeated"
            entity.reports.append((rep.id, action))
            self._seen[sig] = entity.id
            return Outcome(rep.id, entity.id, action, round(score, 2), notes)

        self._n += 1
        entity = Entity(f"E{self._n}", msg.type)
        for other in suggestions:
            entity.flags.append(f"possible duplicate of {other.id}")
            notes.append(f"possible duplicate of {other.id}")
        self._entities[entity.id] = entity
        self._add(entity, rep, t, pos, cfg)
        entity.reports.append((rep.id, "created"))
        self._seen[sig] = entity.id
        return Outcome(rep.id, entity.id, "created", 1.0, notes)

    def _add(self, e: Entity, rep: Report, t: float, pos, cfg: dict) -> None:
        for name, value in rep.msg.fields.items():
            self._seq += 1
            weight = rep.confidence * (0.25 if value == "UNKNOWN" else 1.0)
            e.obs.setdefault(name, []).append(Obs(rep.id, rep.source, t, weight, value, self._seq))
        for k in cfg.get("id_any", ()):
            if k in rep.msg.fields:
                e.ids.setdefault(k, set()).add(rep.msg.fields[k])
        e.sources[rep.source] = max(e.sources.get(rep.source, 0.0), rep.confidence)
        if pos is not None:
            v = c = None
            try:
                if cfg.get("speed_field") in rep.msg.fields and cfg.get("course_field") in rep.msg.fields:
                    v = float(rep.msg.fields[cfg["speed_field"]]) * KNOT_MPS
                    c = float(rep.msg.fields[cfg["course_field"]])
            except ValueError:
                v = c = None
            e.track.append((t, pos, v, c))
        e.latest_time = max(e.latest_time, t)

    def _candidates(self, cfg: dict, msg: Message, t: float, pos):
        cands, suggestions = [], []
        for e in self._entities.values():
            if e.type != msg.type:
                continue
            if cfg.get("identity"):
                keys = cfg["identity"]
                if all(k in msg.fields for k in keys) and all(msg.fields[k] == self._identity_value(e, k) for k in keys):
                    cands.append((1.0, e, "identity", []))
                continue
            sp = self._spatial(e, cfg, msg, t, pos) if cfg.get("spatial") else None
            id_keys = cfg.get("id_any", ())
            if id_keys:
                if any(msg.fields.get(k) in e.ids.get(k, ()) for k in id_keys):
                    cands.append((0.95, e, "request id", []))
                elif sp is not None:
                    suggestions.append(e)
                continue
            if sp is not None:
                cands.append((sp[0], e, "position and time", sp[1]))
        return cands, suggestions

    def _identity_value(self, e: Entity, key: str) -> str | None:
        obs = e.obs.get(key)
        return obs[0].value if obs else None

    def _spatial(self, e: Entity, cfg: dict, msg: Message, t: float, pos):
        if pos is None or not e.track:
            return None
        t0, p0, _, _ = max(e.track, key=lambda x: x[0])
        kin = max((x for x in e.track if x[2] is not None), key=lambda x: x[0], default=None)
        v, course = (kin[2], kin[3]) if kin else (None, None)
        dt = t - t0
        if abs(dt) > cfg["time_window_s"]:
            return None
        for k in cfg.get("must_match", ()):
            have = self._latest(e, k)
            if k in msg.fields and have is not None and have != msg.fields[k]:
                return None
        predicted = v is not None and course is not None
        if predicted:
            p0 = _moved(p0, v, course, dt)
        dist = _distance(p0, pos)
        if dist is None:
            return None
        metres, res = dist
        speed = v if predicted else cfg.get("default_speed_mps", 0.0)
        growth = (0.25 if predicted else 1.0) * speed * abs(dt) + speed * cfg.get("time_resolution_s", 60.0)
        gate = cfg["radius_m"] + res + growth
        if metres > gate:
            return None
        score = 1.0 - 0.8 * metres / gate
        notes = []
        for k in cfg.get("soft_match", ()):
            have = self._latest(e, k)
            if k in msg.fields and have not in (None, "UNKNOWN") and msg.fields[k] not in ("UNKNOWN", have):
                score -= 0.3
                notes.append(f"{k} differs: {have} vs {msg.fields[k]}")
        return (score, notes) if score > 0.1 else None

    @staticmethod
    def _latest(e: Entity, name: str) -> str | None:
        obs = e.obs.get(name)
        return max(obs, key=lambda o: (o.time, o.seq)).value if obs else None

    def _fuse(self, e: Entity, cfg: dict) -> dict[str, FusedField]:
        out: dict[str, FusedField] = {}
        vote = set(cfg.get("vote_fields", ())) if not cfg.get("supersede") else set()
        half = float(cfg.get("time_window_s", 3600))
        for name, obs in e.obs.items():
            if name not in vote:
                best = max(obs, key=lambda o: (o.time, o.weight, o.seq))
                alts = tuple(dict.fromkeys((o.value, o.report_id, o.source) for o in sorted(obs, key=lambda o: -o.time)
                                           if o.value != best.value))[:3]
                out[name] = FusedField(best.value, best.report_id, best.source, "ok", alts)
                continue
            newest = max(o.time for o in obs)
            totals: dict[str, float] = defaultdict(float)
            rep: dict[str, Obs] = {}
            for o in obs:
                w = o.weight * 0.5 ** ((newest - o.time) / half)
                totals[o.value] += w
                if o.value not in rep or o.weight > rep[o.value].weight:
                    rep[o.value] = o
            ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
            top = ranked[0]
            conflict = any(v != "UNKNOWN" and top[0] != "UNKNOWN" and w >= 0.5 * top[1] for v, w in ranked[1:])
            alts = tuple((v, rep[v].report_id, rep[v].source) for v, _ in ranked[1:])
            out[name] = FusedField(top[0], rep[top[0]].report_id, rep[top[0]].source,
                                   "conflict" if conflict else "ok", alts)
        return out

    def snapshot(self) -> list[dict]:
        rows = []
        for e in self._entities.values():
            cfg = self.cat.message(e.type).fusion or {}
            fused = self._fuse(e, cfg)
            flags = list(e.flags) + [f"sources disagree on {n}" for n, f in fused.items() if f.status == "conflict"]
            rows.append({
                "id": e.id, "type": e.type, "confidence": e.confidence, "sources": sorted(e.sources),
                "flags": flags, "reports": [{"id": r, "action": a} for r, a in e.reports],
                "fields": {n: {"value": f.value, "from_report": f.report_id, "source": f.source, "status": f.status,
                               "alternatives": [{"value": v, "report": r, "source": s} for v, r, s in f.alternatives]}
                           for n, f in fused.items()},
            })
        return rows
