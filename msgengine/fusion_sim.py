"""Synthetic multi-source scenario for fusion, with ground truth, plus scoring.

Noise levels and error rates are assumptions chosen by the developer, not measurements.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .catalogue import MONTHS, Catalogue
from .fusion import FusionEngine, Report, dtg_seconds
from .geo import make_mgrs, offset_latlon
from .model import Message

T0 = dtg_seconds("241200ZOCT26")
BASE = (45.0, -63.0)


def dtg(t: float) -> str:
    d = datetime.fromtimestamp(t, timezone.utc)
    return f"{d.day:02d}{d.hour:02d}{d.minute:02d}Z{MONTHS[d.month - 1]}{d.year - 2000:02d}"


@dataclass
class Scenario:
    reports: list[Report] = field(default_factory=list)
    truth: dict[str, str] = field(default_factory=dict)
    attrs: dict[str, dict[str, str]] = field(default_factory=dict)
    exact_duplicates: int = 0
    relay_pairs: list[tuple[str, str]] = field(default_factory=list)


def _msg(cat: Catalogue, mtype: str, fields: dict[str, str]) -> Message:
    mdef = cat.message(mtype)
    return Message(mtype, {k: mdef.field(k).normalize(v) for k, v in fields.items()})


def make_scenario(cat: Catalogue, seed: int, contacts: int = 20, targets: int = 15, medevacs: int = 10,
                  logs: int = 8) -> Scenario:
    rng = random.Random(seed)
    items: list[tuple[float, Message, str, str, float]] = []  # arrival, msg, source, truth key, confidence
    sc = Scenario()

    def delay() -> float:
        return 30 + rng.expovariate(1 / 60) + (900 if rng.random() < 0.05 else 0)

    def emit(t_obs: float, msg: Message, source: str, key: str, p_dup: float = 0.08) -> None:
        conf = round(rng.uniform(0.6, 0.95), 2)
        arrival = t_obs + delay()
        items.append((arrival, msg, source, key, conf))
        if rng.random() < p_dup:
            items.append((arrival + rng.uniform(20, 400), msg, source, key, conf))
            sc.exact_duplicates += 1

    # contacts (air, surface, subsurface); every 5th has a twin 800 m away
    classes = ["FRIEND", "HOSTILE", "NEUTRAL"]
    specs = []
    for i in range(contacts):
        dom = rng.choice(["AIR", "SURFACE", "SUBSURFACE"])
        lat, lon = offset_latlon(*BASE, rng.uniform(-15000, 15000), rng.uniform(-15000, 15000))
        speed = rng.uniform(60, 120) if dom == "AIR" else rng.uniform(5, 25)
        specs.append({"key": f"C{i}", "dom": dom, "lat": lat, "lon": lon, "kts": speed, "course": rng.uniform(0, 360),
                      "cls": rng.choice(classes)})
        if i % 5 == 0:
            lat2, lon2 = offset_latlon(lat, lon, 800, 0)
            specs.append({"key": f"C{i}t", "dom": dom, "lat": lat2, "lon": lon2, "kts": speed,
                          "course": (specs[-1]["course"] + 20) % 360, "cls": rng.choice(classes)})
    sources = ["HMCS-A", "HMCS-B", "SHORE-2"]
    for s in specs:
        sc.attrs[s["key"]] = {"CLASS": s["cls"]}
        v = s["kts"] * 0.514444
        for src in sources:
            for k in range(10):
                if rng.random() > 0.7:
                    continue
                t = T0 + k * 120 + rng.uniform(-20, 20)
                lat, lon = offset_latlon(s["lat"], s["lon"], v * math.sin(math.radians(s["course"])) * (t - T0) + rng.gauss(0, 85),
                                         v * math.cos(math.radians(s["course"])) * (t - T0) + rng.gauss(0, 85))
                roll = rng.random()
                label = s["cls"] if roll < 0.78 else "UNKNOWN" if roll < 0.95 else rng.choice([c for c in classes if c != s["cls"]])
                f = {"REPORTER": src, "DTG": dtg(t), "DOMAIN": s["dom"], "LAT": f"{lat:.4f}", "LON": f"{lon:.4f}", "CLASS": label}
                if rng.random() < 0.7:
                    f["COURSE"] = str(int((s["course"] + rng.gauss(0, 5)) % 360))
                    f["SPEED"] = f"{max(0.0, s['kts'] + rng.gauss(0, 1)):.1f}"
                emit(t, _msg(cat, "CONTACT", f), src, s["key"])
    for i in range(6):
        lat, lon = offset_latlon(*BASE, rng.uniform(-15000, 15000), rng.uniform(-15000, 15000))
        t = T0 + rng.uniform(0, 1200)
        f = {"REPORTER": rng.choice(sources), "DTG": dtg(t), "DOMAIN": rng.choice(["AIR", "SURFACE"]), "LAT": f"{lat:.4f}",
             "LON": f"{lon:.4f}", "CLASS": "UNKNOWN"}
        emit(t, _msg(cat, "CONTACT", f), f["REPORTER"], f"X{i}", 0.0)
        sc.attrs[f"X{i}"] = {"CLASS": "UNKNOWN"}

    # call-for-fire targets (static); every 5th has a twin 300 m away
    tgts = ["INF", "ARM", "VEH", "MORTAR", "ARTY", "BLDG", "OTHER"]
    pts = []
    for i in range(targets):
        e, n = 233870 + rng.uniform(-8000, 8000), 4_900_000 + rng.uniform(-8000, 8000)
        pts.append((f"T{i}", e, n, rng.choice(tgts)))
        if i % 5 == 0:
            pts.append((f"T{i}t", e + 300, n, rng.choice(tgts)))
    for key, e, n, truth in pts:
        sc.attrs[key] = {"TGT": truth}
        for src in rng.sample(["B22", "C14B", "OP3"], k=rng.randint(2, 3)):
            for _ in range(rng.randint(1, 3)):
                t = T0 + rng.uniform(0, 1800)
                grid = make_mgrs(18, "S", e + rng.gauss(0, 40), n + rng.gauss(0, 40), rng.choice((8, 10)))
                label = truth if rng.random() < 0.8 else rng.choice([x for x in tgts if x != truth])
                emit(t, _msg(cat, "CFF", {"OBS": src, "DTG": dtg(t), "TGTGRID": grid, "TGT": label,
                                          "MISSION": rng.choice(["ADJUST", "FFE", "SMOKE", "ILLUM"])}), src, key)

    # medevac events; some relayed under a different request id
    secs = ["NO_ENEMY", "POSSIBLE_ENEMY", "ENEMY", "ARMED_ESCORT"]
    for i in range(medevacs):
        e, n = 233870 + rng.uniform(-8000, 8000), 4_900_000 + rng.uniform(-8000, 8000)
        truth_sec, rid = rng.choice(secs), f"MV-{i:03d}"
        sc.attrs[f"M{i}"] = {"SECURITY": truth_sec}
        t = T0 + rng.uniform(0, 3000)
        for src in ["B22", "C14B"] if rng.random() < 0.5 else ["B22"]:
            label = truth_sec if rng.random() < 0.85 else rng.choice([x for x in secs if x != truth_sec])
            for _ in range(rng.randint(1, 2)):
                tt = t + rng.uniform(0, 600)
                emit(tt, _msg(cat, "MEDEVAC", {"REQID": rid, "DTG": dtg(tt), "PICKUP": make_mgrs(18, "S", e + rng.gauss(0, 20), n + rng.gauss(0, 20), 10),
                                              "FREQ": "243.500", "PRECEDENCE": "URGENT", "SECURITY": label}), src, f"M{i}")
        if rng.random() < 0.3:
            tt = t + rng.uniform(60, 600)
            emit(tt, _msg(cat, "MEDEVAC", {"REQID": f"RLY-{i:03d}", "DTG": dtg(tt), "PICKUP": make_mgrs(18, "S", e + rng.gauss(0, 30), n + rng.gauss(0, 30), 10),
                                          "FREQ": "243.500", "PRECEDENCE": "URGENT", "SECURITY": truth_sec}), "OP3", f"M{i}r")
            sc.relay_pairs.append((f"M{i}", f"M{i}r"))

    # logistics: periodic reports per (unit, item); late arrivals out of order
    pairs = rng.sample([(u, it) for u in ("1PL", "2PL", "3PL", "A-COY") for it in ("WATER", "DIESEL", "RATIONS")], k=logs)
    for unit, item in pairs:
        key, onhand = f"L-{unit}-{item}", rng.randint(2000, 5000)
        latest_t = -1.0
        for k in range(7):
            t = T0 + k * 600
            onhand = max(0, onhand - rng.randint(50, 400))
            msg = _msg(cat, "LOGSTAT", {"UNIT": unit, "DTG": dtg(t), "SUPPLY_CLASS": "I", "ITEM": item, "ONHAND": str(onhand),
                                        "STATUS": "GREEN" if onhand > 1500 else "AMBER" if onhand > 500 else "RED"})
            emit(t, msg, f"S-{unit}", key)
            if t > latest_t:
                latest_t, sc.attrs[key] = t, {"ONHAND": str(onhand)}
    items.sort(key=lambda x: x[0])
    for n, (arrival, msg, source, key, conf) in enumerate(items, 1):
        rid = f"R{n}"
        sc.reports.append(Report(rid, source, msg, arrival, conf))
        sc.truth[rid] = key
    return sc


def _pairs(groups: Counter) -> int:
    return sum(n * (n - 1) // 2 for n in groups.values())


def evaluate(cat: Catalogue, sc: Scenario) -> dict:
    eng = FusionEngine(cat)
    where: dict[str, str] = {}
    actions = Counter()
    for rep in sc.reports:
        out = eng.ingest(rep)
        where[rep.id] = out.entity_id
        actions[out.action] += 1
    snap = {e["id"]: e for e in eng.snapshot()}
    types = {r.id: r.msg.type for r in sc.reports}
    res: dict = {"reports": len(sc.reports), "entities": len(snap), "actions": dict(actions),
                 "injected_exact_duplicates": sc.exact_duplicates}

    for mtype in ("CONTACT", "CFF", "MEDEVAC", "LOGSTAT"):
        ids = [r for r in types if types[r] == mtype]
        by_truth, by_ent, cell = Counter(sc.truth[r] for r in ids), Counter(where[r] for r in ids), Counter((sc.truth[r], where[r]) for r in ids)
        same_e, same_t, both = _pairs(by_ent), _pairs(by_truth), _pairs(cell)
        res[mtype] = {"reports": len(ids), "entities": len(by_ent), "truth_entities": len(by_truth),
                      "pair_precision": round(both / same_e, 4) if same_e else 1.0,
                      "pair_recall": round(both / same_t, 4) if same_t else 1.0}

    def majority_truth(eid: str) -> str:
        return Counter(sc.truth[r["id"]] for r in snap[eid]["reports"]).most_common(1)[0][0]

    def attr_score(mtype: str, name: str, truth_name: str) -> dict:
        right = wrong = unknown = flagged_wrong = 0
        for eid, e in snap.items():
            if e["type"] != mtype or name not in e["fields"]:
                continue
            f, want = e["fields"][name], sc.attrs.get(majority_truth(eid), {}).get(truth_name)
            if want is None:
                continue
            if f["value"] == want:
                right += 1
            elif f["value"] == "UNKNOWN":
                unknown += 1
            else:
                wrong += 1
                flagged_wrong += f["status"] == "conflict"
        n = right + wrong + unknown
        return {"entities": n, "correct": right, "unknown": unknown, "wrong": wrong, "wrong_and_flagged": flagged_wrong}

    res["CONTACT"]["class"] = attr_score("CONTACT", "CLASS", "CLASS")
    res["CFF"]["target_type"] = attr_score("CFF", "TGT", "TGT")
    res["MEDEVAC"]["security"] = attr_score("MEDEVAC", "SECURITY", "SECURITY")
    res["LOGSTAT"]["latest_onhand"] = attr_score("LOGSTAT", "ONHAND", "ONHAND")

    contact_reports = [r for r in sc.reports if r.msg.type == "CONTACT" and not sc.truth[r.id].startswith("X")]
    raw = Counter("correct" if r.msg.fields["CLASS"] == sc.attrs[sc.truth[r.id]]["CLASS"] else
                  "unknown" if r.msg.fields["CLASS"] == "UNKNOWN" else "wrong" for r in contact_reports)
    res["CONTACT"]["raw_report_class"] = dict(raw)

    hits = 0
    for a, b in sc.relay_pairs:
        ea = {where[r] for r, k in sc.truth.items() if k == a}
        eb = {where[r] for r, k in sc.truth.items() if k == b}
        if ea.isdisjoint(eb) and any(f"possible duplicate of {x}" in snap[y]["flags"] or f"possible duplicate of {y}" in snap[x]["flags"]
                                     for y in eb for x in ea):
            hits += 1
    res["MEDEVAC"]["relay_duplicates"] = {"injected": len(sc.relay_pairs), "flagged_as_possible_duplicate": hits}

    seen = Counter(r["id"] for e in snap.values() for r in e["reports"])
    res["provenance"] = {"every_report_in_exactly_one_entity": set(seen) == {r.id for r in sc.reports} and max(seen.values()) == 1,
                         "fused_fields_trace_to_known_reports": all(f["from_report"] in seen for e in snap.values() for f in e["fields"].values())}
    return res
