import math
import random

import pytest

from msgengine.catalogue import Catalogue, CatalogueError
from msgengine.fusion import FusionEngine, Report, dtg_seconds
from msgengine.fusion_sim import T0, dtg, evaluate, make_scenario
from msgengine.geo import (grid_distance_m, haversine_m, make_mgrs, offset_latlon, parse_mgrs)
from msgengine.model import Message
from tests.conftest import CATALOGUE_PATH


def contact(cat, rid, src, secs, east=0.0, north=0.0, cls="UNKNOWN", dom="SURFACE", course=None, speed=None, conf=0.9):
    lat, lon = offset_latlon(45.0, -63.0, east, north)
    f = {"REPORTER": src, "DTG": dtg(T0 + secs), "DOMAIN": dom, "LAT": f"{lat:.4f}", "LON": f"{lon:.4f}", "CLASS": cls}
    if course is not None:
        f["COURSE"], f["SPEED"] = str(course), f"{speed:.1f}"
    mdef = cat.message("CONTACT")
    return Report(rid, src, Message("CONTACT", {k: mdef.field(k).normalize(v) for k, v in f.items()}), 0.0, conf)


def cff(cat, rid, src, secs, east=0.0, north=0.0, tgt="INF"):
    grid = make_mgrs(18, "S", 233870 + east, 4_900_000 + north, 10)
    f = {"OBS": src, "DTG": dtg(T0 + secs), "TGTGRID": grid, "TGT": tgt, "MISSION": "FFE"}
    return Report(rid, src, Message("CFF", f), 0.0, 0.9)


def medevac(cat, rid, src, secs, reqid, east=0.0, north=0.0, sec="NO_ENEMY"):
    f = {"REQID": reqid, "DTG": dtg(T0 + secs), "PICKUP": make_mgrs(18, "S", 233870 + east, 4_900_000 + north, 10),
         "FREQ": "243.500", "PRECEDENCE": "URGENT", "SECURITY": sec}
    return Report(rid, src, Message("MEDEVAC", f), 0.0, 0.9)


def logstat(cat, rid, secs, unit="1PL", item="WATER", onhand="100", src=None):
    f = {"UNIT": unit, "DTG": dtg(T0 + secs), "SUPPLY_CLASS": "I", "ITEM": item, "ONHAND": onhand, "STATUS": "GREEN"}
    return Report(rid, src or f"S-{unit}", Message("LOGSTAT", f), 0.0, 0.9)


# geo

def test_mgrs_make_parse_round_trip_across_zones():
    rng = random.Random(1)
    for _ in range(500):
        zone = rng.randint(1, 60)
        e, n = rng.uniform(100_000, 799_000), rng.uniform(0, 9_000_000)
        p = parse_mgrs(make_mgrs(zone, "S", e, n, 10))
        assert p.zone == zone and abs(p.easting - e) <= 1 and abs(p.northing_mod - n % 2_000_000) <= 1


def test_grid_distance_matches_construction_including_wrap():
    a = parse_mgrs(make_mgrs(18, "S", 300_000, 4_999_500, 10))
    b = parse_mgrs(make_mgrs(18, "S", 300_000, 5_000_700, 10))
    assert grid_distance_m(a, b) == pytest.approx(1200, abs=2)
    c = parse_mgrs(make_mgrs(19, "S", 300_000, 4_999_500, 10))
    assert grid_distance_m(a, c) is None


def test_grid_resolution_reflects_digits():
    assert parse_mgrs("18SUJ2338706520").resolution == 1.0
    assert parse_mgrs("18SUJ23380652").resolution == 10.0
    assert parse_mgrs("18SUJ2306").resolution == 1000.0


@pytest.mark.parametrize("bad", ["18SUJ123", "18SIJ12341234", "99", "18SUJ1234O234"])
def test_bad_mgrs_rejected(bad):
    with pytest.raises(ValueError):
        parse_mgrs(bad)


def test_haversine_and_offset_agree():
    assert haversine_m(45, -63, 46, -63) == pytest.approx(111_195, rel=0.001)
    lat, lon = offset_latlon(45.0, -63.0, 3000, 4000)
    assert haversine_m(45.0, -63.0, lat, lon) == pytest.approx(5000, rel=0.002)


# fusion rules in the catalogue

def test_catalogue_fusion_rules_validated():
    import json
    base = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
    for patch, text in [
        (lambda c: c.update(bogus=1), "unknown keys"),
        (lambda c: c.update(spatial={"grid": "OBS"}), "mgrs field"),
        (lambda c: c.update(spatial={"lat": "LAT"}), "either 'grid' or"),
        (lambda c: c.pop("time_window_s"), "time_window_s"),
        (lambda c: c.update(vote_fields=["NOPE"]), "unknown fields"),
        (lambda c: c.clear(), "needs identity"),
    ]:
        d = json.loads(json.dumps(base))
        cfg = d["messages"][0]["fusion"]
        patch(cfg)
        with pytest.raises(CatalogueError, match=text):
            Catalogue.from_dict(d)


def test_family_without_rules_is_never_merged(cat):
    import json
    d = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
    del d["messages"][0]["fusion"]
    cat2 = Catalogue.from_dict(d)
    eng = FusionEngine(cat2)
    a, b = cff(cat2, "R1", "B22", 0), cff(cat2, "R2", "OP3", 10)
    assert eng.ingest(a).entity_id != eng.ingest(b).entity_id


# duplicates, merging, separation

def test_identical_resend_from_same_source_is_ignored_but_recorded(cat):
    eng = FusionEngine(cat)
    r = contact(cat, "R1", "HMCS-A", 0)
    first = eng.ingest(r)
    again = eng.ingest(Report("R2", r.source, r.msg, 5.0, r.confidence))
    assert again.action == "duplicate_ignored" and again.entity_id == first.entity_id
    snap = eng.snapshot()
    assert len(snap) == 1 and [x["action"] for x in snap[0]["reports"]] == ["created", "duplicate_ignored"]


def test_two_sources_same_contact_merge_and_corroborate(cat):
    eng = FusionEngine(cat)
    a = eng.ingest(contact(cat, "R1", "HMCS-A", 0, 0, 0, cls="HOSTILE"))
    b = eng.ingest(contact(cat, "R2", "SHORE-2", 30, 150, 100, cls="HOSTILE"))
    assert b.entity_id == a.entity_id and b.action in ("corroborated", "updated")
    e = eng.entities()[0]
    assert e.confidence > 0.9 and set(e.sources) == {"HMCS-A", "SHORE-2"}


def test_far_apart_or_different_domain_stay_separate(cat):
    eng = FusionEngine(cat)
    ids = {eng.ingest(contact(cat, "R1", "HMCS-A", 0)).entity_id,
           eng.ingest(contact(cat, "R2", "HMCS-B", 0, east=8000)).entity_id,
           eng.ingest(contact(cat, "R3", "HMCS-B", 0, dom="AIR")).entity_id}
    assert len(ids) == 3


def test_outside_time_window_does_not_merge(cat):
    eng = FusionEngine(cat)
    a = eng.ingest(contact(cat, "R1", "HMCS-A", 0))
    b = eng.ingest(contact(cat, "R2", "HMCS-A", 1800))
    assert a.entity_id != b.entity_id


def test_kinematic_gate_uses_course_and_speed(cat):
    kts, secs = 10.0, 480
    along = kts * 0.514444 * secs
    ok = FusionEngine(cat)
    ok.ingest(contact(cat, "R1", "HMCS-A", 0, course=90, speed=kts))
    assert ok.ingest(contact(cat, "R2", "HMCS-B", secs, east=along + 100, course=90, speed=kts)).action != "created"
    far = FusionEngine(cat)
    first = far.ingest(contact(cat, "R1", "HMCS-A", 0, course=90, speed=kts))
    assert far.ingest(contact(cat, "R2", "HMCS-B", secs, east=10_000, course=90, speed=kts)).entity_id != first.entity_id
    loose = FusionEngine(cat)
    first = loose.ingest(contact(cat, "R1", "HMCS-A", 0))
    assert loose.ingest(contact(cat, "R2", "HMCS-B", secs, east=10_000)).entity_id == first.entity_id


def test_out_of_order_report_does_not_overwrite_newer_position(cat):
    eng = FusionEngine(cat)
    eng.ingest(contact(cat, "R1", "HMCS-A", 600, east=1000, cls="HOSTILE"))
    late = eng.ingest(contact(cat, "R2", "HMCS-B", 0, east=0, cls="HOSTILE"))
    assert "arrived after newer information" in late.notes
    lat_now = float(eng.snapshot()[0]["fields"]["LAT"]["value"])
    assert lat_now == pytest.approx(float(contact(cat, "x", "HMCS-A", 600, east=1000).msg.fields["LAT"]))
    assert eng.snapshot()[0]["fields"]["LON"]["from_report"] == "R1"


# conflicts and voting

def test_conflicting_class_is_flagged_not_hidden(cat):
    eng = FusionEngine(cat)
    eng.ingest(contact(cat, "R1", "HMCS-A", 0, cls="FRIEND"))
    out = eng.ingest(contact(cat, "R2", "HMCS-B", 20, 50, 50, cls="HOSTILE"))
    assert out.action == "conflict" and any("CLASS differs" in n for n in out.notes)
    row = eng.snapshot()[0]
    assert row["fields"]["CLASS"]["status"] == "conflict"
    assert {a["value"] for a in row["fields"]["CLASS"]["alternatives"]} <= {"FRIEND", "HOSTILE"}
    assert any("disagree on CLASS" in f for f in row["flags"])


def test_majority_outvotes_a_single_wrong_report_without_conflict(cat):
    eng = FusionEngine(cat)
    for i, (src, cls) in enumerate([("HMCS-A", "HOSTILE"), ("HMCS-B", "HOSTILE"), ("SHORE-2", "HOSTILE"), ("HMCS-A", "NEUTRAL")]):
        eng.ingest(contact(cat, f"R{i}", src, i * 10, i * 30.0, cls=cls))
    f = eng.snapshot()[0]["fields"]["CLASS"]
    assert f["value"] == "HOSTILE" and f["status"] == "ok"


def test_unknown_is_weak_evidence(cat):
    eng = FusionEngine(cat)
    eng.ingest(contact(cat, "R1", "HMCS-A", 0, cls="UNKNOWN"))
    eng.ingest(contact(cat, "R2", "HMCS-B", 10, 30, cls="UNKNOWN"))
    eng.ingest(contact(cat, "R3", "SHORE-2", 20, 60, cls="NEUTRAL"))
    f = eng.snapshot()[0]["fields"]["CLASS"]
    assert f["value"] == "NEUTRAL" and f["status"] == "ok"


def test_newer_identification_outweighs_old_one(cat):
    eng = FusionEngine(cat)
    eng.ingest(contact(cat, "R1", "HMCS-A", 0, cls="UNKNOWN"))
    eng.ingest(contact(cat, "R2", "HMCS-A", 120, 20, cls="FRIEND"))
    eng.ingest(contact(cat, "R3", "HMCS-B", 240, 40, cls="FRIEND"))
    assert eng.snapshot()[0]["fields"]["CLASS"]["value"] == "FRIEND"


# other families

def test_cff_merges_close_targets_and_keeps_distant_ones_apart(cat):
    eng = FusionEngine(cat)
    a = eng.ingest(cff(cat, "R1", "B22", 0))
    assert eng.ingest(cff(cat, "R2", "C14B", 600, east=60, north=40)).entity_id == a.entity_id
    assert eng.ingest(cff(cat, "R3", "OP3", 700, east=400)).entity_id != a.entity_id


def test_cff_target_type_votes(cat):
    eng = FusionEngine(cat)
    for i, t in enumerate(["INF", "INF", "INF", "ARM"]):
        eng.ingest(cff(cat, f"R{i}", ["B22", "C14B", "OP3", "B22"][i], i * 30, east=10 * i, tgt=t))
    f = eng.snapshot()[0]["fields"]["TGT"]
    assert f["value"] == "INF" and f["status"] == "ok" and f["alternatives"][0]["value"] == "ARM"


def test_two_to_one_split_is_flagged_as_conflict(cat):
    eng = FusionEngine(cat)
    for i, t in enumerate(["INF", "INF", "ARM"]):
        eng.ingest(cff(cat, f"R{i}", ["B22", "C14B", "OP3"][i], i * 30, east=10 * i, tgt=t))
    f = eng.snapshot()[0]["fields"]["TGT"]
    assert f["value"] == "INF" and f["status"] == "conflict"


def test_logstat_identity_supersedes_and_ignores_stale_updates(cat):
    eng = FusionEngine(cat)
    eng.ingest(logstat(cat, "R1", 0, onhand="500"))
    eng.ingest(logstat(cat, "R2", 1200, onhand="300"))
    late = eng.ingest(logstat(cat, "R3", 600, onhand="400"))
    assert "arrived after newer information" in late.notes
    snap = eng.snapshot()
    assert len(snap) == 1 and snap[0]["fields"]["ONHAND"]["value"] == "300"
    assert eng.ingest(logstat(cat, "R4", 0, item="DIESEL")).entity_id != snap[0]["id"]
    assert eng.ingest(logstat(cat, "R5", 0, unit="2PL")).entity_id != snap[0]["id"]


def test_medevac_same_request_id_merges_even_from_far(cat):
    eng = FusionEngine(cat)
    a = eng.ingest(medevac(cat, "R1", "B22", 0, "MV-001"))
    b = eng.ingest(medevac(cat, "R2", "OP3", 3000, "MV-001", east=500, sec="ENEMY"))
    assert b.entity_id == a.entity_id


def test_medevac_different_ids_are_suggested_not_merged(cat):
    eng = FusionEngine(cat)
    a = eng.ingest(medevac(cat, "R1", "B22", 0, "MV-001"))
    b = eng.ingest(medevac(cat, "R2", "OP3", 300, "RLY-001", east=30))
    assert b.entity_id != a.entity_id and b.action == "created"
    assert any("possible duplicate" in n for n in b.notes)
    assert f"possible duplicate of {a.entity_id}" in {e.id: e for e in eng.entities()}[b.entity_id].flags
    far = eng.ingest(medevac(cat, "R3", "OP3", 300, "MV-009", east=5000))
    assert far.notes == []


# provenance and scale

def test_every_fused_value_points_to_a_report_that_supplied_it(cat):
    sc = make_scenario(cat, 7)
    eng = FusionEngine(cat)
    by_id = {r.id: r for r in sc.reports}
    for r in sc.reports:
        eng.ingest(r)
    for e in eng.snapshot():
        for name, f in e["fields"].items():
            assert by_id[f["from_report"]].msg.fields[name] == f["value"]
            assert any(x["id"] == f["from_report"] for x in e["reports"])
            for alt in f["alternatives"]:
                assert by_id[alt["report"]].msg.fields[name] == alt["value"]


def test_ingest_order_does_not_change_logistics_outcome(cat):
    reps = [logstat(cat, f"R{i}", i * 600, onhand=str(1000 - i * 100)) for i in range(6)]
    results = set()
    for seed in range(5):
        shuffled = reps[:]
        random.Random(seed).shuffle(shuffled)
        eng = FusionEngine(cat)
        for r in shuffled:
            eng.ingest(r)
        results.add(eng.snapshot()[0]["fields"]["ONHAND"]["value"])
    assert results == {"500"}


def test_scenario_scores_meet_floors_and_provenance_is_complete(cat):
    for seed in (101, 102, 103):
        res = evaluate(cat, make_scenario(cat, seed))
        assert res["provenance"]["every_report_in_exactly_one_entity"]
        assert res["provenance"]["fused_fields_trace_to_known_reports"]
        assert res["actions"].get("duplicate_ignored", 0) == res["injected_exact_duplicates"]
        assert res["CONTACT"]["pair_precision"] >= 0.8 and res["CONTACT"]["pair_recall"] >= 0.75
        assert res["CFF"]["pair_precision"] >= 0.95 and res["CFF"]["pair_recall"] >= 0.8
        assert res["MEDEVAC"]["pair_precision"] == 1.0 and res["LOGSTAT"]["pair_recall"] == 1.0
        assert res["LOGSTAT"]["latest_onhand"]["wrong"] == 0
        assert res["MEDEVAC"]["relay_duplicates"]["flagged_as_possible_duplicate"] == res["MEDEVAC"]["relay_duplicates"]["injected"]
        assert res["entities"] < res["reports"] * 0.3
