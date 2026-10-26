import random

import pytest

from msgengine.codec_char import escape, parse, render, unescape
from msgengine.generator import NOISE_KINDS, corrupt, make_message
from msgengine.model import Message

CFF = ("CFF/OBS=B22/DTG=241830ZOCT26/TGTGRID=18SUJ2338706520/TGT=INF/MISSION=FFE//")


def test_parse_valid_message(cat):
    r = parse(cat, CFF)
    assert r.ok
    assert r.message == Message("CFF", {"OBS": "B22", "DTG": "241830ZOCT26", "TGTGRID": "18SUJ2338706520",
                                         "TGT": "INF", "MISSION": "FFE"})


def test_line_wrapping_and_whitespace_tolerated(cat):
    wrapped = "CFF/OBS = B22 /DTG=241830ZOCT26\n/TGTGRID=18SUJ2338706520/TGT=INF\r\n/MISSION=FFE//\n"
    assert parse(cat, wrapped).message == parse(cat, CFF).message
    assert parse(cat, wrapped).ok


def test_field_order_is_free(cat):
    shuffled = "CFF/MISSION=FFE/TGT=INF/TGTGRID=18SUJ2338706520/DTG=241830ZOCT26/OBS=B22//"
    assert parse(cat, shuffled).message == parse(cat, CFF).message


def test_escaping_round_trip(cat):
    for raw in ["a/b", "x=y", "t~e", "~S literal", "all / = ~ together"]:
        assert unescape(escape(raw)) == raw
    msg = Message("CFF", {**parse(cat, CFF).message.fields, "DESC": "3/4 strength = low ~ maybe"})
    text = render(cat, msg)
    assert text.count("/") == len(msg.fields) + 1 + 1  # separators + terminator slash pair
    assert parse(cat, text).message == msg


def test_bad_escape_flagged(cat):
    r = parse(cat, CFF.replace("//", "/DESC=bad~X//"))
    assert [i.code for i in r.issues] == ["INVALID_VALUE"]


@pytest.mark.parametrize("text, code", [
    (CFF[:-2], "TRUNCATED"),
    (CFF[:30], "TRUNCATED"),
    ("XYZ/A=1//", "UNKNOWN_TYPE"),
    (CFF.replace("TGT=INF", "TGT=BOGUS"), "INVALID_VALUE"),
    (CFF.replace("/MISSION=FFE", ""), "MISSING_REQUIRED"),
    (CFF.replace("/TGT=INF", "/TGT=INF/TGT=ARM"), "DUPLICATE_FIELD"),
    (CFF.replace("/TGT=INF", "/NOPE=1/TGT=INF"), "UNKNOWN_FIELD"),
    (CFF.replace("/TGT=INF", "/TGT"), "MALFORMED_FIELD"),
    (CFF.replace("/TGT=INF", "//TGT=INF"), "EMPTY_SEGMENT"),
    (CFF.replace("18SUJ", "99SUJ"), "INVALID_VALUE"),
])
def test_problems_are_reported_not_raised(cat, text, code):
    r = parse(cat, text)
    assert not r.ok
    assert code in [i.code for i in r.issues]


def test_partial_message_kept_when_type_known(cat):
    r = parse(cat, CFF[:40])
    assert r.message is not None and r.message.type == "CFF"
    assert r.message.fields.get("OBS") == "B22"


def test_unknown_type_returns_no_message(cat):
    assert parse(cat, "XYZ/A=1//").message is None


@pytest.mark.parametrize("garbage", ["", "//", "/", "///", "=", "CFF", "CFF//", "\x00\x01", "CFF/=/=//"])
def test_parser_never_raises(cat, garbage):
    parse(cat, garbage)


def test_render_parse_round_trip_many(cat):
    rng = random.Random(7)
    for _ in range(500):
        msg = make_message(cat, rng)
        r = parse(cat, render(cat, msg))
        assert r.ok and r.message == msg


@pytest.mark.parametrize("kind", NOISE_KINDS)
def test_corruption_never_crashes_parser(cat, kind):
    rng = random.Random(3)
    for _ in range(200):
        text = render(cat, make_message(cat, rng))
        parse(cat, corrupt(text, kind, rng))


def test_recoverable_noise_is_recovered_exactly(cat):
    rng = random.Random(11)
    for kind in ("extra_whitespace", "shuffle_fields"):
        for _ in range(100):
            msg = make_message(cat, rng)
            r = parse(cat, corrupt(render(cat, msg), kind, rng))
            assert r.ok and r.message == msg


def test_structural_damage_is_always_flagged(cat):
    rng = random.Random(5)
    for kind in ("truncate", "drop_terminator", "duplicate_segment"):
        for _ in range(100):
            text = render(cat, make_message(cat, rng))
            assert parse(cat, corrupt(text, kind, rng)).issues, kind
