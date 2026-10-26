import random

import pytest

from msgengine.codec_bin import BinaryError, decode, encode
from msgengine.codec_char import parse, render
from msgengine.codec_xml import to_xml
from msgengine.generator import make_message
from msgengine.model import Message

CFF_TEXT = "CFF/OBS=B22/DTG=241830ZOCT26/TGTGRID=18SUJ2338706520/TGT=INF/MISSION=FFE//"


def test_binary_round_trip_many(cat):
    rng = random.Random(31)
    for _ in range(1000):
        msg = make_message(cat, rng)
        assert decode(cat, encode(cat, msg)) == msg


def test_edge_values_round_trip(cat):
    cases = [
        Message("CONTACT", {"REPORTER": "X1", "DTG": "010000ZJAN00", "DOMAIN": "AIR", "LAT": "-90.0000",
                            "LON": "180.0000", "CLASS": "HOSTILE", "SPEED": "999.9", "COURSE": "359"}),
        Message("CONTACT", {"REPORTER": "X1", "DTG": "312359ZDEC99", "DOMAIN": "LAND", "LAT": "0.0000",
                            "LON": "-180.0000", "CLASS": "UNKNOWN"}),
        Message("MEDEVAC", {"REQID": "M", "DTG": "290000ZFEB28", "PICKUP": "1CCA00", "FREQ": "30.000",
                            "PRECEDENCE": "URGENT", "SECURITY": "ENEMY"}),
        Message("CFF", {"OBS": "E1", "DTG": "241830ZOCT26", "TGTGRID": "60XVV9999999999", "TGT": "INF",
                        "MISSION": "FFE", "DESC": "Éé 日本語 ünï"}),
    ]
    for msg in cases:
        for k, v in list(msg.fields.items()):
            msg.fields[k] = cat.message(msg.type).field(k).normalize(v)
        assert decode(cat, encode(cat, msg)) == msg


def test_binary_is_much_smaller_than_xml_and_text(cat):
    rng = random.Random(32)
    xml = text = binary = 0
    for _ in range(300):
        msg = make_message(cat, rng, p_optional=0.0)
        xml += len(to_xml(cat, msg))
        text += len(render(cat, msg).encode())
        binary += len(encode(cat, msg))
    assert binary < text < xml
    assert binary < xml / 4


def test_any_single_bit_flip_is_detected_or_harmless(cat):
    rng = random.Random(33)
    for _ in range(200):
        msg = make_message(cat, rng)
        frame = bytearray(encode(cat, msg))
        i = rng.randrange(len(frame) * 8)
        frame[i // 8] ^= 1 << (i % 8)
        try:
            out = decode(cat, bytes(frame))
        except BinaryError:
            continue
        assert out == msg  # only acceptable if the content is unchanged


def test_truncated_frames_rejected(cat):
    frame = encode(cat, parse(cat, CFF_TEXT).message)
    for n in range(len(frame)):
        with pytest.raises(BinaryError):
            decode(cat, frame[:n])


def test_wrong_catalogue_tag_and_trailing_bytes(cat):
    frame = encode(cat, parse(cat, CFF_TEXT).message)
    with pytest.raises(BinaryError, match="tag"):
        decode(cat, bytes([frame[0] ^ 0xFF]) + frame[1:])
    with pytest.raises(BinaryError):
        decode(cat, frame[:-2] + b"\x00" + frame[-2:])


def test_random_bytes_never_crash(cat):
    rng = random.Random(34)
    for _ in range(2000):
        data = bytes(rng.randrange(256) for _ in range(rng.randint(0, 40)))
        try:
            decode(cat, data)
        except BinaryError:
            pass
