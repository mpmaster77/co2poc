import random

import pytest

from msgengine.codec_bin import BinaryError, encode
from msgengine.delta import decode_delta, encode_delta, frame_baseline_id, is_delta, baseline_id
from msgengine.generator import make_message
from msgengine.model import Message
from msgengine.workload import update


def test_delta_round_trip_many(cat):
    rng = random.Random(41)
    for _ in range(500):
        base = make_message(cat, rng)
        new = update(cat, base, rng)
        frame = encode_delta(cat, base, new)
        assert is_delta(frame)
        assert decode_delta(cat, frame, base) == new


def test_delta_handles_added_and_removed_optional_fields(cat):
    rng = random.Random(42)
    for _ in range(300):
        a, b = make_message(cat, rng, "CFF"), make_message(cat, rng, "CFF")
        assert decode_delta(cat, encode_delta(cat, a, b), a) == b


def test_unchanged_message_gives_tiny_delta(cat):
    msg = make_message(cat, random.Random(1))
    frame = encode_delta(cat, msg, msg)
    assert decode_delta(cat, frame, msg) == msg
    assert len(frame) <= 10


def test_typical_update_is_much_smaller_than_full_frame(cat):
    rng = random.Random(43)
    full = delta = 0
    for _ in range(300):
        base = make_message(cat, rng)
        new = update(cat, base, rng)
        full += len(encode(cat, new))
        delta += len(encode_delta(cat, base, new))
    assert delta < full * 0.5


def test_wrong_baseline_rejected(cat):
    rng = random.Random(44)
    base = make_message(cat, rng, "LOGSTAT")
    other = make_message(cat, rng, "LOGSTAT")
    frame = encode_delta(cat, base, update(cat, base, rng))
    with pytest.raises(BinaryError, match="baseline"):
        decode_delta(cat, frame, other)
    with pytest.raises(BinaryError):
        decode_delta(cat, frame, make_message(cat, rng, "CFF"))


def test_frame_exposes_baseline_id(cat):
    base = make_message(cat, random.Random(2))
    frame = encode_delta(cat, base, base)
    assert frame_baseline_id(frame) == baseline_id(cat, base)


def test_type_mismatch_and_missing_required_refused(cat):
    rng = random.Random(45)
    a, b = make_message(cat, rng, "CFF"), make_message(cat, rng, "MEDEVAC")
    with pytest.raises(ValueError):
        encode_delta(cat, a, b)
    broken = Message("CFF", {k: v for k, v in a.fields.items() if k != "OBS"})
    with pytest.raises(ValueError):
        encode_delta(cat, a, broken)


def test_bit_flips_never_yield_a_different_message(cat):
    rng = random.Random(46)
    for _ in range(300):
        base = make_message(cat, rng)
        new = update(cat, base, rng)
        frame = bytearray(encode_delta(cat, base, new))
        i = rng.randrange(len(frame) * 8)
        frame[i // 8] ^= 1 << (i % 8)
        try:
            out = decode_delta(cat, bytes(frame), base)
        except BinaryError:
            continue
        assert out == new


def test_truncated_and_random_frames_never_crash(cat):
    rng = random.Random(47)
    base = make_message(cat, rng)
    frame = encode_delta(cat, base, update(cat, base, rng))
    for n in range(len(frame)):
        with pytest.raises(BinaryError):
            decode_delta(cat, frame[:n], base)
    for _ in range(1000):
        data = bytes([frame[0], frame[1] | 0x80]) + bytes(rng.randrange(256) for _ in range(rng.randint(0, 30)))
        try:
            decode_delta(cat, data, base)
        except BinaryError:
            pass
