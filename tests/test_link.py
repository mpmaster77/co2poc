import pytest

from msgengine.link import ENCODINGS, Link, LinkProfile, simulate
from msgengine.workload import make_stream

CLEAN = LinkProfile("clean", rate_bps=9600, latency_s=0.5, loss_good=0.0)
LOSSY = LinkProfile("lossy", rate_bps=2400, latency_s=1.0, loss_good=0.05, loss_bad=0.6,
                    p_good_to_bad=0.05, p_bad_to_good=0.3, outage_period_s=300, outage_duration_s=60)
BROADCAST = LinkProfile("broadcast", rate_bps=200, latency_s=0.1, loss_good=0.1, return_channel=False)


@pytest.fixture(scope="module")
def stream(cat):
    return make_stream(cat, seed=5, tracks=6, updates=5, interval_s=60)


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_clean_link_delivers_everything_verified(cat, stream, encoding):
    r = simulate(cat, CLEAN, stream, encoding, seed=1)
    assert r["messages"] == len(stream)
    assert r["delivered"] == r["verified"] == r["confirmed_to_sender"] == len(stream)


def test_clean_link_latency_matches_arithmetic(cat, stream):
    r = simulate(cat, CLEAN, stream[:1], "binary", seed=1)
    size = r["payload_mean_bytes"] + 6
    expected = size * 8 / 9600 + 0.5
    assert r["latency_mean_s"] == pytest.approx(expected, abs=0.02)


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_lossy_link_never_delivers_a_wrong_message(cat, stream, encoding):
    r = simulate(cat, LOSSY, stream, encoding, seed=2)
    assert r["verified"] == r["delivered"]
    assert r["delivered"] > 0


def test_simulation_is_deterministic(cat, stream):
    assert simulate(cat, LOSSY, stream, "delta", seed=3) == simulate(cat, LOSSY, stream, "delta", seed=3)


def test_loss_costs_extra_bytes(cat, stream):
    clean = simulate(cat, CLEAN, stream, "binary", seed=1)
    lossy = simulate(cat, LOSSY, stream, "binary", seed=1)
    assert lossy["packets_sent"] > clean["packets_sent"]


def test_smaller_encoding_uses_fewer_wire_bytes(cat, stream):
    xml = simulate(cat, CLEAN, stream, "xml", seed=1)["wire_bytes_total"]
    binary = simulate(cat, CLEAN, stream, "binary", seed=1)["wire_bytes_total"]
    delta = simulate(cat, CLEAN, stream, "delta", seed=1)["wire_bytes_total"]
    assert delta < binary < xml


def test_outage_only_delays_never_loses(cat, stream):
    outage = LinkProfile("outage", 9600, 0.5, 0.0, outage_period_s=120, outage_duration_s=45)
    r = simulate(cat, outage, stream, "binary", seed=4)
    assert r["delivered"] == r["verified"] == len(stream)
    assert r["latency_p95_s"] > simulate(cat, CLEAN, stream, "binary", seed=4)["latency_p95_s"]


def test_hopeless_link_reports_undelivered_instead_of_pretending(cat, stream):
    dead = LinkProfile("dead", 1200, 1.0, loss_good=1.0)
    r = simulate(cat, dead, stream[:3], "binary", seed=1, deadline_s=60)
    assert r["delivered"] == r["verified"] == r["confirmed_to_sender"] == 0


def test_broadcast_mode_has_no_confirmation_and_no_delta(cat, stream):
    r = simulate(cat, BROADCAST, stream, "delta", seed=6)
    assert r["confirmed_to_sender"] == 0
    assert r["verified"] == r["delivered"]
    full = simulate(cat, BROADCAST, stream, "binary", seed=6)
    assert r["payload_mean_bytes"] == full["payload_mean_bytes"]


def test_outage_window_logic():
    link = Link(LinkProfile("x", 1000, 0.1, 0.0, outage_period_s=100, outage_duration_s=10), __import__("random").Random(0), 20.0)
    assert link.down(25, 26)
    assert link.down(15, 21)
    assert not link.down(31, 40)
    assert link.down(119, 125)
    assert not link.down(40, 119)
