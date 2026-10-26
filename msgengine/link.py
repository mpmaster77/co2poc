"""Discrete-event emulation of a constrained link (virtual time, seeded, no real sockets).

Forward and return channels share one outage schedule. Loss follows a two-state burst model.
Transfer: messages are sent one at a time, split into fragments, with selective acknowledgement,
retransmission with exponential back-off, and a per-message deadline. Without a return channel the
sender repeats every fragment a fixed number of times and cannot confirm delivery.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass

from .catalogue import Catalogue
from .codec_bin import BinaryError, decode, encode
from .codec_char import parse, render
from .codec_xml import XmlError, from_xml, to_xml
from .delta import baseline_id, decode_delta, encode_delta, frame_baseline_id, is_delta
from .model import Message

ENCODINGS = ("xml", "char", "binary", "delta")


@dataclass(frozen=True)
class LinkProfile:
    name: str
    rate_bps: float
    latency_s: float
    loss_good: float
    loss_bad: float = 0.0
    p_good_to_bad: float = 0.0
    p_bad_to_good: float = 1.0
    outage_period_s: float = 0.0
    outage_duration_s: float = 0.0
    return_channel: bool = True


class Link:
    def __init__(self, profile: LinkProfile, rng: random.Random, phase: float) -> None:
        self.p, self.rng, self.phase, self.bad = profile, rng, phase, False

    def down(self, t0: float, t1: float) -> bool:
        p = self.p
        if not p.outage_period_s or not p.outage_duration_s:
            return False
        k0 = math.floor((t0 - self.phase) / p.outage_period_s)
        k1 = math.floor((t1 - self.phase) / p.outage_period_s)
        for k in range(k0, k1 + 1):
            start = k * p.outage_period_s + self.phase
            if start < t1 and start + p.outage_duration_s > t0:
                return True
        return False

    def delivers(self, t0: float, t1: float) -> bool:
        if self.bad:
            if self.rng.random() < self.p.p_bad_to_good:
                self.bad = False
        elif self.rng.random() < self.p.p_good_to_bad:
            self.bad = True
        if self.down(t0, t1):
            return False
        return self.rng.random() >= (self.p.loss_bad if self.bad else self.p.loss_good)


@dataclass
class Item:
    track: int
    arrival: float
    msg: Message


def _payload(cat: Catalogue, encoding: str, item: Item, baseline: Message | None) -> bytes:
    if encoding == "xml":
        return to_xml(cat, item.msg)
    if encoding == "char":
        return render(cat, item.msg).encode("utf-8")
    full = encode(cat, item.msg)
    if encoding == "binary" or baseline is None:
        return full
    delta = encode_delta(cat, baseline, item.msg)
    return delta if len(delta) < len(full) else full


def _receive(cat: Catalogue, encoding: str, payload: bytes, store: dict[bytes, Message]) -> Message | None:
    try:
        if encoding == "xml":
            return from_xml(cat, payload)
        if encoding == "char":
            return parse(cat, payload.decode("utf-8")).message
        if is_delta(payload):
            base = store.get(frame_baseline_id(payload))
            return decode_delta(cat, payload, base) if base else None
        return decode(cat, payload)
    except (BinaryError, XmlError, UnicodeDecodeError, ValueError):
        return None


def simulate(cat: Catalogue, profile: LinkProfile, items: list[Item], encoding: str, seed: int = 1,
             mtu: int = 96, overhead: int = 6, deadline_s: float = 900.0, base_rto_s: float | None = None,
             max_rto_s: float = 60.0, repeat: int = 3) -> dict:
    if encoding not in ENCODINGS:
        raise ValueError(encoding)
    rng = random.Random(seed)
    phase = rng.uniform(0, profile.outage_period_s) if profile.outage_period_s else 0.0
    fwd = Link(profile, random.Random(seed * 7 + 1), phase)
    rev = Link(profile, random.Random(seed * 7 + 2), phase)
    bps = profile.rate_bps
    rto0 = base_rto_s or (2 * profile.latency_s + (mtu + overhead) * 8 / bps + 0.5)

    fwd_free = rev_free = sender_free = 0.0
    receiver_store: dict[int, dict[bytes, Message]] = {}
    sender_baseline: dict[int, Message] = {}
    records = []
    wire_total = 0

    for item in sorted(items, key=lambda i: i.arrival):
        t = max(item.arrival, sender_free)
        deadline = item.arrival + deadline_s
        baseline = sender_baseline.get(item.track) if profile.return_channel else None
        payload = _payload(cat, encoding, item, baseline)
        frags = [payload[i:i + mtu] for i in range(0, len(payload), mtu)] or [b""]
        n = len(frags)
        got: set[int] = set()
        recv_done: float | None = None
        acked: set[int] = set()
        pending: list[tuple[float, frozenset[int]]] = []
        confirmed_at: float | None = None
        wire = packets = 0

        def send(idx: int, start_at: float) -> float:
            nonlocal fwd_free, rev_free, wire, packets, recv_done
            tx_start = max(start_at, fwd_free)
            size = overhead + len(frags[idx])
            tx_end = tx_start + size * 8 / bps
            fwd_free = tx_end
            wire += size
            packets += 1
            if fwd.delivers(tx_start, tx_end):
                arrive = tx_end + profile.latency_s
                got.add(idx)
                if len(got) == n and recv_done is None:
                    recv_done = arrive
                if profile.return_channel:
                    ack_size = 4 + (n + 7) // 8
                    a_start = max(arrive, rev_free)
                    a_end = a_start + ack_size * 8 / bps
                    rev_free = a_end
                    wire += ack_size
                    if rev.delivers(a_start, a_end):
                        pending.append((a_end + profile.latency_s, frozenset(got)))
            return tx_end

        if not profile.return_channel:
            for _ in range(repeat):
                for idx in range(n):
                    t = send(idx, t)
            sender_free = t
        else:
            rto, now = rto0, t
            while now < deadline:
                for idx in sorted(set(range(n)) - acked):
                    send(idx, now)
                timeout_at = fwd_free + rto
                for arrive, snap in sorted(pending):
                    if arrive <= timeout_at:
                        acked |= snap
                        if len(acked) == n and confirmed_at is None:
                            confirmed_at = arrive
                            break
                if confirmed_at is not None:
                    break
                now, rto = timeout_at, min(rto * 2, max_rto_s)
            sender_free = confirmed_at if confirmed_at is not None else max(now, fwd_free)
            if confirmed_at is not None:
                sender_baseline[item.track] = item.msg

        verified = False
        if recv_done is not None:
            store = receiver_store.setdefault(item.track, {})
            out = _receive(cat, encoding, payload, store)
            verified = out == item.msg
            if verified:
                store[baseline_id(cat, item.msg)] = item.msg
        wire_total += wire
        records.append({"arrival": item.arrival, "recv_done": recv_done, "verified": verified,
                        "confirmed": confirmed_at is not None, "payload": len(payload), "frags": n,
                        "packets": packets, "wire": wire})

    delivered = [r for r in records if r["recv_done"] is not None]
    lat = sorted(r["recv_done"] - r["arrival"] for r in delivered)
    return {
        "profile": profile.name, "encoding": encoding, "messages": len(records),
        "delivered": len(delivered), "verified": sum(r["verified"] for r in records),
        "confirmed_to_sender": sum(r["confirmed"] for r in records),
        "latency_mean_s": round(statistics.fmean(lat), 2) if lat else None,
        "latency_p95_s": round(lat[min(len(lat) - 1, math.ceil(0.95 * len(lat)) - 1)], 2) if lat else None,
        "payload_mean_bytes": round(statistics.fmean(r["payload"] for r in records), 1),
        "packets_sent": sum(r["packets"] for r in records),
        "wire_bytes_total": wire_total,
        "wire_bytes_per_delivered": round(wire_total / len(delivered), 1) if delivered else None,
        "last_delivery_s": round(max(r["recv_done"] for r in delivered), 1) if delivered else None,
    }
