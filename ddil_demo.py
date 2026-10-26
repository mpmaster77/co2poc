"""DDIL emulation demo: same update stream over representative links, four encodings.

    python3 ddil_demo.py [--tracks 12] [--updates 8] [--interval 45] [--seed 42] [--runs 20]

Each cell is the mean over --runs link seeds (same traffic, different loss/outage draws).
Link profiles are representative, unclassified assumptions, not measurements of any real radio.
Writes results/ddil_results.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from msgengine.catalogue import Catalogue
from msgengine.link import ENCODINGS, LinkProfile, simulate
from msgengine.workload import make_stream

ROOT = Path(__file__).resolve().parent

PROFILES = (
    LinkProfile("HF 1.2 kbps", 1200, 1.5, 0.03, 0.6, 0.02, 0.25, outage_period_s=420, outage_duration_s=60),
    LinkProfile("HF 2.4 kbps", 2400, 1.5, 0.03, 0.6, 0.02, 0.25, outage_period_s=420, outage_duration_s=60),
    LinkProfile("UHF SATCOM 9.6 kbps", 9600, 0.6, 0.02, outage_period_s=300, outage_duration_s=30),
    LinkProfile("Degraded IP 64 kbps", 64000, 0.25, 0.10, outage_period_s=180, outage_duration_s=20),
    LinkProfile("VLF/LF broadcast 200 bps (one-way)", 200, 0.1, 0.10, return_channel=False),
)

METRICS = ("delivered", "verified", "confirmed_to_sender", "latency_mean_s", "latency_p95_s",
           "payload_mean_bytes", "wire_bytes_per_delivered", "wire_bytes_total")


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r[key] is not None]
    return round(statistics.fmean(vals), 1) if vals else None


def _cell(v: float | None, width: int) -> str:
    return f"{v:>{width}}" if v is not None else f"{'-':>{width}}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, default=12)
    ap.add_argument("--updates", type=int, default=8)
    ap.add_argument("--interval", type=float, default=45.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs", type=int, default=20)
    a = ap.parse_args()

    cat = Catalogue.load(ROOT / "catalogue" / "demo_catalogue.json")
    stream = make_stream(cat, a.seed, a.tracks, a.updates, a.interval)
    n = len(stream)
    print(f"Stream: {a.tracks} tracks x {a.updates + 1} reports = {n} messages, ~{a.interval:.0f}s apart per track; "
          f"mean of {a.runs} link seeds\n")
    summary = []
    for p in PROFILES:
        print(p.name)
        print(f"  {'encoding':8s} {'deliv':>6s} {'verif':>6s} {'conf':>5s} {'mean s':>8s} {'p95 s':>8s} "
              f"{'payload B':>9s} {'wire B/msg':>10s} {'wire total':>10s}")
        for enc in ENCODINGS:
            rows = [simulate(cat, p, stream, enc, seed=a.seed + k) for k in range(a.runs)]
            cell = {"profile": p.name, "encoding": enc, "messages": n, "runs": a.runs,
                    **{k: _mean(rows, k) for k in METRICS},
                    "min_delivered": min(r["delivered"] for r in rows),
                    "verified_equals_delivered_in_all_runs": all(r["verified"] == r["delivered"] for r in rows)}
            summary.append(cell)
            print(f"  {enc:8s} {_cell(cell['delivered'], 6)} {_cell(cell['verified'], 6)} "
                  f"{_cell(cell['confirmed_to_sender'], 5)} {_cell(cell['latency_mean_s'], 8)} "
                  f"{_cell(cell['latency_p95_s'], 8)} {_cell(cell['payload_mean_bytes'], 9)} "
                  f"{_cell(cell['wire_bytes_per_delivered'], 10)} {_cell(cell['wire_bytes_total'], 10)}")
        print()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "ddil_results.json").write_text(json.dumps({"args": vars(a), "results": summary}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
