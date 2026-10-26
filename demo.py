"""Proof-of-concept demo: measures round-trip fidelity, size and corruption behaviour on synthetic data.

    python3 demo.py [--n 400] [--seed 42]

Writes results/results.json, results/samples.txt and results/demo_catalogue.xsd.
"""
from __future__ import annotations

import argparse
import json
import random
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from msgengine.catalogue import Catalogue
from msgengine.codec_bin import BinaryError, decode, encode
from msgengine.codec_char import parse, render
from msgengine.codec_xml import build_xsd, from_xml, to_xml, validate_xml
from msgengine.generator import NOISE_KINDS, corrupt, make_message

ROOT = Path(__file__).resolve().parent


def _pct(a: int, b: int) -> float:
    return round(100.0 * a / b, 2) if b else 0.0


def run(n: int, seed: int) -> dict:
    cat = Catalogue.load(ROOT / "catalogue" / "demo_catalogue.json")
    rng = random.Random(seed)
    types = sorted(cat.messages)
    msgs = [make_message(cat, rng, types[i % len(types)]) for i in range(n)]

    fidelity = Counter()
    sizes: dict[str, defaultdict] = {t: defaultdict(int) for t in [*types, "ALL"]}
    for msg in msgs:
        text = render(cat, msg)
        xml = to_xml(cat, msg)
        frame = encode(cat, msg)
        fidelity["total"] += 1
        fidelity["char_parse_ok"] += parse(cat, text).message == msg
        fidelity["xml_xsd_valid"] += not validate_xml(cat, xml)
        fidelity["char_to_xml_to_char_identical"] += render(cat, from_xml(cat, xml)) == text
        fidelity["binary_round_trip_ok"] += decode(cat, frame) == msg
        for key in (msg.type, "ALL"):
            s = sizes[key]
            s["count"] += 1
            s["char_bytes"] += len(text.encode("utf-8"))
            s["xml_bytes"] += len(xml)
            s["xml_zlib_bytes"] += len(zlib.compress(xml, 9))
            s["binary_bytes"] += len(frame)

    size_table = {}
    for key, s in sizes.items():
        size_table[key] = {
            **dict(s),
            "binary_vs_xml_pct": _pct(s["binary_bytes"], s["xml_bytes"]),
            "binary_vs_char_pct": _pct(s["binary_bytes"], s["char_bytes"]),
            "xml_zlib_vs_xml_pct": _pct(s["xml_zlib_bytes"], s["xml_bytes"]),
        }

    noise_rng = random.Random(seed + 1)
    char_noise: dict[str, Counter] = {k: Counter() for k in NOISE_KINDS}
    for msg in msgs:
        text = render(cat, msg)
        for kind in NOISE_KINDS:
            bad = corrupt(text, kind, noise_rng)
            r = parse(cat, bad)
            if r.issues:
                outcome = "flagged"
            elif r.message == msg:
                outcome = "recovered_identical"
            else:
                outcome = "silent_change"
            char_noise[kind][outcome] += 1

    bin_noise = Counter()
    for msg in msgs:
        frame = bytearray(encode(cat, msg))
        for _ in range(10):
            f = bytearray(frame)
            i = noise_rng.randrange(len(f) * 8)
            f[i // 8] ^= 1 << (i % 8)
            try:
                out = decode(cat, bytes(f))
            except BinaryError:
                bin_noise["detected"] += 1
                continue
            bin_noise["undetected_identical" if out == msg else "silent_change"] += 1

    return {
        "seed": seed,
        "messages": n,
        "catalogue": {"name": cat.name, "version": cat.version, "families": types},
        "fidelity": dict(fidelity),
        "sizes": size_table,
        "char_noise": {k: dict(v) for k, v in char_noise.items()},
        "binary_bit_flip": dict(bin_noise),
    }


def write_artifacts(results: dict, out: Path) -> None:
    cat = Catalogue.load(ROOT / "catalogue" / "demo_catalogue.json")
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "demo_catalogue.xsd").write_bytes(build_xsd(cat))
    rng = random.Random(results["seed"])
    lines = []
    for t in sorted(cat.messages):
        msg = make_message(cat, rng, t)
        frame = encode(cat, msg)
        lines += [f"== {t} ({cat.message(t).title}) ==", render(cat, msg), to_xml(cat, msg).decode(),
                  f"binary ({len(frame)} bytes): {frame.hex()}", ""]
    (out / "samples.txt").write_text("\n".join(lines), encoding="utf-8")


def print_report(r: dict) -> None:
    n = r["messages"]
    print(f"Synthetic messages: {n} (seed {r['seed']}), families: {', '.join(r['catalogue']['families'])}")
    print("\nFidelity (must all equal total):")
    for k, v in r["fidelity"].items():
        print(f"  {k:32s} {v}")
    print("\nSize in bytes (mean per message):")
    print(f"  {'type':9s} {'char':>6s} {'xml':>6s} {'xml+zlib':>9s} {'binary':>7s} {'bin/xml':>8s} {'bin/char':>9s}")
    for k, s in r["sizes"].items():
        c = s["count"]
        print(f"  {k:9s} {s['char_bytes']/c:6.1f} {s['xml_bytes']/c:6.1f} {s['xml_zlib_bytes']/c:9.1f} "
              f"{s['binary_bytes']/c:7.1f} {s['binary_vs_xml_pct']:7.1f}% {s['binary_vs_char_pct']:8.1f}%")
    print("\nCharacter-text corruption outcomes (per kind):")
    for kind, c in r["char_noise"].items():
        print(f"  {kind:18s} " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    b = r["binary_bit_flip"]
    total = sum(b.values())
    print(f"\nBinary frame single-bit flips ({total} trials): "
          + "  ".join(f"{k}={v}" for k, v in sorted(b.items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    res = run(args.n, args.seed)
    write_artifacts(res, ROOT / "results")
    print_report(res)
