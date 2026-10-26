"""Fusion demo: multi-source scenario with ground truth, mean over seeds.

    python3 fusion_demo.py [--seeds 20] [--seed 42]

Writes results/fusion_results.json and results/fusion_sample.json (entities from the first seed).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from msgengine.catalogue import Catalogue
from msgengine.fusion import FusionEngine
from msgengine.fusion_sim import evaluate, make_scenario

ROOT = Path(__file__).resolve().parent
FAMILIES = ("CONTACT", "CFF", "MEDEVAC", "LOGSTAT")


def _agg(values: list[float]) -> dict:
    return {"mean": round(statistics.fmean(values), 3), "min": round(min(values), 3), "max": round(max(values), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cat = Catalogue.load(ROOT / "catalogue" / "demo_catalogue.json")
    runs = [evaluate(cat, make_scenario(cat, a.seed + k)) for k in range(a.seeds)]

    summary: dict = {"seeds": a.seeds, "families": {}}
    print(f"{a.seeds} scenarios, seeds {a.seed}..{a.seed + a.seeds - 1}; reports per scenario ~{statistics.fmean(r['reports'] for r in runs):.0f}\n")
    print(f"{'family':8s} {'reports':>8s} {'entities':>9s} {'truth':>6s} {'pair prec':>10s} {'pair rec':>9s} (mean; min in brackets)")
    for fam in FAMILIES:
        row = {k: _agg([r[fam][k] for r in runs]) for k in ("reports", "entities", "truth_entities", "pair_precision", "pair_recall")}
        summary["families"][fam] = row
        print(f"{fam:8s} {row['reports']['mean']:8.0f} {row['entities']['mean']:9.1f} {row['truth_entities']['mean']:6.1f} "
              f"{row['pair_precision']['mean']:6.3f} [{row['pair_precision']['min']:.3f}] {row['pair_recall']['mean']:5.3f} [{row['pair_recall']['min']:.3f}]")

    print("\nFused attribute vs truth (summed over seeds):")
    for fam, key, label in (("CONTACT", "class", "contact class"), ("CFF", "target_type", "target type"),
                            ("MEDEVAC", "security", "medevac security"), ("LOGSTAT", "latest_onhand", "latest on-hand")):
        tot = {k: sum(r[fam][key][k] for r in runs) for k in ("entities", "correct", "unknown", "wrong", "wrong_and_flagged")}
        summary["families"][fam][key] = tot
        n = tot["entities"]
        print(f"  {label:17s} entities {n:5d}  correct {tot['correct']/n:6.1%}  unknown {tot['unknown']/n:6.1%}  "
              f"wrong {tot['wrong']/n:6.1%}  (of wrong, flagged as conflict {tot['wrong_and_flagged']}/{tot['wrong']})")
    raw = {k: sum(r["CONTACT"]["raw_report_class"].get(k, 0) for r in runs) for k in ("correct", "unknown", "wrong")}
    n = sum(raw.values())
    summary["raw_contact_report_class"] = raw
    print(f"  single contact reports: correct {raw['correct']/n:.1%}  unknown {raw['unknown']/n:.1%}  wrong {raw['wrong']/n:.1%}")

    relay = {k: sum(r["MEDEVAC"]["relay_duplicates"][k] for r in runs) for k in ("injected", "flagged_as_possible_duplicate")}
    summary["relay_duplicates"] = relay
    print(f"\nMEDEVAC relayed under a different request id: {relay['flagged_as_possible_duplicate']} of {relay['injected']} flagged as possible duplicates (not merged)")
    dup = sum(r["actions"].get("duplicate_ignored", 0) for r in runs)
    inj = sum(r["injected_exact_duplicates"] for r in runs)
    summary["exact_duplicates"] = {"injected": inj, "ignored": dup}
    print(f"Exact resends: {dup} ignored of {inj} injected")
    acts = {}
    for r in runs:
        for k, v in r["actions"].items():
            acts[k] = acts.get(k, 0) + v
    summary["actions"] = acts
    print("Actions: " + ", ".join(f"{k} {v}" for k, v in sorted(acts.items())))
    prov = all(r["provenance"]["every_report_in_exactly_one_entity"] and r["provenance"]["fused_fields_trace_to_known_reports"] for r in runs)
    summary["provenance_ok_in_all_runs"] = prov
    print(f"Provenance complete in all runs: {prov}")
    total_reports = sum(r["reports"] for r in runs)
    total_entities = sum(r["entities"] for r in runs)
    print(f"Operator items: {total_reports} reports became {total_entities} entities ({total_entities/total_reports:.0%})")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "fusion_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    eng = FusionEngine(cat)
    for rep in make_scenario(cat, a.seed).reports:
        eng.ingest(rep)
    (out / "fusion_sample.json").write_text(json.dumps(eng.snapshot()[:12], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
