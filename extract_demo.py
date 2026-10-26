"""Free-text extraction evaluation on hand-written messages.

    python3 extract_demo.py

data/handwritten.json: written first, then used while developing (so it is a development set).
data/holdout.json: written after the cue lists were fixed, run once before any further change.
Writes results/extract_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from msgengine.catalogue import Catalogue
from msgengine.evaluate import Score
from msgengine.extract import extract

ROOT = Path(__file__).resolve().parent


def run_set(cat: Catalogue, path: Path) -> tuple[Score, dict[str, Score], list[dict]]:
    total, by_lang, rows = Score(), {}, []
    for case in json.loads(path.read_text(encoding="utf-8")):
        ext = extract(cat, case["text"])
        total.add(case["id"], ext, case["type"], case["expect"])
        by_lang.setdefault(case["lang"], Score()).add(case["id"], ext, case["type"], case["expect"])
        rows.append({"id": case["id"], "type": ext.type, "type_confidence": ext.type_confidence,
                     "review": ext.review_reasons(), "draft": ext.draft_text(cat)})
    return total, by_lang, rows


def main() -> None:
    cat = Catalogue.load(ROOT / "catalogue" / "demo_catalogue.json")
    report = {}
    for name in ("handwritten", "holdout"):
        total, by_lang, rows = run_set(cat, ROOT / "data" / f"{name}.json")
        s = total.summary()
        report[name] = {"summary": s, "by_language": {k: v.summary() for k, v in by_lang.items()},
                        "calibration": total.calibration(), "errors": total.errors, "rows": rows}
        print(f"== {name} ({s['messages']} messages) ==")
        print(f"  type correct {s['type_correct']}/{s['messages']}   exact messages {s['exact_messages']}/{s['messages']}")
        print(f"  field precision {s['precision']:.3f}  recall {s['recall']:.3f}  (tp {s['fields_tp']}, fp {s['fields_fp']}, fn {s['fields_fn']})")
        print(f"  flagged for review {s['flagged_for_review']}/{s['messages']}   wrong but not flagged {s['wrong_but_not_flagged']}")
        for lang, sc in by_lang.items():
            ls = sc.summary()
            print(f"  [{lang}] exact {ls['exact_messages']}/{ls['messages']}  precision {ls['precision']:.3f}  recall {ls['recall']:.3f}")
        print("  confidence vs accuracy: " + "  ".join(f"{c['confidence']} n={c['n']} acc={c['accuracy']}" for c in total.calibration()))
        for e in total.errors:
            print("   -", e)
        print()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "extract_results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
