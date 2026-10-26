"""Scoring for extraction experiments (DESC is verbatim and excluded)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .extract import Extraction

IGNORED = {"DESC"}


@dataclass
class Score:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    messages: int = 0
    exact: int = 0
    type_ok: int = 0
    flagged: int = 0
    exact_and_flagged: int = 0
    wrong_and_unflagged: int = 0
    points: list[tuple[float, bool]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    def add(self, label: str, ext: Extraction, expect_type: str, expect: dict[str, str], threshold: float = 0.8) -> None:
        self.messages += 1
        self.type_ok += ext.type == expect_type
        got = {k: v for k, v in ext.fields.items() if k not in IGNORED}
        all_right = ext.type == expect_type
        for k, exp in expect.items():
            if k in got and got[k].value == exp:
                self.tp += 1
            else:
                self.fn += 1
                all_right = False
                self.errors.append(f"{label} {k}: expected {exp!r}, got {got[k].value!r}" if k in got
                                   else f"{label} {k}: expected {exp!r}, not extracted")
        for k, e in got.items():
            correct = k in expect and e.value == expect[k]
            self.points.append((e.confidence, correct))
            if not correct:
                self.fp += 1
                all_right = False
                if k not in expect:
                    self.errors.append(f"{label} {k}: extracted {e.value!r} but not expected")
        flagged = bool(ext.review_reasons(threshold))
        self.exact += all_right
        self.flagged += flagged
        self.exact_and_flagged += all_right and flagged
        self.wrong_and_unflagged += (not all_right) and not flagged

    def calibration(self, edges=(0.0, 0.5, 0.8, 0.9, 1.01)) -> list[dict]:
        rows = []
        for lo, hi in zip(edges, edges[1:]):
            pts = [ok for c, ok in self.points if lo <= c < hi]
            rows.append({"confidence": f"{lo:.2f}-{min(hi, 1.0):.2f}", "n": len(pts),
                         "accuracy": round(sum(pts) / len(pts), 3) if pts else None})
        return rows

    def summary(self) -> dict:
        return {"messages": self.messages, "type_correct": self.type_ok, "exact_messages": self.exact,
                "precision": round(self.precision, 3), "recall": round(self.recall, 3),
                "flagged_for_review": self.flagged, "wrong_but_not_flagged": self.wrong_and_unflagged,
                "fields_tp": self.tp, "fields_fp": self.fp, "fields_fn": self.fn}
