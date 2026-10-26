"""Free-text to structured draft: deterministic, catalogue-driven, with confidence scores and evidence spans.

No machine learning. Cues, synonyms, units and keywords come from the catalogue, so a new format needs data,
not code. The result is a draft for an operator to confirm; it is never released automatically.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from .catalogue import DTG_PATTERN, MONTHS, Catalogue, Field, MessageDef
from .codec_char import render
from .model import Message

NUM = r"-?\d+(?:[.,]\d+)?"
_STOP = {"for", "the", "is", "at", "from", "this", "a", "an", "to", "and", "with", "our", "your", "en", "de", "la",
         "le", "un", "une", "of", "on", "in", "by", "no", "not", "all", "any", "be", "are", "was", "it", "as"}
_FILLER = r"(?:(?:is|est|of|at|about|approx|environ|are)\s+)?"
_DTG_RE = re.compile(r"(?<![a-z0-9])" + DTG_PATTERN.lower() + r"(?![a-z0-9])")
_DTG_SHORT = re.compile(r"(?<![a-z0-9])([01][0-9]|2[0-3])([0-5][0-9])\s?z(?![a-z0-9])")
_MGRS_RE = re.compile(r"(?<![a-z0-9])(\d{1,2})\s?([c-hj-np-x])\s?([a-hj-np-z])([a-hj-np-v])\s?(\d{2,5}\s\d{2,5}|\d{2,10})(?![a-z0-9])")
_PAIR_RE = re.compile(r"(?<![0-9.])(-?\d{1,2}\.\d+)\s*[,;/]\s*(-?\d{1,3}\.\d+)(?![0-9])")


def fold(text: str) -> str:
    """Lowercase, strip accents, keep the length so spans map back to the original text."""
    out = []
    for ch in text:
        d = "".join(c for c in unicodedata.normalize("NFD", ch) if unicodedata.category(c) != "Mn")
        out.append(d if len(d) == 1 else ch)
    return "".join(out).lower()


def _phrase_re(phrase: str) -> str:
    parts = [re.escape(p) for p in fold(phrase).split()]
    return r"[\s-]+".join(parts)


def _alt(phrases) -> str:
    return "|".join(_phrase_re(p) for p in sorted(set(phrases), key=len, reverse=True))


@dataclass
class Extracted:
    value: str
    confidence: float
    span: tuple[int, int] | None
    source: str
    alternatives: tuple[str, ...] = ()


@dataclass
class Extraction:
    type: str | None
    type_confidence: float
    fields: dict[str, Extracted] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def message(self) -> Message | None:
        return Message(self.type, {k: v.value for k, v in self.fields.items()}) if self.type else None

    def review_reasons(self, threshold: float = 0.8) -> list[str]:
        reasons = []
        if self.type is None:
            return ["message type not recognised"]
        if self.type_confidence < threshold:
            reasons.append(f"message type uncertain ({self.type_confidence:.2f})")
        reasons += [f"required field {n} missing" for n in self.missing_required]
        reasons += [f"{n} low confidence ({e.confidence:.2f})" for n, e in self.fields.items() if e.confidence < threshold]
        reasons += [f"{n} ambiguous: also {', '.join(e.alternatives)}" for n, e in self.fields.items() if e.alternatives]
        return reasons

    def draft_text(self, cat: Catalogue) -> str | None:
        msg = self.message
        return render(cat, msg) if msg else None


class _Ctx:
    def __init__(self, text: str, mdef: MessageDef) -> None:
        self.text, self.low, self.mdef = text, fold(text), mdef
        self.used: list[tuple[int, int]] = []
        self.out: dict[str, Extracted] = {}
        self.issues: list[str] = []

    def free(self, span: tuple[int, int]) -> bool:
        return not any(span[0] < e and s < span[1] for s, e in self.used)

    def put(self, f: Field, raw: str, conf: float, span: tuple[int, int] | None, source: str, alts=(), claim=True) -> bool:
        if f.name in self.out:
            return False
        try:
            value = f.normalize(raw)
        except ValueError as exc:
            self.issues.append(f"{f.name}: {raw!r} rejected ({exc})")
            return False
        self.out[f.name] = Extracted(value, round(conf, 2), span, source, tuple(alts))
        if span and claim:
            self.used.append(span)
        return True


def classify(cat: Catalogue, text: str) -> tuple[str | None, float]:
    low = fold(text)
    scores: dict[str, float] = {}
    for m in cat.messages.values():
        s = 0.0
        for kw in m.keywords:
            if re.search(r"(?<![a-z0-9])" + _phrase_re(kw) + r"(?![a-z0-9])", low):
                s += 1 + 0.1 * len(kw.split())
        scores[m.id] = s
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_id, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if best <= 0:
        return None, 0.0
    conf = 0.5 * ((best - second) / best) + 0.5 * min(1.0, best / 2.0)
    return best_id, round(min(conf, 0.99), 2)


def extract(cat: Catalogue, text: str, msg_type: str | None = None, ref_dtg: str | None = None,
            include_description: bool = True) -> Extraction:
    if msg_type:
        mtype, tconf = msg_type, 1.0
    else:
        mtype, tconf = classify(cat, text)
        if mtype is None or tconf < 0.6:
            mtype, tconf = _type_by_fill(cat, text, ref_dtg, mtype, tconf)
    if mtype is None:
        return Extraction(None, 0.0, issues=["no message type keywords found"])
    return _run(cat, text, mtype, tconf, ref_dtg, include_description)


def _type_by_fill(cat: Catalogue, text: str, ref_dtg: str | None, mtype: str | None, tconf: float) -> tuple[str | None, float]:
    """When keywords are weak, prefer the type whose required fields the text fills best."""
    fills = {}
    for m in cat.messages.values():
        ext = _run(cat, text, m.id, 1.0, ref_dtg, False)
        required = [f.name for f in m.fields if f.required]
        fills[m.id] = (len(required) - len(ext.missing_required)) / len(required) if required else 0.0
    ranked = sorted(fills.items(), key=lambda kv: kv[1], reverse=True)
    best_id, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if best < 0.5 or best - second < 0.15:
        return mtype, tconf
    return best_id, max(tconf, round(0.5 * best, 2))


def _run(cat: Catalogue, text: str, mtype: str, tconf: float, ref_dtg: str | None, include_description: bool) -> Extraction:
    ctx = _Ctx(text, cat.message(mtype))
    _dtg(ctx, ref_dtg)
    _mgrs(ctx)
    _hemisphere_decimals(ctx)
    _tokens(ctx)
    _numbers(ctx, ("decimal",))
    _enums(ctx)
    _numbers(ctx, ("int",))
    _coordinate_pair(ctx)
    if include_description:
        for f in ctx.mdef.fields:
            if f.type == "text":
                flat = re.sub(r"\s+", " ", text).strip()
                if len(flat) > (f.max_len or 120):
                    flat = flat[:(f.max_len or 120)].rstrip()
                if flat:
                    ctx.put(f, flat, 1.0, None, "verbatim", claim=False)
    missing = [f.name for f in ctx.mdef.fields if f.required and f.name not in ctx.out]
    ordered = {f.name: ctx.out[f.name] for f in ctx.mdef.fields if f.name in ctx.out}
    return Extraction(mtype, tconf, ordered, missing, ctx.issues)


def _dtg(ctx: _Ctx, ref_dtg: str | None) -> None:
    for f in (f for f in ctx.mdef.fields if f.type == "dtg"):
        for m in _DTG_RE.finditer(ctx.low):
            span = m.span()
            if ctx.free(span) and ctx.put(f, ctx.text[span[0]:span[1]].upper(), 0.98, span, "dtg pattern"):
                break
        if f.name not in ctx.out and ref_dtg:
            for m in _DTG_SHORT.finditer(ctx.low):
                span = m.span()
                value = f"{ref_dtg[:2]}{m.group(1)}{m.group(2)}Z{ref_dtg[7:]}"
                if ctx.free(span) and ctx.put(f, value, 0.75, span, "time with reference date"):
                    break


def _mgrs(ctx: _Ctx) -> None:
    fields = [f for f in ctx.mdef.fields if f.type == "mgrs"]
    if not fields:
        return
    cands = []
    for m in _MGRS_RE.finditer(ctx.low):
        digits = m.group(5).replace(" ", "")
        if len(digits) % 2 or (" " in m.group(5) and len(m.group(5).split()[0]) != len(m.group(5).split()[1])):
            continue
        value = f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}{digits}".upper()
        if ctx.free(m.span()):
            cands.append((m.span(), value))
    leftovers = []
    for span, value in cands:
        best, best_len = None, -1
        for f in fields:
            for cue in f.cues:
                for cm in re.finditer(r"(?<![a-z0-9])" + _phrase_re(cue) + r"(?![a-z0-9])", ctx.low[:span[0]]):
                    gap = ctx.low[cm.end():span[0]]
                    if len(gap) <= 45 and not re.search(r"[.;!?]\s", gap) and f.name not in ctx.out:
                        if cm.end() > best_len:
                            best, best_len = f, cm.end()
        if best and ctx.put(best, value, 0.9, span, "grid after cue"):
            continue
        leftovers.append((span, value))
    for span, value in leftovers:
        pool = [f for f in fields if f.name not in ctx.out]
        pool.sort(key=lambda f: not f.required)
        if pool:
            ctx.put(pool[0], value, 0.7 if pool[0].required else 0.55, span, "grid by position")
        else:
            ctx.issues.append(f"unassigned grid {value}")


def _hemisphere_decimals(ctx: _Ctx) -> None:
    for f in (f for f in ctx.mdef.fields if f.type == "decimal" and f.hemis):
        plus, minus = ("n", "s") if f.hemis == "NS" else ("e", "w")
        rx = re.compile(r"(?<![a-z0-9.,])(\d+[.,]\d+)\s*°?\s*([" + plus + minus + r"])(?![a-z0-9])")
        for m in rx.finditer(ctx.low):
            span = (m.start(1), m.end(2))
            if not ctx.free(span):
                continue
            value = m.group(1).replace(",", ".")
            if m.group(2) == minus:
                value = "-" + value
            if _put_decimal(ctx, f, value, 0.92, span, "coordinate with hemisphere"):
                break


def _coordinate_pair(ctx: _Ctx) -> None:
    lat = next((f for f in ctx.mdef.fields if f.type == "decimal" and f.hemis == "NS"), None)
    lon = next((f for f in ctx.mdef.fields if f.type == "decimal" and f.hemis == "EW"), None)
    if not lat or not lon or lat.name in ctx.out or lon.name in ctx.out:
        return
    for m in _PAIR_RE.finditer(ctx.low):
        span = m.span()
        if ctx.free(span):
            a, b = m.group(1), m.group(2)
            try:
                lat.normalize(_quantize(lat, a))
                lon.normalize(_quantize(lon, b))
            except ValueError:
                continue
            ctx.put(lat, _quantize(lat, a), 0.7, span, "coordinate pair")
            ctx.put(lon, _quantize(lon, b), 0.7, span, "coordinate pair", claim=False)
            return


def _quantize(f: Field, raw: str) -> str:
    try:
        d = Decimal(raw.replace(",", "."))
    except InvalidOperation as exc:
        raise ValueError("not a number") from exc
    return format(d.quantize(Decimal(1).scaleb(-f.scale), rounding=ROUND_HALF_UP), "f")


def _put_decimal(ctx: _Ctx, f: Field, raw: str, conf: float, span, source: str) -> bool:
    try:
        raw_q = _quantize(f, raw)
    except ValueError:
        return False
    places = len(raw.replace(",", ".").split(".")[1]) if "." in raw.replace(",", ".") else 0
    return ctx.put(f, raw_q, conf if places <= f.scale else min(conf, 0.7), span, source)


def _numbers(ctx: _Ctx, types: tuple[str, ...]) -> None:
    for f in ctx.mdef.fields:
        if f.type not in types or f.name in ctx.out or (not f.cues and not f.units):
            continue
        patterns = []
        if f.cues:
            patterns.append((re.compile(r"(?<![a-z0-9])(?:" + _alt(f.cues) + r")(?![a-z0-9])[\s:=#]*" + _FILLER
                                        + r"(" + NUM + r")(?![a-z0-9])"), 0.9, "number after cue"))
        if f.units:
            patterns.append((re.compile(r"(?<![a-z0-9.,])(" + NUM + r")\s*(?:" + _alt(f.units) + r")(?![a-z0-9])"),
                             0.85, "number before unit"))
        done = False
        for rx, conf, source in patterns:
            for m in rx.finditer(ctx.low):
                span = m.span(1)
                raw = m.group(1).replace(",", ".")
                if not ctx.free(span):
                    continue
                if f.type == "int":
                    if "." in raw or not ctx.put(f, raw, conf, span, source):
                        continue
                elif not _put_decimal(ctx, f, raw, conf, span, source):
                    continue
                done = True
                break
            if done:
                break


def _enums(ctx: _Ctx) -> None:
    for f in (f for f in ctx.mdef.fields if f.type == "enum"):
        syn = {v: list(ps) for v, ps in f.synonyms}
        cue_spans = []
        if f.cues:
            for cm in re.finditer(r"(?<![a-z0-9])(?:" + _alt(f.cues) + r")(?![a-z0-9])", ctx.low):
                cue_spans.append(cm.span())
        cands: list[tuple[int, int, str, bool]] = []
        for v in f.values:
            phrases = [(p, False) for p in syn.get(v, [])]
            if len(v) >= 3:
                phrases.append((v.lower().replace("_", " "), True))
            for p, is_default in phrases:
                for m in re.finditer(r"(?<![a-z0-9])" + _phrase_re(p) + r"(?![a-z0-9])", ctx.low):
                    near = any(0 <= m.start() - ce <= 40 and not re.search(r"[.;!?]\s", ctx.low[ce:m.start()])
                               for _, ce in cue_spans)
                    if is_default and f.cues and not near:
                        continue
                    cands.append((m.start(), m.end(), v, near))
        cands = [c for c in cands if not any(o[2] != c[2] and o[0] <= c[0] and c[1] <= o[1] and (o[1] - o[0]) > (c[1] - c[0])
                                             for o in cands)]
        if not cands:
            continue
        cands.sort()
        values = list(dict.fromkeys(c[2] for c in cands))
        near_vals = list(dict.fromkeys(c[2] for c in cands if c[3]))
        if near_vals:
            chosen, conf = near_vals[0], 0.9
        elif len(values) == 1:
            chosen, conf = values[0], 0.85
        else:
            chosen, conf = values[0], 0.45
        alts = [v for v in values if v != chosen]
        span = next((c[0], c[1]) for c in cands if c[2] == chosen)
        if near_vals and len(near_vals) > 1:
            alts, conf = [v for v in near_vals if v != chosen], 0.5
        ctx.put(f, chosen, conf, span, "synonym" if not near_vals else "value after cue", alts, claim=False)


def _tokens(ctx: _Ctx) -> None:
    for f in (f for f in ctx.mdef.fields if f.type == "token"):
        if f.cues:
            rx = re.compile(r"(?<![a-z0-9])(?:" + _alt(f.cues) + r")(?![a-z0-9])[\s:=#-]*" + _FILLER
                            + r"([a-z0-9][a-z0-9_.\-]*)")
            for m in rx.finditer(ctx.low):
                s, e = m.span(1)
                raw = ctx.text[s:e].rstrip(".-_")
                e = s + len(raw)
                if not raw or raw.lower() in _STOP or not ctx.free((s, e)):
                    continue
                if not (any(c.isdigit() for c in raw) or (raw.isupper() and len(raw) >= 2)):
                    continue
                if ctx.put(f, raw.upper(), 0.9, (s, e), "identifier after cue"):
                    break
        if f.name not in ctx.out:
            for ex in f.examples:
                m = re.search(r"(?<![a-z0-9])" + _phrase_re(ex.replace("_", " ")) + r"(?![a-z0-9])", ctx.low)
                if m and ctx.free(m.span()):
                    ctx.put(f, ex, 0.75, m.span(), "known value")
                    break
