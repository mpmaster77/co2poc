import json
import random
import string
from pathlib import Path

import pytest

from msgengine.catalogue import Catalogue, CatalogueError
from msgengine.codec_char import parse
from msgengine.evaluate import Score
from msgengine.extract import classify, extract, fold
from tests.conftest import CATALOGUE_PATH

DATA = Path(__file__).resolve().parent.parent / "data"


def _vals(cat, text, **kw):
    ext = extract(cat, text, **kw)
    return ext, {k: v.value for k, v in ext.fields.items()}


def test_fold_keeps_length_and_strips_accents():
    for s in ["Évacuation sanitaire", "nœuds", "Aéronef à 3000 pi", "ÀÉÎÔÛ çñ", "plain"]:
        assert len(fold(s)) == len(s)
    assert fold("Évacuation") == "evacuation"


def test_structured_values_extracted_with_spans(cat):
    text = "Observer B22 target grid 18S UJ 23387 06520 DTG 241830ZOCT26 infantry fire for effect"
    ext, v = _vals(cat, text)
    assert ext.type == "CFF"
    assert v["TGTGRID"] == "18SUJ2338706520" and v["DTG"] == "241830ZOCT26" and v["OBS"] == "B22"
    s, e = ext.fields["DTG"].span
    assert text[s:e] == "241830ZOCT26"


def test_negative_coordinate_sign_survives_cue(cat):
    _, v = _vals(cat, "Contact reporting HMCS-A 241830ZOCT26 hostile surface lat 44.5 lon -62.1")
    assert v["LAT"] == "44.5000" and v["LON"] == "-62.1000"


def test_hemisphere_letters_and_french_commas(cat):
    _, v = _vals(cat, "Ici HMCS-A, navire ennemi à 44,8000N 62,3000W, 241830ZOCT26")
    assert v["LAT"] == "44.8000" and v["LON"] == "-62.3000"
    _, v = _vals(cat, "Contact HMCS-A 241830ZOCT26 vessel unknown 10.5S 20.25E")
    assert v["LAT"] == "-10.5000" and v["LON"] == "20.2500"


def test_coordinate_pair_fallback_is_lower_confidence(cat):
    ext, v = _vals(cat, "Contact HMCS-B visual merchant ship 46.2000,-61.5000 241830ZOCT26")
    assert (v["LAT"], v["LON"]) == ("46.2000", "-61.5000")
    assert ext.fields["LAT"].confidence < 0.8


def test_out_of_range_values_are_rejected_not_forced(cat):
    ext, v = _vals(cat, "Contact HMCS-A 241830ZOCT26 vessel lat 95.0 lon 10.0 course 400 degrees")
    assert "LAT" not in v and "COURSE" not in v
    assert any("rejected" in i for i in ext.issues)


def test_missing_required_reported_and_flagged(cat):
    ext, v = _vals(cat, "9-line request EVAC12 pickup 18SUJ2338706520 123.450 mhz urgent surgical area secure")
    assert ext.type == "MEDEVAC" and "DTG" in ext.missing_required
    assert any("DTG" in r for r in ext.review_reasons())


def test_ambiguous_enum_is_flagged_with_alternatives(cat):
    ext, _ = _vals(cat, "Medevac request MV-001: 3 ambulatory, 0 urgent. Routine.")
    prec = ext.fields["PRECEDENCE"]
    assert prec.confidence < 0.5 and prec.alternatives
    assert any("PRECEDENCE" in r for r in ext.review_reasons())


def test_cue_resolves_enum_ambiguity(cat):
    ext, v = _vals(cat, "Request MV-081 pickup 18TUL5000060000, 2 routine, 0 urgent, precedence routine")
    assert v["PRECEDENCE"] == "ROUTINE"


def test_longest_phrase_wins_over_contained_phrase(cat):
    _, v = _vals(cat, "Medevac request MV-002 urgent surgical")
    assert v["PRECEDENCE"] == "URGENT_SURG"
    _, v = _vals(cat, "Medevac request MV-002 no enemy")
    assert v["SECURITY"] == "NO_ENEMY"


def test_weak_keywords_fall_back_to_field_coverage(cat):
    text = "Unit B-SQN reporting batteries, 60 left, need 150, status amber, grid 18SUJ0000100001, 090900ZMAY26."
    assert classify(cat, text)[0] != "LOGSTAT"
    ext = extract(cat, text)
    assert ext.type == "LOGSTAT" and ext.type_confidence < 0.8


def test_explicit_type_overrides_classification(cat):
    ext = extract(cat, "Observer B22 target 18SUJ2338706520 241830ZOCT26 troops ffe", msg_type="CFF")
    assert ext.type == "CFF" and ext.type_confidence == 1.0


def test_unrecognised_text_gives_no_type(cat):
    ext = extract(cat, "The quick brown fox jumps over the lazy dog")
    assert ext.type is None and ext.review_reasons() == ["message type not recognised"]


def test_reference_date_completes_short_time(cat):
    _, v = _vals(cat, "Logstat unit 1PL class III diesel 40 on hand status red 1830Z", ref_dtg="241200ZOCT26")
    assert v["DTG"] == "241830ZOCT26"
    _, v = _vals(cat, "Logstat unit 1PL class III diesel 40 on hand status red 1830Z")
    assert "DTG" not in v


def test_description_is_verbatim_and_truncated(cat):
    text = "Observer B22 target 18SUJ2338706520 241830ZOCT26 troops ffe " + "x" * 200
    ext = extract(cat, text)
    assert ext.fields["DESC"].source == "verbatim" and len(ext.fields["DESC"].value) <= 120
    assert "DESC" not in extract(cat, text, include_description=False).fields


def test_every_extracted_value_is_catalogue_valid(cat):
    for name in ("handwritten", "holdout"):
        for case in json.loads((DATA / f"{name}.json").read_text(encoding="utf-8")):
            ext = extract(cat, case["text"])
            mdef = cat.message(ext.type)
            for k, e in ext.fields.items():
                assert mdef.field(k).normalize(e.value) == e.value
                assert 0.0 <= e.confidence <= 1.0


def test_draft_text_parses_back_when_complete(cat):
    for case in json.loads((DATA / "handwritten.json").read_text(encoding="utf-8")):
        ext = extract(cat, case["text"])
        if not ext.missing_required:
            r = parse(cat, ext.draft_text(cat))
            assert r.ok and r.message == ext.message, case["id"]


def test_never_crashes_on_garbage(cat):
    rng = random.Random(1)
    alphabet = string.printable + "éàçœ°"
    for _ in range(500):
        extract(cat, "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 200))))
    for junk in ["", " ", "///", "18S", "lat lon", "0" * 500, "observer " * 100]:
        extract(cat, junk)


def test_regression_on_development_and_holdout_sets(cat):
    floors = {"handwritten": (24, 0.95), "holdout": (9, 0.95)}
    for name, (min_exact, min_precision) in floors.items():
        score = Score()
        for case in json.loads((DATA / f"{name}.json").read_text(encoding="utf-8")):
            score.add(case["id"], extract(cat, case["text"]), case["type"], case["expect"])
        assert score.exact >= min_exact - (2 if name == "handwritten" else 0), (name, score.errors)
        assert score.precision >= min_precision, (name, score.errors)
        assert score.wrong_and_unflagged == 0, (name, score.errors)


@pytest.mark.parametrize("patch, message", [
    (lambda f: f.update(synonyms={"X": ["a"]}), "synonyms must map"),
    (lambda f: f.update(hemis="XX"), "hemis"),
    (lambda f: f.update(cues=[""]), "non-empty strings"),
])
def test_bad_extraction_metadata_rejected(patch, message):
    d = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
    enum_field = next(f for f in d["messages"][0]["fields"] if f["name"] == "TGT")
    patch(enum_field)
    with pytest.raises(CatalogueError, match=message):
        Catalogue.from_dict(d)


def test_synonyms_on_non_enum_rejected():
    d = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
    d["messages"][0]["fields"][0]["synonyms"] = {"A": ["b"]}
    with pytest.raises(CatalogueError, match="only apply to enum"):
        Catalogue.from_dict(d)
