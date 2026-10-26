import copy
import json

import pytest

from msgengine.catalogue import Catalogue, CatalogueError
from tests.conftest import CATALOGUE_PATH


def _data():
    return json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))


def test_loads_four_message_families(cat):
    assert set(cat.messages) == {"CFF", "MEDEVAC", "LOGSTAT", "CONTACT"}
    assert {m.domain for m in cat.messages.values()} == {"land", "maritime-air"}
    assert {m.service for m in cat.messages.values()} == {"fires", "medical", "logistics", "jisr"}


def test_tag_is_deterministic():
    assert Catalogue.from_dict(_data()).tag == Catalogue.from_dict(_data()).tag


@pytest.mark.parametrize("mutate, text", [
    (lambda d: d["messages"][0]["fields"].append({"name": "OBS", "type": "token"}), "duplicate field"),
    (lambda d: d["messages"][1].__setitem__("code", 1), "unique integer"),
    (lambda d: d["messages"][0]["fields"][0].__setitem__("type", "float"), "unknown type"),
    (lambda d: d["messages"][0]["fields"][0].__setitem__("typo_key", 1), "unknown field keys"),
    (lambda d: d["messages"][0]["fields"][5].__setitem__("values", []), "enum needs"),
    (lambda d: d["messages"][0]["fields"][4].update(min=10, max=1), "min > max"),
    (lambda d: d["messages"][0]["fields"][0].__setitem__("pattern", "("), "bad pattern"),
    (lambda d: d["messages"][0]["fields"][0].__setitem__("examples", ["lowercase"]), "example"),
    (lambda d: d.pop("namespace"), "missing top-level"),
])
def test_bad_catalogues_rejected(mutate, text):
    d = copy.deepcopy(_data())
    mutate(d)
    with pytest.raises(CatalogueError, match=text):
        Catalogue.from_dict(d)


@pytest.mark.parametrize("ftype, field_name, good, bad", [
    ("dtg", "DTG", "241830ZOCT26", "311830ZFEB26"),
    ("mgrs", "TGTGRID", "18SUJ2338706520", "99SUJ2338706520"),
    ("int", "QTY", "12", "0"),
    ("enum", "TGT", "INF", "inf"),
])
def test_field_validation(cat, ftype, field_name, good, bad):
    f = cat.message("CFF").field(field_name)
    assert f.type == ftype
    assert f.normalize(good) == good
    with pytest.raises(ValueError):
        f.normalize(bad)


def test_decimal_canonical_form_and_scale(cat):
    f = cat.message("CONTACT").field("LAT")
    assert f.normalize("45.5") == "45.5000"
    assert f.normalize("-0.0000") == "0.0000"
    with pytest.raises(ValueError):
        f.normalize("45.12345")
    with pytest.raises(ValueError):
        f.normalize("91")
