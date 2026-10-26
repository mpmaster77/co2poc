import random

import pytest
from lxml import etree

from msgengine.codec_char import parse, render
from msgengine.codec_xml import XmlError, build_xsd, from_xml, to_xml, validate_xml
from msgengine.generator import make_message
from msgengine.model import Message

CFF_TEXT = "CFF/OBS=B22/DTG=241830ZOCT26/TGTGRID=18SUJ2338706520/TGT=INF/MISSION=FFE//"


def test_generated_xsd_is_a_valid_schema(cat):
    etree.XMLSchema(etree.fromstring(build_xsd(cat)))


def test_xml_round_trip_many(cat):
    rng = random.Random(21)
    for _ in range(500):
        msg = make_message(cat, rng)
        xml = to_xml(cat, msg)
        assert validate_xml(cat, xml) == []
        assert from_xml(cat, xml) == msg


def test_char_xml_char_is_lossless(cat):
    rng = random.Random(22)
    for _ in range(300):
        msg = make_message(cat, rng)
        text = render(cat, msg)
        back = render(cat, from_xml(cat, to_xml(cat, parse(cat, text).message)))
        assert back == text


def test_xml_contains_namespace_and_values(cat):
    xml = to_xml(cat, parse(cat, CFF_TEXT).message).decode()
    assert xml.startswith(f'<CFF xmlns="{cat.namespace}">')
    assert "<OBS>B22</OBS>" in xml


@pytest.mark.parametrize("old, new", [
    ("<TGT>INF</TGT>", "<TGT>BOGUS</TGT>"),
    ("<DTG>241830ZOCT26</DTG>", "<DTG>241830ZXXX26</DTG>"),
    ("<TGT>INF</TGT>", ""),
    ("<OBS>B22</OBS>", "<OBS>B22</OBS><OBS>B23</OBS>"),
    ("</CFF>", "<EXTRA>1</EXTRA></CFF>"),
])
def test_invalid_xml_rejected(cat, old, new):
    xml = to_xml(cat, parse(cat, CFF_TEXT).message).decode()
    bad = xml.replace(old, new).encode()
    assert validate_xml(cat, bad)
    with pytest.raises(XmlError):
        from_xml(cat, bad)


def test_not_well_formed_rejected(cat):
    assert validate_xml(cat, b"<CFF")
    with pytest.raises(XmlError):
        from_xml(cat, b"not xml")


def test_entity_and_dtd_attacks_rejected(cat, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET")
    xxe = (f'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY x SYSTEM "file://{secret}">]>'
           f'<CFF xmlns="{cat.namespace}"><OBS>&x;</OBS></CFF>').encode()
    assert validate_xml(cat, xxe) == ["DTD and entity declarations are not allowed"]
    with pytest.raises(XmlError):
        from_xml(cat, xxe)


def test_text_with_markup_characters_survives(cat):
    fields = parse(cat, CFF_TEXT).message.fields | {"DESC": "a < b & c > d \"q\" 'x' é"}
    msg = Message("CFF", fields)
    assert from_xml(cat, to_xml(cat, msg)) == msg
