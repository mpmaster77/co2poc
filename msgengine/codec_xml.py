"""XML representation, XSD generated from the catalogue, and schema validation."""
from __future__ import annotations

from xml.sax.saxutils import escape as _xml_escape, quoteattr

from lxml import etree

from .catalogue import Catalogue, Field
from .model import Message


class XmlError(ValueError):
    pass


def _simple_type(f: Field) -> str:
    if f.type == "token":
        pattern = f.pattern or r"[A-Za-z0-9_.\-]+"
        return (f'<xs:restriction base="xs:string"><xs:maxLength value="{f.max_len or 32}"/>'
                f"<xs:pattern value={quoteattr(pattern)}/></xs:restriction>")
    if f.type == "text":
        return (f'<xs:restriction base="xs:string"><xs:minLength value="1"/>'
                f'<xs:maxLength value="{f.max_len or 120}"/></xs:restriction>')
    if f.type == "enum":
        enums = "".join(f'<xs:enumeration value="{v}"/>' for v in f.values)
        return f'<xs:restriction base="xs:string">{enums}</xs:restriction>'
    if f.type == "int":
        bounds = ""
        if f.minimum is not None:
            bounds += f'<xs:minInclusive value="{f.minimum}"/>'
        if f.maximum is not None:
            bounds += f'<xs:maxInclusive value="{f.maximum}"/>'
        return f'<xs:restriction base="xs:integer">{bounds}</xs:restriction>'
    if f.type == "decimal":
        bounds = f'<xs:fractionDigits value="{f.scale}"/>'
        if f.minimum is not None:
            bounds += f'<xs:minInclusive value="{f.minimum}"/>'
        if f.maximum is not None:
            bounds += f'<xs:maxInclusive value="{f.maximum}"/>'
        return f'<xs:restriction base="xs:decimal">{bounds}</xs:restriction>'
    from .catalogue import DTG_PATTERN, MGRS_PATTERN

    pattern = DTG_PATTERN if f.type == "dtg" else MGRS_PATTERN
    return f'<xs:restriction base="xs:string"><xs:pattern value={quoteattr(pattern)}/></xs:restriction>'


def build_xsd(cat: Catalogue) -> bytes:
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           f'<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace={quoteattr(cat.namespace)} '
           f'xmlns={quoteattr(cat.namespace)} elementFormDefault="qualified">']
    for mdef in cat.messages.values():
        out.append(f'<xs:element name="{mdef.id}"><xs:annotation><xs:documentation>'
                   f"{_xml_escape(mdef.title)}</xs:documentation></xs:annotation><xs:complexType><xs:sequence>")
        for f in mdef.fields:
            min_occurs = "1" if f.required else "0"
            out.append(f'<xs:element name="{f.name}" minOccurs="{min_occurs}"><xs:simpleType>'
                       f"{_simple_type(f)}</xs:simpleType></xs:element>")
        out.append("</xs:sequence></xs:complexType></xs:element>")
    out.append("</xs:schema>")
    return "".join(out).encode("utf-8")


def _schema(cat: Catalogue) -> etree.XMLSchema:
    if "xsd" not in cat._cache:
        cat._cache["xsd"] = etree.XMLSchema(etree.fromstring(build_xsd(cat)))
    return cat._cache["xsd"]


def to_xml(cat: Catalogue, msg: Message) -> bytes:
    mdef = cat.message(msg.type)
    root = etree.Element(f"{{{cat.namespace}}}{msg.type}", nsmap={None: cat.namespace})
    for f in mdef.fields:
        if f.name in msg.fields:
            etree.SubElement(root, f"{{{cat.namespace}}}{f.name}").text = msg.fields[f.name]
    return etree.tostring(root, encoding="utf-8", xml_declaration=False)


def _parser() -> etree.XMLParser:
    return etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False)


def validate_xml(cat: Catalogue, xml: bytes) -> list[str]:
    """Return XSD validation errors (empty list means valid)."""
    if b"<!DOCTYPE" in xml or b"<!ENTITY" in xml:
        return ["DTD and entity declarations are not allowed"]
    try:
        doc = etree.fromstring(xml, _parser())
    except etree.XMLSyntaxError as exc:
        return [f"not well-formed: {exc}"]
    schema = _schema(cat)
    if schema.validate(doc):
        return []
    return [e.message for e in schema.error_log]


def from_xml(cat: Catalogue, xml: bytes) -> Message:
    errors = validate_xml(cat, xml)
    if errors:
        raise XmlError("; ".join(errors))
    root = etree.fromstring(xml, _parser())
    msg_type = etree.QName(root).localname
    mdef = cat.message(msg_type)
    fields: dict[str, str] = {}
    for child in root:
        name = etree.QName(child).localname
        fdef = mdef.field(name)
        if fdef is None or name in fields:
            raise XmlError(f"unexpected or repeated element {name}")
        try:
            fields[name] = fdef.normalize(child.text or "")
        except ValueError as exc:
            raise XmlError(f"{name}: {exc}") from exc
    return Message(msg_type, fields)
