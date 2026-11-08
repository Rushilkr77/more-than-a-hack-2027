import xml.etree.ElementTree as ET

from scripts.smart_tagging_audit import audit_file
from src.data.idsse_smart_tagging import Annotation, write_xml


def test_write_xml_matches_the_smart_tagging_container_contract(tmp_path):
    output = tmp_path / "smart_tagging.xml"
    records = [
        Annotation("1", 10.0, 15.0, "SET PIECES", {"Team": "N/A", "Type": "Throw-in", "Half": "1st Half"}),
        Annotation("2", 20.0, 31.5, "BUILD UP", {"Team": "Home", "Half": "1st Half"}),
    ]
    write_xml(records, output)

    root = ET.parse(output).getroot()
    assert root.tag == "file"
    assert len(root.findall("./ALL_INSTANCES/instance")) == 2
    assert len(root.findall("./ROWS/row")) == 14
    first = root.find("./ALL_INSTANCES/instance")
    assert first.findtext("ID") == "1" and first.findtext("code") == "SET PIECES"
    assert {l.findtext("group"): l.findtext("text") for l in first.findall("label")}["Type"] == "Throw-in"
    assert not audit_file(output).errors
