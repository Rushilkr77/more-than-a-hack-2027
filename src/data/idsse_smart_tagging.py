"""Smart Tagging XML container: the `Annotation` record and a writer in Metrica's SportsCode XML shape.

Used by src/features/tagger.py. The events-based generator that lived here copied exact human set pieces from DFL
events (risk A5 in docs/data-layer/SPORTEC.md) and was removed."""

from dataclasses import dataclass
from typing import Dict, List
import xml.etree.ElementTree as ET


ROWS = (
    ("BUILD UP", "35000", "45000", "60000"),
    ("PROGRESSION", "30000", "40000", "55000"),
    ("CREATING CHANCES", "24000", "36000", "50000"),
    ("DEFENDING IN ATTACKING THIRD", "62000", "32000", "37000"),
    ("DEFENDING IN MIDDLE THIRD", "58000", "28000", "33000"),
    ("DEFENDING IN DEFENSIVE THIRD", "56000", "26000", "30000"),
    ("ATTACKING TRANSITION", "52000", "47000", "22000"),
    ("DEFENSIVE TRANSITION", "48000", "43000", "18000"),
    ("SET PIECES", "22000", "47000", "32000"),
    ("GOALS", "18000", "43000", "28000"),
    ("LONG BALL", "44000", "39000", "54000"),
    ("BALL IN FINAL THIRD", "42000", "37000", "52000"),
    ("BALL IN THE BOX", "38000", "34000", "48000"),
    ("PLAYERS IN THE BOX", "36000", "32000", "46000"),
)


@dataclass
class Annotation:
    id: str
    start_s: float
    end_s: float
    code: str
    labels: Dict[str, str]


def _seconds_text(value: float) -> str:
    return "{:.3f}".format(value).rstrip("0").rstrip(".") + (".0" if value == int(value) else "")


def write_xml(records: List[Annotation], output_path) -> None:
    """Write annotations in the SportsCode XML shape used by Smart Tagging."""
    root = ET.Element("file")
    instances = ET.SubElement(root, "ALL_INSTANCES")
    for record in records:
        instance = ET.SubElement(instances, "instance")
        for name, value in (("ID", record.id), ("start", _seconds_text(record.start_s)),
                            ("end", _seconds_text(record.end_s)), ("code", record.code)):
            ET.SubElement(instance, name).text = value
        for group in ("Team", "Type", "Half", "Side", "Direction of ball entry", "Max Players in the box", "Confidence"):
            if group in record.labels:
                label = ET.SubElement(instance, "label")
                ET.SubElement(label, "group").text = group
                ET.SubElement(label, "text").text = record.labels[group]
    rows = ET.SubElement(root, "ROWS")
    for code, red, green, blue in ROWS:
        row = ET.SubElement(rows, "row")
        for name, value in (("code", code), ("R", red), ("G", green), ("B", blue)):
            ET.SubElement(row, name).text = value
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
