import xml.etree.ElementTree as ET
import subprocess
import sys
from pathlib import Path

from src.data.idsse_smart_tagging import generate, write_xml
from scripts.smart_tagging_audit import audit_file


METADATA = """<PutDataRequest><MatchInformation>
  <General KickoffTime="2022-10-15T11:00:00.000+00:00" />
  <Teams>
    <Team TeamId="HOME" TeamName="Home FC" Role="home" />
    <Team TeamId="AWAY" TeamName="Away FC" Role="guest" />
  </Teams>
</MatchInformation></PutDataRequest>"""

EVENTS = """<PutDataRequest>
  <Event EventId="2" EventTime="2022-10-15T11:00:20.000+00:00">
    <ThrowIn Team="AWAY" DecisionTimestamp="2022-10-15T11:00:15.000+00:00" />
  </Event>
  <Event EventId="1" EventTime="2022-10-15T11:00:00.000+00:00">
    <KickOff GameSection="firstHalf"><Play Team="HOME" /></KickOff>
  </Event>
  <Event EventId="3" EventTime="2022-10-15T11:00:40.000+00:00"><Foul TeamFouler="HOME" /></Event>
  <Event EventId="4" EventTime="2022-10-15T12:00:00.000+00:00">
    <KickOff GameSection="secondHalf"><Play Team="HOME" /></KickOff>
  </Event>
  <Event EventId="5" EventTime="2022-10-15T12:00:10.000+00:00">
    <ShotAtGoal Team="AWAY"><SuccessfulShot /></ShotAtGoal>
  </Event>
</PutDataRequest>"""


def write_inputs(tmp_path):
    metadata = tmp_path / "metadata.xml"
    events = tmp_path / "events.xml"
    metadata.write_text(METADATA, encoding="utf-8")
    events.write_text(EVENTS, encoding="utf-8")
    return events, metadata


def test_generate_sorts_and_maps_only_direct_events(tmp_path):
    events, metadata = write_inputs(tmp_path)

    records = generate(events, metadata)

    assert [(record.code, record.labels.get("Type")) for record in records] == [
        ("SET PIECES", "Kick Off (Start)"),
        ("SET PIECES", "Throw-in"),
        ("SET PIECES", "Kick Off (Start)"),
        ("GOALS", None),
    ]
    assert [(record.id, record.start_s, record.end_s) for record in records] == [
        ("1", 0.0, 5.0), ("2", 15.0, 25.0), ("3", 3600.0, 3605.0), ("4", 3610.0, 3615.0),
    ]
    assert [record.labels["Half"] for record in records] == ["1st Half", "1st Half", "2nd Half", "2nd Half"]
    assert [record.labels["Team"] for record in records] == ["Home FC", "Away FC", "Home FC", "Away FC"]
    assert [record.labels["Confidence"] for record in records] == ["0.90", "1.00", "0.90", "0.90"]


def test_write_xml_matches_the_smart_tagging_container_contract(tmp_path):
    events, metadata = write_inputs(tmp_path)
    output = tmp_path / "smart_tagging.xml"

    write_xml(generate(events, metadata), output)

    root = ET.parse(output).getroot()
    assert root.tag == "file"
    assert len(root.findall("./ALL_INSTANCES/instance")) == 4
    assert len(root.findall("./ROWS/row")) == 14
    first = root.find("./ALL_INSTANCES/instance")
    assert first.findtext("ID") == "1"
    assert {label.findtext("group"): label.findtext("text") for label in first.findall("label")}["Confidence"] == "0.90"


def test_generator_cli_writes_an_auditable_file_from_local_inputs(tmp_path):
    events, metadata = write_inputs(tmp_path)
    output = tmp_path / "generated.xml"
    root = Path(__file__).parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/generate_idsse_smart_tagging.py", "--events", str(events),
         "--metadata", str(metadata), "--output", str(output)],
        cwd=root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert audit_file(output).errors == []
