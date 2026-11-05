"""Create direct-event Smart Tagging annotations from IDSSE XML feeds."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
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
SET_PIECE_TYPES = {
    "ThrowIn": "Throw-in",
    "CornerKick": "Corner Kick",
    "GoalKick": "Goal Kick",
    "FreeKick": "Free Kick",
}


@dataclass
class Annotation:
    id: str
    start_s: float
    end_s: float
    code: str
    labels: Dict[str, str]


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _seconds(value: datetime, kickoff: datetime) -> float:
    return round((value - kickoff).total_seconds(), 3)


def _team_names(metadata_path: Path):
    root = ET.parse(metadata_path).getroot()
    general = root.find("./MatchInformation/General")
    if general is None or not general.get("KickoffTime"):
        raise ValueError("metadata has no MatchInformation/General@KickoffTime")
    teams = {
        team.get("TeamId"): team.get("TeamName")
        for team in root.findall("./MatchInformation/Teams/Team")
        if team.get("TeamId") and team.get("TeamName")
    }
    return _timestamp(general.get("KickoffTime")), teams


def _source_team(event_type: ET.Element) -> Optional[str]:
    if event_type.tag == "KickOff":
        play = event_type.find(".//Play")
        return play.get("Team") if play is not None else None
    return event_type.get("Team")


def _candidate(event: ET.Element, kickoff: datetime, teams: Dict[str, str], second_half_start: datetime):
    if not len(event) or not event.get("EventTime"):
        return None
    event_type = event[0]
    source_time = _timestamp(event.get("EventTime"))
    team = _source_team(event_type)
    if not team or team not in teams:
        return None

    code, type_label = None, None
    if event_type.tag == "KickOff":
        code = "SET PIECES"
        type_label = "Kick Off (Start)" if event_type.get("GameSection") in {"firstHalf", "secondHalf"} else "Kick Off (After Goal)"
    elif event_type.tag in SET_PIECE_TYPES:
        code, type_label = "SET PIECES", SET_PIECE_TYPES[event_type.tag]
    elif event_type.tag == "ShotAtGoal" and event_type.find("SuccessfulShot") is not None:
        code = "GOALS"
    else:
        return None

    decision = event_type.get("DecisionTimestamp")
    try:
        start_time = _timestamp(decision) if decision else source_time
    except ValueError:
        start_time = source_time
        decision = None
    start_s, event_s = _seconds(start_time, kickoff), _seconds(source_time, kickoff)
    if start_s < 0 or event_s < 0:
        return None
    labels = {"Team": teams[team], "Half": "2nd Half" if source_time >= second_half_start else "1st Half"}
    if type_label:
        labels["Type"] = type_label
    labels["Confidence"] = "1.00" if decision else "0.90"
    return start_s, event_s + 5.0, code, labels, event.get("EventId", "")


def generate(events_path, metadata_path) -> List[Annotation]:
    """Return sorted, directly observed restart and goal annotations."""
    kickoff, teams = _team_names(Path(metadata_path))
    root = ET.parse(events_path).getroot()
    events = root.findall("./Event")
    second_half = next(
        (_timestamp(event.get("EventTime")) for event in events
         if len(event) and event[0].tag == "KickOff" and event[0].get("GameSection") == "secondHalf"),
        None,
    )
    if second_half is None:
        raise ValueError("events has no second-half KickOff")
    candidates = [_candidate(event, kickoff, teams, second_half) for event in events]
    candidates = [candidate for candidate in candidates if candidate is not None]
    candidates.sort(key=lambda item: (item[0], item[1], item[4]))
    return [Annotation(str(index), start_s, end_s, code, labels)
            for index, (start_s, end_s, code, labels, _) in enumerate(candidates, start=1)]


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
        for group in ("Team", "Type", "Half", "Confidence"):
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
