"""Generate one IDSSE Smart Tagging XML with direct events and first-pass inferred phases."""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.smart_tagging_audit import audit_file  # noqa: E402
from src.data.idsse_phase_inference import (attack_directions_from_events, infer_team_phase_records,
                                             merge_records)  # noqa: E402
from src.data.idsse_smart_tagging import generate, write_xml  # noqa: E402
from src.tracking import SportecTrackingAdapter  # noqa: E402


def _teams(metadata_path):
    teams = {team.get("Role"): (team.get("TeamId"), team.get("TeamName"))
             for team in ET.parse(metadata_path).getroot().iter("Team")}
    if not all(teams.get(role, (None, None))[0] and teams[role][1] for role in ("home", "guest")):
        raise ValueError("{}: expected named home and guest teams".format(metadata_path))
    return teams["home"], teams["guest"]


def generate_inferred(events_path, metadata_path, positions_path):
    home, away = _teams(metadata_path)
    directions = attack_directions_from_events(events_path, home[0], away[0])
    tracking = SportecTrackingAdapter(positions_path, metadata_path, directions).load()
    return merge_records(generate(events_path, metadata_path), infer_team_phase_records(tracking, (home[1], away[1])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--positions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = generate_inferred(args.events, args.metadata, args.positions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_xml(records, args.output)
    report = audit_file(args.output)
    if report.errors:
        raise ValueError("generated XML failed audit: {}".format(report.to_dict()))
    print("{}: {} annotations ({} inferred)".format(args.output, len(records),
                                                       sum(record.labels.get("Confidence") not in {"1.00", "0.90"} for record in records)))


if __name__ == "__main__":
    main()
