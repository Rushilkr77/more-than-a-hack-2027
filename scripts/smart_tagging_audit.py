"""Validate the structural contract of a Metrica Smart Tagging XML export."""

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional
import xml.etree.ElementTree as ET


CODES = {
    "BUILD UP", "PROGRESSION", "CREATING CHANCES", "DEFENDING IN ATTACKING THIRD",
    "DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD", "ATTACKING TRANSITION",
    "DEFENSIVE TRANSITION", "SET PIECES", "GOALS", "LONG BALL", "BALL IN FINAL THIRD",
    "BALL IN THE BOX", "PLAYERS IN THE BOX",
}
REQUIRED_BY_CODE = {
    "SET PIECES": {"Type"},
    "BALL IN FINAL THIRD": {"Side", "Max Players in the box"},
    "BALL IN THE BOX": {"Direction of ball entry", "Max Players in the box"},
    "PLAYERS IN THE BOX": {"Max Players in the box"},
}
VALUES = {
    "Team": set(),
    "Half": {"1st Half", "2nd Half"},
    "Type": {"Throw-in", "Corner Kick", "Goal Kick", "Free Kick", "Kick Off (After Goal)", "Kick Off (Start)"},
    "Side": {"Central", "Left", "Right"},
    "Direction of ball entry": {"Horizontal", "Vertical", "Diagonal"},
    "Max Players in the box": {"1", "2", "3", "4", "5", "6", "7+"},
    "Confidence": set(),
}


@dataclass
class Issue:
    severity: str
    rule: str
    message: str
    instance_index: Optional[int] = None
    instance_id: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    code: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class AuditReport:
    instance_count: int = 0
    issues: List[Issue] = field(default_factory=list)

    @property
    def errors(self):
        return [issue for issue in self.issues if issue.severity == "error"]

    def to_dict(self):
        return {
            "instance_count": self.instance_count,
            "error_count": len(self.errors),
            "warning_count": sum(issue.severity == "warning" for issue in self.issues),
            "issues": [asdict(issue) for issue in self.issues],
        }


def _text(element, name):
    child = element.find(name)
    return None if child is None or child.text is None else child.text.strip()


def _issue(report, severity, rule, message, index=None, instance_id=None, start=None, end=None, code=None):
    report.issues.append(Issue(severity, rule, message, index, instance_id, start, end, code))


def _audit_instance(report, instance, index):
    instance_id, start, end, code = (_text(instance, name) for name in ("ID", "start", "end", "code"))
    context = dict(index=index, instance_id=instance_id, start=start, end=end, code=code)
    if not instance_id:
        _issue(report, "warning", "missing_id", "instance has no ID", **context)
    if not code:
        _issue(report, "error", "missing_code", "instance has no code", **context)
    elif code not in CODES:
        _issue(report, "warning", "unknown_code", "code {!r} is not in the observed Smart Tagging taxonomy".format(code), **context)

    times = {}
    for name, value in (("start", start), ("end", end)):
        try:
            times[name] = float(value)
            if not math.isfinite(times[name]):
                raise ValueError
        except (TypeError, ValueError):
            _issue(report, "error", "invalid_" + name, "{} must be a finite number of seconds".format(name), **context)
    if len(times) == 2 and times["end"] <= times["start"]:
        _issue(report, "error", "non_positive_duration", "end must be later than start", **context)

    labels = {}
    for label in instance.findall("label"):
        group, text = _text(label, "group"), _text(label, "text")
        if not group:
            _issue(report, "error", "missing_label_group", "label has no group", **context)
            continue
        if not text:
            _issue(report, "error", "missing_label_text", "label group {!r} has no text".format(group), **context)
            continue
        labels.setdefault(group, []).append(text)
        if group not in VALUES:
            _issue(report, "warning", "unknown_label_group", "label group {!r} is not in the observed schema".format(group), **context)
        elif group == "Confidence":
            try:
                confidence = float(text)
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError
            except ValueError:
                _issue(report, "error", "invalid_confidence", "Confidence must be a finite decimal in [0, 1]", **context)
        elif group != "Team" and text not in VALUES[group]:
            severity = "error" if group == "Half" else "warning"
            rule = "invalid_half" if group == "Half" else "unknown_label_value"
            _issue(report, severity, rule, "{!r} is not a known value for {!r}".format(text, group), **context)

    for group, values in labels.items():
        if len(values) > 1:
            _issue(report, "warning", "duplicate_label_group", "label group {!r} appears {} times; the existing loader keeps only the last value".format(group, len(values)), **context)
    for group in ("Team", "Half"):
        if group not in labels:
            _issue(report, "error", "missing_" + group.lower(), "instance has no {!r} label".format(group), **context)
    if "N/A" in labels.get("Team", []):
        _issue(report, "warning", "team_not_available", "Team is N/A, so this interval cannot directly supervise a team label", **context)
    for group in REQUIRED_BY_CODE.get(code, set()):
        if group not in labels:
            _issue(report, "warning", "missing_code_label", "{} requires a {!r} label in the observed schema".format(code, group), **context)


def audit_file(path):
    """Return an :class:`AuditReport` without mutating the supplied XML."""
    report = AuditReport()
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        line, column = exc.position
        report.issues.append(Issue("error", "xml_parse_error", str(exc), line=line, column=column))
        return report

    if root.tag != "file":
        _issue(report, "warning", "unexpected_root", "expected <file>, found <{}>".format(root.tag))
    instances = root.findall("./ALL_INSTANCES/instance")
    if root.find("ALL_INSTANCES") is None:
        _issue(report, "error", "missing_instances_container", "missing <ALL_INSTANCES> container")
        instances = root.findall(".//instance")
    report.instance_count = len(instances)
    seen_ids = Counter()
    for index, instance in enumerate(instances, start=1):
        _audit_instance(report, instance, index)
        instance_id = _text(instance, "ID")
        if instance_id:
            seen_ids[instance_id] += 1
    for instance_id, count in seen_ids.items():
        if count > 1:
            _issue(report, "error", "duplicate_id", "ID {!r} occurs {} times".format(instance_id, count), instance_id=instance_id)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml_path", type=Path)
    parser.add_argument("--report", type=Path, help="write the complete JSON report to this path")
    parser.add_argument("--fail-on-error", action="store_true", help="exit 1 when structural errors are found")
    args = parser.parse_args()
    report = audit_file(args.xml_path)
    payload = report.to_dict()
    print(json.dumps(payload, indent=2))
    if args.report:
        args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 1 if args.fail_on_error and report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
