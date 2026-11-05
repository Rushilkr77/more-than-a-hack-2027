"""Generate direct-event Smart Tagging XML from IDSSE event feeds."""

import argparse
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.smart_tagging_audit import audit_file  # noqa: E402
from src.data.idsse_smart_tagging import generate, write_xml


COMPETITIONS = {
    "J03WMX": "DFL-COM-000001",
    "J03WN1": "DFL-COM-000001",
    "J03WOH": "DFL-COM-000002",
    "J03WOY": "DFL-COM-000002",
    "J03WPY": "DFL-COM-000002",
    "J03WQQ": "DFL-COM-000002",
    "J03WR9": "DFL-COM-000002",
}
BASE_URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/{}"


def _filename(match_id, kind):
    competition = COMPETITIONS[match_id]
    prefix = "DFL_02_01_matchinformation" if kind == "metadata" else "DFL_03_02_events_raw"
    return "{}_{}_DFL-MAT-{}.xml".format(prefix, competition, match_id)


def _download(match_id, kind, destination):
    with urlopen(BASE_URL.format(_filename(match_id, kind)), timeout=60) as response:
        destination.write_bytes(response.read())


def _write(events, metadata, output):
    records = generate(events, metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_xml(records, output)
    report = audit_file(output)
    if report.errors:
        raise ValueError("generated XML failed audit: {}".format(report.to_dict()))
    print("{}: {} annotations".format(output, len(records)))


def _remote(match_ids, output_dir):
    with TemporaryDirectory(prefix="idsse-smart-tagging-") as directory:
        directory = Path(directory)
        for match_id in match_ids:
            events, metadata = directory / (match_id + "_events.xml"), directory / (match_id + "_metadata.xml")
            _download(match_id, "event", events)
            _download(match_id, "metadata", metadata)
            _write(events, metadata, output_dir / (match_id + "_smart_tagging.xml"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, help="local IDSSE event XML")
    parser.add_argument("--metadata", type=Path, help="local IDSSE metadata XML")
    parser.add_argument("--output", type=Path, help="output path for one local match")
    parser.add_argument("--match-id", action="append", choices=sorted(COMPETITIONS), help="download and generate one IDSSE match")
    parser.add_argument("--all", action="store_true", help="download and generate all seven IDSSE matches")
    parser.add_argument("--output-dir", type=Path, help="output directory for --match-id or --all")
    args = parser.parse_args()

    local = any((args.events, args.metadata, args.output))
    remote = args.all or args.match_id
    if local and remote:
        parser.error("choose local input or --match-id/--all, not both")
    if local:
        if not all((args.events, args.metadata, args.output)):
            parser.error("local generation needs --events, --metadata, and --output")
        _write(args.events, args.metadata, args.output)
        return 0
    if not remote or not args.output_dir:
        parser.error("use local input or --match-id/--all with --output-dir")
    _remote(sorted(COMPETITIONS) if args.all else args.match_id, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
