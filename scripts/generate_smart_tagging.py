"""Write our generated Smart Tagging XML (src/features/tagger.py) for every Sportec bundle and for BAR-ATH, then audit it.

    python scripts/generate_smart_tagging.py

    data/sportec_bundles/<M>/tactical/<M>_tactical_smart_tagging.xml   next to the bundle's ATD, like BAR-ATH's folder
    data/generated/BAR-ATH_generated_smart_tagging.xml                  Metrica's own file is never touched

The model does not read these files: build_table computes the same tags in-process (`gtag_*` columns). The XML is for
inspection and format parity, and it goes through the same structural audit as Metrica's export.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402
from scripts.smart_tagging_audit import audit_file  # noqa: E402
from src.data.idsse_smart_tagging import write_xml  # noqa: E402
from src.features import frame_table as ft  # noqa: E402
from src.features import tagger  # noqa: E402

BUNDLES = os.path.join(ROOT, "data", "sportec_bundles")


def write(tags, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_xml(tags, path)
    report = audit_file(path)
    if report.errors:
        raise ValueError("{}: {} audit errors".format(path, len(report.errors)))
    return "{}  {} tags, {} audit warnings".format(os.path.relpath(path, ROOT), len(tags), len(report.issues))


def main():
    for m in sorted(os.listdir(BUNDLES)):
        bundle = os.path.join(BUNDLES, m)
        args = ft.bundle_args(os.path.join(bundle, "tactical", m + "_tactical_atd.xml"))
        table, slots = ft.bundle_table(bundle)
        tags, _ = tagger.generate_tags(table, slots, args["pitch"], args["fps"])
        print(write(tags, os.path.join(bundle, "tactical", m + "_tactical_smart_tagging.xml")))
    tags, _ = tagger.generate_tags(ft.load(), ft.load_slots(), GAMES[GAME]["pitch"], GAMES[GAME]["fps"])
    print(write(tags, os.path.join(ROOT, "data", "generated", "BAR-ATH_generated_smart_tagging.xml")))


if __name__ == "__main__":
    main()
