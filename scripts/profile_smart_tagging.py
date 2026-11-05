"""Write a reproducible tracking profile for every reference Smart Tagging interval.

    .venv/bin/python scripts/profile_smart_tagging.py
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402
from src.data.smart_tagging_profile import profile  # noqa: E402
from src.features import frame_table as ft  # noqa: E402
from starter import load_smart_tagging  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(ROOT, "outputs", "profiling", "BAR-ATH_smart_tagging_profile.json"))
    args = parser.parse_args()
    report = profile(ft.load(), ft.load_slots(), load_smart_tagging.load(), ft.TEAM_TAG,
                     *GAMES[GAME]["pitch"], GAMES[GAME]["fps"])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("wrote {} codes / {} intervals to {}".format(
        len(report["codes"]), sum(code["summary"]["interval_count"] for code in report["codes"].values()), args.output))


if __name__ == "__main__":
    main()
