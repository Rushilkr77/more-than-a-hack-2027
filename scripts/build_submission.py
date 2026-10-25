"""One command: licensed bundle -> cached frame table -> evidence -> decoder -> validated CSV.

    python scripts/build_submission.py [--config configs/baseline.json] [--data-dir DIR] [--rebuild]
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import GAMES, GAME  # noqa: E402
from starter.baseline import write_submission  # noqa: E402
from src.decoder import majority  # noqa: E402
from src.evidence import ball_filled, ball_nearest  # noqa: E402
from src.features import frame_table  # noqa: E402

EVIDENCE = {"ball_nearest": lambda table, cfg: ball_nearest.score(table),
            # optional, marked broadcast fills; needs the broadcast ATD files (see src/evidence/ball_filled.py)
            "ball_filled": lambda table, cfg: ball_filled.score(table, tiers=tuple(cfg.get("tiers", (1, 2))))}
DECODERS = {"majority": lambda ev, cfg: majority.decode(ev, cfg["window"], cfg.get("min_anchor", 0.0), cfg.get("tie", "home"))}


def build_submission(config, out_dir, table=None, n_rows=None):
    """Returns (csv path, sha256). `table` defaults to the cached frame table."""
    table = frame_table.load() if table is None else table
    n_rows = n_rows or GAMES[GAME]["n_video_frames"]
    if config.get("uses_broadcast"):
        print("WARNING: {} uses broadcast-filled ball evidence (optional, not validated to help spellF1); "
              "output is not the default baseline".format(config["run_id"]))
    evidence = EVIDENCE[config["evidence"]](table, config)
    labels = DECODERS[config["decoder"]["name"]](evidence, config["decoder"])

    path = write_submission(os.path.join(out_dir, config["run_id"] + ".csv"), labels)
    reason = pm.validate_submission(pm.read_label_csv(path), n_rows)
    if reason:
        raise SystemExit("INVALID submission {}: {}".format(path, reason))
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    write_manifest(path, config, table, evidence, labels, sha)
    return path, sha


def write_manifest(csv_path, config, table, evidence, labels, sha):
    """<run_id>.manifest.json next to the CSV: what produced it and how much of it rests on weak evidence."""
    g = table["graded"].to_numpy()
    voting = evidence["score_H"].notna().to_numpy()
    source = evidence["ball_source"].to_numpy() if "ball_source" in evidence else np.where(voting, 0, -1)
    share = lambda m: round(float(m[g].mean()), 4)  # noqa: E731
    meta_path = frame_table._paths()[2]
    manifest = {
        "run_id": config["run_id"], "config": config, "output_sha256": sha,
        "code_sha256": frame_table.code_sha256(("scripts/build_submission.py", "src/decoder/majority.py", "src/evidence/ball_nearest.py",
                                                "src/evidence/ball_filled.py", "src/features/frame_table.py")),
        "frame_table_fingerprint": frame_table.sha256_file(meta_path) if os.path.isfile(meta_path) else None,
        "uses_broadcast": bool(config.get("uses_broadcast")),
        "graded_frame_shares": {
            "evidence_tactical_ball": share(source == 0), "evidence_broadcast_fill_tier1": share(source == 1),
            "evidence_broadcast_fill_tier2": share(source == 2), "no_ball_evidence": share(source < 0),
            "low_margin_lt_1m": share(table["ball_margin_m"].lt(1).to_numpy()),
            "low_shape_conf_lt_0.5_home": share(table["home_shape_conf"].lt(0.5).to_numpy()),
            "low_shape_conf_lt_0.5_away": share(table["away_shape_conf"].lt(0.5).to_numpy()),
            "label_H": share(labels == "H"), "label_A": share(labels == "A"), "label_D": share(labels == "D")},
    }
    with open(os.path.splitext(csv_path)[0] + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "baseline.json"))
    ap.add_argument("--data-dir", help="defaults to MTAH_DATA_DIR or ./data")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--rebuild", action="store_true", help="rebuild the frame table from raw")
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    table = frame_table.load(rebuild=args.rebuild, data_dir=args.data_dir)
    print(frame_table.quality_report(table))
    path, sha = build_submission(config, args.out, table)
    print("valid: {}\nsha256: {}".format(os.path.relpath(path, ROOT), sha))
