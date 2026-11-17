"""Write the final BAR-ATH possession CSV from the selected blend (configs/best_model_selection.json).

    python scripts/make_submission.py --out <team_name>.csv

Needs proba_BAR-ATH.npy in every member folder (P2/P4: train_p2/train_p4 --export; P3: scripts/export_p3_bar_ath.py).
Blends member probabilities (equal arithmetic), applies the common Viterbi smoother once, validates the file.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "evaluation")]

from scripts.blend import blend  # noqa: E402
from src.decoder.viterbi import smooth  # noqa: E402
from src.features import frame_table as ft  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=os.path.join(ROOT, "configs", "best_model_selection.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", help="BAR-ATH bundle root; defaults to MTAH_DATA_DIR or ./data")
    args = ap.parse_args()
    cand = json.load(open(args.manifest))["candidate"]
    proba = blend([np.load(os.path.join(ROOT, m, "proba_BAR-ATH.npy")) for m in cand["members"]])
    table = ft.load(data_dir=args.data_dir)
    labels = smooth(proba, table["period"], cand["switch_penalty"])
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(args.out, index=False)
    print("wrote", args.out, "sha256", hashlib.sha256(open(args.out, "rb").read()).hexdigest())
    subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), args.out], check=True)


if __name__ == "__main__":
    main()
