"""P4-local extra damaged copies of the TRAINING matches (option A2: more ball-gap variety in training).

Same damage code and RANGES as scripts/build_sportec_bundles.py; only the per-match seed changes, so each copy draws
new ball gaps, tracklets and noise. Validation and test matches are never copied: every pipeline is scored on the
one common damaged version. Output goes to cache/p4/aug<k>/ (git-ignored) and is deterministic for Rule 3.

    python scripts/build_p4_aug.py --copies 2
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_sportec_bundles as builder
from src.common import sportec

AUG_DIR = os.path.join(ROOT, "cache", "p4")
SEED_OFFSET = 100   # copy k uses seed_for(match) + 100 * k; the builder itself uses seed, seed + 1, seed + 2


def bundle_dir(k, match):
    return os.path.join(AUG_DIR, "aug{}".format(k), "bundles", match)


def build(k, match):
    base_seed = builder.seed_for
    builder.OUT_DIR = os.path.join(AUG_DIR, "aug{}".format(k), "bundles")
    builder.LABEL_DIR = os.path.join(AUG_DIR, "aug{}".format(k), "labels")
    builder.seed_for = lambda m: base_seed(m) + SEED_OFFSET * k
    try:
        return builder.build(match)
    finally:
        builder.seed_for = base_seed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=2)
    for k in range(1, ap.parse_args().copies + 1):
        for match in sportec.split()["train"]:
            meta = build(k, match)
            print("aug{} {} seed {}".format(k, match, meta["seed"]), flush=True)
