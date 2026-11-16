"""Build two deterministic, train-only damaged copies for P5.

The damage code and ranges are shared with the common Sportec bundle builder.
Only the seed changes. Validation and test matches are never copied.
"""

import argparse
import os
import sys
from typing import Dict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_sportec_bundles as builder
from src.common import sportec


AUG_DIR = os.path.join(ROOT, "cache", "p5")
SEED_OFFSET = 100


def bundle_dir(copy_index: int, match: str) -> str:
    if copy_index <= 0:
        raise ValueError("copy_index must be positive")
    return os.path.join(AUG_DIR, "aug{}".format(copy_index), "bundles", match)


def build(copy_index: int, match: str) -> Dict[str, object]:
    """Build one copy while restoring builder globals even after failure."""

    original_seed_for = builder.seed_for
    original_out_dir = builder.OUT_DIR
    original_label_dir = builder.LABEL_DIR
    builder.OUT_DIR = os.path.join(AUG_DIR, "aug{}".format(copy_index), "bundles")
    builder.LABEL_DIR = os.path.join(AUG_DIR, "aug{}".format(copy_index), "labels")
    builder.seed_for = lambda name: original_seed_for(name) + SEED_OFFSET * copy_index
    try:
        return builder.build(match)
    finally:
        builder.seed_for = original_seed_for
        builder.OUT_DIR = original_out_dir
        builder.LABEL_DIR = original_label_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=2)
    args = parser.parse_args()
    if args.copies <= 0:
        raise ValueError("copies must be positive")
    for copy_index in range(1, args.copies + 1):
        for match in sportec.split()["train"]:
            metadata = build(copy_index, match)
            print(
                "aug{} {} seed {}".format(copy_index, match, metadata["seed"]),
                flush=True,
            )


if __name__ == "__main__":
    main()
