"""Keep rule: a change is kept only if its spellF1 gain beats zero under a resampling test over 2-minute chunks.

The validation frames are cut into 2-minute chunks (inside the graded runs). Each chunk is scored with spellF1 for
the baseline and the candidate, and the per-chunk gains are bootstrapped. Keep = the lower bound of the two-sided 95%
interval is above 0. Ties (an interval spanning 0) go to the simpler option: the baseline when the candidate adds
something, the candidate when it only removes something.
"""

import numpy as np

from evaluation import possession_metrics as pm

CHUNK = 3000            # 2 minutes at 25 fps


def chunk_gains(gt_codes, base_codes, cand_codes):
    gains = []
    for a, b in pm.valid_runs(gt_codes):
        for s in range(a, b, CHUNK):
            e = min(b, s + CHUNK)
            if e - s < CHUNK // 2:
                continue
            runs = [(s, e)]
            f = lambda p: pm.spell_scores(gt_codes, p, runs)["mean_f1"]  # noqa: E731
            gains.append(f(cand_codes) - f(base_codes))
    return np.array(gains)


def keep_test(gains, n_draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(gains, size=(n_draws, len(gains)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_gain": float(gains.mean()), "ci95": (float(lo), float(hi)), "chunks": len(gains), "keep": bool(lo > 0)}
