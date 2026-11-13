import numpy as np

from evaluation import possession_metrics as pm
from src.common import keep_rule


def test_identical_predictions_are_not_kept_and_a_better_one_is():
    rng = np.random.default_rng(0)
    gt = np.repeat(rng.integers(0, 3, 400), 50)                    # 20,000 frames of 2 s spells
    noisy = gt.copy()
    flip = rng.random(len(gt)) < 0.05
    noisy[flip] = rng.integers(0, 3, flip.sum())                  # flicker = fragmented spells
    same = keep_rule.keep_test(keep_rule.chunk_gains(gt, noisy, noisy))
    better = keep_rule.keep_test(keep_rule.chunk_gains(gt, noisy, gt))
    assert not same["keep"] and better["keep"] and better["mean_gain"] > 0
