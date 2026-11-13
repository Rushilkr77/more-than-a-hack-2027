import numpy as np

from src.decoder.viterbi import smooth


def _proba(seq, p=0.8):
    m = {"H": 0, "A": 1, "D": 2}
    out = np.full((len(seq), 3), (1 - p) / 2)
    out[np.arange(len(seq)), [m[c] for c in seq]] = p
    return out


def test_zero_penalty_is_argmax():
    seq = "HHAHDDA"
    assert "".join(smooth(_proba(seq), np.ones(len(seq)), 0.0)) == seq


def test_short_flicker_removed_sustained_switch_kept():
    seq = "H" * 50 + "A" * 2 + "H" * 50 + "A" * 100
    out = "".join(smooth(_proba(seq), np.ones(len(seq)), 5.0))
    assert out == "H" * 102 + "A" * 100


def test_no_smoothing_across_period_change():
    seq = "H" * 30 + "A" * 30
    period = np.r_[np.ones(30), np.full(30, 2)]
    out = "".join(smooth(_proba(seq), period, 1e6))
    assert out == seq


def test_default_separate_penalties_match_single():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(3), size=300)
    per = np.ones(300)
    assert (smooth(p, per, 3.0) == smooth(p, per, 3.0, team_penalty=3.0, dead_penalty=3.0)).all()


def test_separate_penalties_cheap_dead_switch_kept_team_flicker_removed():
    # note: a team flicker can also be routed H->D->A, so it is only removed if 4 * dead_penalty is costly too
    seq = "H" * 50 + "D" * 10 + "H" * 50 + "A" * 3 + "H" * 50
    out = "".join(smooth(_proba(seq), np.ones(len(seq)), 0, team_penalty=20.0, dead_penalty=3.0))
    assert out == "H" * 50 + "D" * 10 + "H" * 103


def test_min_frames_one_equals_plain():
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(3), size=200)
    per = np.ones(200)
    assert (smooth(p, per, 2.0) == smooth(p, per, 2.0, min_frames=1)).all()


def test_min_frames_removes_short_spell_without_penalty():
    seq = "H" * 40 + "A" * 4 + "H" * 40 + "A" * 40
    out = "".join(smooth(_proba(seq), np.ones(len(seq)), 0, min_frames=10))
    assert out == "H" * 84 + "A" * 40
    assert min(len(r) for r in out.replace("HA", "H A").split()) >= 10
