import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import blend


def test_blend_identity_weights_and_normalisation():
    rng = np.random.default_rng(0)
    a, b = rng.dirichlet(np.ones(3), 50), rng.dirichlet(np.ones(3), 50)
    assert np.array_equal(blend.blend([a, b], [1, 0]), a) and np.array_equal(blend.blend([a, b], [0, 1]), b)
    assert np.allclose(blend.blend([a, b]), (a + b) / 2)
    assert np.allclose(blend.blend([a, b], [3, 1]), 0.75 * a + 0.25 * b)
    geo = blend.blend([a, b], mean="geo")
    assert np.allclose(geo.sum(axis=1), 1) and np.allclose(blend.blend([a, a], mean="geo"), a, atol=1e-5)


def test_blend_rejects_invalid_members_and_weights():
    valid = np.full((5, 3), 1 / 3)
    with pytest.raises(ValueError, match="same"):
        blend.blend([valid, valid[:4]])
    with pytest.raises(ValueError, match="summing to one"):
        blend.blend([valid * 2])
    with pytest.raises(ValueError, match="weights"):
        blend.blend([valid, valid], [1])
    with pytest.raises(ValueError, match="weights"):
        blend.blend([valid, valid], [1, -1])


def test_missing_ball_blend_only_adds_extra_member_where_missing():
    home = np.tile([0.8, 0.1, 0.1], (4, 1))
    away = np.tile([0.1, 0.8, 0.1], (4, 1))
    dead = np.tile([0.1, 0.1, 0.8], (4, 1))
    visible = np.asarray([True, False, True, False])

    result = blend.missing_ball_blend([home, away], dead, visible)

    assert np.allclose(result[visible], (home[visible] + away[visible]) / 2)
    assert np.allclose(result[~visible], (home[~visible] + away[~visible] + dead[~visible]) / 3)


def test_missing_ball_blend_requires_aligned_boolean_mask():
    proba = np.full((4, 3), 1 / 3)
    with pytest.raises(ValueError, match="one boolean"):
        blend.missing_ball_blend([proba], proba, np.ones(4))
