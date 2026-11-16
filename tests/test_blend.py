import os
import sys

import numpy as np

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
