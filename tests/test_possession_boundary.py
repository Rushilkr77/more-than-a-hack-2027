from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.decoder.viterbi import smooth
from src.models import possession_boundary as boundary


def test_constant_frame_cost_matches_existing_smoother() -> None:
    probabilities = np.random.default_rng(0).dirichlet(np.ones(3), size=80)
    period = np.repeat([1, 2], 40)
    expected = smooth(probabilities, period, 3)
    actual = smooth(probabilities, period, 3, frame_switch_cost=np.full(80, 3))
    np.testing.assert_array_equal(actual, expected)


def test_viterbi_uses_cost_of_entering_frame_and_resets_at_half() -> None:
    probabilities = np.array([[0.89, 0.1, 0.01], [0.89, 0.1, 0.01],
                              [0.1, 0.89, 0.01], [0.1, 0.89, 0.01],
                              [0.1, 0.89, 0.01], [0.1, 0.89, 0.01]])
    period = np.array([1, 1, 1, 1, 2, 2])
    costs = np.array([20, 20, 0, 20, 200, 20])
    result = smooth(probabilities, period, 20, frame_switch_cost=costs)
    np.testing.assert_array_equal(result, ["H", "H", "A", "A", "A", "A"])


@pytest.mark.parametrize("costs", [[1], [1, -1], [1, np.nan]])
def test_viterbi_rejects_invalid_frame_costs(costs: list) -> None:
    with pytest.raises(ValueError, match="one finite, non-negative value"):
        smooth(np.full((2, 3), 1 / 3), [1, 1], 20, frame_switch_cost=np.array(costs))


def test_boundary_targets_exclude_unknown_edges_and_period_starts() -> None:
    labels = np.array(["H", "H", "A", "?", "D", "D", "H", "H", "A"])
    period = np.array([1] * 6 + [2] * 3)
    expected = [-1, 1, 1, -1, -1, 0, -1, 1, 1]
    np.testing.assert_array_equal(boundary.targets(labels, period, 5), expected)


def test_features_respect_periods_and_mirroring() -> None:
    columns = len(boundary.COLUMNS)
    values = np.repeat([[1], [2], [300], [400]], columns, axis=1)
    frame = pd.DataFrame(values, columns=boundary.COLUMNS)
    period = np.array([1, 1, 2, 2])
    normal = boundary.features(SimpleNamespace(frame_features=frame, period=period), 25)
    mirrored_frame = frame.assign(ball_margin_m=-frame.ball_margin_m)
    mirrored = boundary.features(SimpleNamespace(frame_features=mirrored_frame, period=period), 25)
    np.testing.assert_array_equal(normal, mirrored)
    assert normal[1, columns * 2] == 2
    assert normal[2, 0] == 300


def test_switch_cost_is_neutral_at_prior_and_bounded() -> None:
    costs = boundary.switch_cost(np.array([0.0, 0.04, 1.0]), 0.04, 20)
    np.testing.assert_allclose(costs, [30, 20, 10])
