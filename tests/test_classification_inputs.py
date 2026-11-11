"""Tests for the consolidated flat + object-set classification contract."""

import numpy as np
import pytest

from src.features import classification_inputs as ci
from src.features import model_inputs as mi
from tests.test_model_inputs import _slots, _table


def test_policy_rejects_acausal_schema_in_causal_mode():
    with pytest.raises(ValueError, match="contains acausal features"):
        ci.InputPolicy(schema=mi.SCHEMA_PHASE, temporal_mode=ci.TemporalMode.CAUSAL)


def test_policy_rejects_untyped_temporal_mode():
    with pytest.raises(ValueError, match="must be a TemporalMode"):
        ci.InputPolicy(schema=mi.SCHEMA_CORE, temporal_mode="causal")


def test_consolidated_inputs_preserve_flat_schema_and_object_shapes():
    table = _table()
    slots = _slots()
    result = ci.make_classification_inputs(
        table,
        slots,
        (105.0, 68.0),
        ci.InputPolicy(
            schema=mi.SCHEMA_PHASE,
            temporal_mode=ci.TemporalMode.OFFLINE_ACAUSAL,
        ),
        fps=1,
    )

    assert tuple(result.frame_features.columns) == mi.schema_columns(mi.SCHEMA_PHASE)
    assert result.players.values.shape == (8, 4, len(ci.PLAYER_VALUE_CHANNELS))
    assert result.ball.values.shape == (8, len(ci.BALL_VALUE_CHANNELS))
    assert result.players.team.tolist() == [0, 0, 1, 1]
    assert result.ball.seconds_until_seen_s is not None
    assert result.frame_id.tolist() == list(range(8))


def test_coordinates_use_one_home_attack_frame_across_halves():
    table = _table(4)
    table["home_attacks_right"] = [True, True, False, False]
    slots = _slots(4)
    slots["x"][:, 0] = [10.0, 11.0, 95.0, 94.0]

    result = ci.make_object_inputs(
        table, slots, (105.0, 68.0), ci.TemporalMode.OFFLINE_ACAUSAL, fps=1
    )
    x_channel = ci.PLAYER_VALUE_CHANNELS.index("x_home_norm")

    assert result.players.values[0, 0, x_channel] == pytest.approx(2 * 10 / 105 - 1)
    assert result.players.values[2, 0, x_channel] == pytest.approx(2 * 10 / 105 - 1)


def test_velocity_never_crosses_period_boundary():
    table = _table(4)
    slots = _slots(4)
    slots["x"][:, 0] = [10.0, 11.0, 90.0, 91.0]

    result = ci.make_object_inputs(
        table, slots, (105.0, 68.0), ci.TemporalMode.OFFLINE_ACAUSAL, fps=1
    )
    vx_channel = ci.PLAYER_VALUE_CHANNELS.index("vx_home_ms")

    assert result.players.values[1, 0, vx_channel] == pytest.approx(1.0)
    assert np.isnan(result.players.values[2, 0, vx_channel])


def test_slot_permutation_only_permutes_object_axis():
    table = _table()
    slots = _slots()
    permutation = np.array([2, 0, 3, 1])
    permuted = {name: values[:, permutation] for name, values in slots.items() if name in {"x", "y", "s"}}
    permuted["team"] = slots["team"][permutation]

    original = ci.make_object_inputs(
        table, slots, (105.0, 68.0), ci.TemporalMode.OFFLINE_ACAUSAL, fps=1
    ).players
    changed = ci.make_object_inputs(
        table, permuted, (105.0, 68.0), ci.TemporalMode.OFFLINE_ACAUSAL, fps=1
    ).players

    np.testing.assert_allclose(changed.values, original.values[:, permutation], equal_nan=True)
    np.testing.assert_array_equal(changed.observed, original.observed[:, permutation])
    np.testing.assert_array_equal(changed.keeper_proxy, original.keeper_proxy[:, permutation])
    np.testing.assert_array_equal(changed.team, original.team[permutation])


def test_missingness_has_mask_and_observation_age_without_imputation():
    table = _table(4)
    slots = _slots(4)
    slots["x"][1:3, 0] = np.nan
    slots["y"][1:3, 0] = np.nan
    slots["s"][1:3, 0] = np.nan

    result = ci.make_object_inputs(
        table, slots, (105.0, 68.0), ci.TemporalMode.OFFLINE_ACAUSAL, fps=2
    ).players

    assert result.observed[:, 0].tolist() == [True, False, False, True]
    assert result.seconds_since_seen_s[0, 0] == 0.0
    assert result.seconds_since_seen_s[1, 0] == pytest.approx(0.5)
    assert np.isnan(result.seconds_since_seen_s[2, 0])
    assert result.seconds_since_seen_s[3, 0] == 0.0
    assert np.isnan(result.values[1:3, 0]).all()


def test_causal_object_view_does_not_expose_future_observation_age():
    result = ci.make_object_inputs(
        _table(),
        _slots(),
        (105.0, 68.0),
        ci.TemporalMode.CAUSAL,
        fps=1,
    )

    assert result.ball.seconds_until_seen_s is None


def test_unobserved_coordinates_never_leak_partial_values():
    table = _table()
    slots = _slots()
    slots["x"][1, 0] = np.nan
    table.loc[1, "ball_visible"] = False

    result = ci.make_object_inputs(
        table, slots, (105.0, 68.0), ci.TemporalMode.OFFLINE_ACAUSAL, fps=1
    )

    assert not result.players.observed[1, 0]
    assert np.isnan(result.players.values[1, 0]).all()
    assert not result.ball.observed[1]
    assert np.isnan(result.ball.values[1]).all()


def test_object_view_requires_slots_from_both_teams():
    slots = _slots()
    slots["team"][:] = 0

    with pytest.raises(ValueError, match="must contain both teams"):
        ci.make_object_inputs(
            _table(), slots, (105.0, 68.0), ci.TemporalMode.OFFLINE_ACAUSAL, fps=1
        )
