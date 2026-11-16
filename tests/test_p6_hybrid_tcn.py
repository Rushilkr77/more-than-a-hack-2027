import torch

from src.models.p6_hybrid_tcn import HybridBatch, HybridTCN, TrajectorySetTCN


def _batch(order=None):
    torch.manual_seed(0)
    batch, frames, slots = 2, 17, 6
    order = torch.arange(slots) if order is None else order
    team = torch.tensor([0, 0, 0, 1, 1, 1])[order]
    values = torch.randn(batch, frames, slots, 5)[:, :, order]
    observed = torch.rand(batch, frames, slots)[:, :, order] > 0.2
    return HybridBatch(
        flat=torch.randn(batch, frames, 10),
        player_values=values,
        player_observed=observed,
        player_inside=observed,
        player_age_s=torch.rand(batch, frames, slots)[:, :, order],
        player_keeper=torch.zeros(batch, frames, slots, dtype=torch.bool)[:, :, order],
        player_team=team,
        ball_values=torch.randn(batch, frames, 4),
        ball_observed=torch.ones(batch, frames, dtype=torch.bool),
        ball_inside=torch.ones(batch, frames, dtype=torch.bool),
        ball_since_s=torch.zeros(batch, frames),
        ball_until_s=torch.zeros(batch, frames),
    )


def test_output_shape_and_receptive_field():
    model = HybridTCN(flat_features=10, hidden=16, object_hidden=8)
    assert model(_batch()).shape == (2, 17, 3)
    assert model.receptive_field_frames == 127


def test_slot_permutation_does_not_change_predictions():
    model = HybridTCN(flat_features=10, hidden=16, object_hidden=8).eval()
    original = _batch()
    order = torch.tensor([2, 0, 1, 5, 3, 4])
    permuted = HybridBatch(
        flat=original.flat,
        player_values=original.player_values[:, :, order],
        player_observed=original.player_observed[:, :, order],
        player_inside=original.player_inside[:, :, order],
        player_age_s=original.player_age_s[:, :, order],
        player_keeper=original.player_keeper[:, :, order],
        player_team=original.player_team[order],
        ball_values=original.ball_values,
        ball_observed=original.ball_observed,
        ball_inside=original.ball_inside,
        ball_since_s=original.ball_since_s,
        ball_until_s=original.ball_until_s,
    )
    with torch.no_grad():
        assert torch.allclose(model(original), model(permuted), atol=1e-6)


def test_trajectory_first_model_has_declared_asymmetric_window():
    model = TrajectorySetTCN(flat_features=10, hidden=16, object_hidden=8)
    assert model.history_frames == 75
    assert model.future_frames == 25
    assert model.receptive_field_frames == 101
    assert model(_batch()).shape == (2, 17, 3)


def test_trajectory_first_model_is_slot_permutation_invariant():
    model = TrajectorySetTCN(flat_features=10, hidden=16, object_hidden=8).eval()
    original = _batch()
    order = torch.tensor([5, 1, 3, 0, 4, 2])
    permuted = HybridBatch(
        flat=original.flat,
        player_values=original.player_values[:, :, order],
        player_observed=original.player_observed[:, :, order],
        player_inside=original.player_inside[:, :, order],
        player_age_s=original.player_age_s[:, :, order],
        player_keeper=original.player_keeper[:, :, order],
        player_team=original.player_team[order],
        ball_values=original.ball_values,
        ball_observed=original.ball_observed,
        ball_inside=original.ball_inside,
        ball_since_s=original.ball_since_s,
        ball_until_s=original.ball_until_s,
    )
    with torch.no_grad():
        assert torch.allclose(model(original), model(permuted), atol=1e-6)

