import numpy as np
import pytest

from src.decoder import semi_markov


def _training_model() -> semi_markov.DurationModel:
    labels = np.asarray(list("HHAAADDDDDDDHHAAADDDDDDDD"))
    periods = np.asarray([1] * 13 + [2] * (len(labels) - 13))
    return semi_markov.fit_duration_model(
        [(labels, periods)], token_frames=1
    )


def test_duration_model_learns_class_specific_training_durations() -> None:
    model = _training_model()

    assert model.duration_mu[2] > model.duration_mu[0]
    assert model.duration_mu[2] > model.duration_mu[1]
    assert np.allclose(np.exp(model.duration_log_probability).sum(axis=1), 1)
    assert np.isneginf(np.diag(model.transition_log_probability)).all()


def test_decode_returns_period_safe_full_frame_labels() -> None:
    model = _training_model()
    probabilities = np.vstack(
        (
            np.tile([0.98, 0.01, 0.01], (2, 1)),
            np.tile([0.01, 0.98, 0.01], (2, 1)),
        )
    )
    periods = np.asarray([1] * 2 + [2] * 2)

    labels = semi_markov.decode(probabilities, periods, model)

    assert labels.shape == (4,)
    assert set(labels[:2]) == {"H"}
    assert set(labels[2:]) == {"A"}


def test_decode_rejects_non_probability_rows() -> None:
    model = _training_model()
    with pytest.raises(ValueError, match="sum to one"):
        semi_markov.decode(np.ones((5, 3)), np.ones(5), model)
