import numpy as np
import pytest

from src.models import linear_stacker


def _probability_from_codes(codes, confidence):
    probability = np.full((len(codes), 3), (1 - confidence) / 2)
    probability[np.arange(len(codes)), codes] = confidence
    return probability


def test_stack_features_returns_member_log_probabilities():
    first = np.tile([0.8, 0.1, 0.1], (4, 1))
    second = np.tile([0.1, 0.8, 0.1], (4, 1))

    features = linear_stacker.stack_features([first, second])

    assert features.shape == (4, 6)
    assert np.allclose(features[:, :3], np.log(first))
    assert np.allclose(features[:, 3:], np.log(second))


def test_linear_stacker_learns_reliable_member_and_normalizes():
    codes = np.tile(np.arange(3), 100)
    labels = np.asarray(linear_stacker.CLASSES)[codes]
    reliable = _probability_from_codes(codes, 0.9)
    wrong = _probability_from_codes((codes + 1) % 3, 0.9)
    features = linear_stacker.stack_features([reliable, wrong])

    model = linear_stacker.LinearStacker.fit(features, labels)
    probability = model.predict_proba(features)

    assert np.mean(probability.argmax(axis=1) == codes) > 0.99
    assert np.allclose(probability.sum(axis=1), 1.0)


def test_linear_stacker_round_trip(tmp_path):
    codes = np.tile(np.arange(3), 20)
    labels = np.asarray(linear_stacker.CLASSES)[codes]
    features = linear_stacker.stack_features([_probability_from_codes(codes, 0.8)])
    model = linear_stacker.LinearStacker.fit(features, labels)
    path = tmp_path / "stacker.npz"

    model.save(path)
    restored = linear_stacker.LinearStacker.load(path)

    assert np.allclose(restored.predict_proba(features), model.predict_proba(features))
    assert restored.classes == linear_stacker.CLASSES


def test_stacker_rejects_misaligned_or_unnormalized_inputs():
    valid = np.full((6, 3), 1 / 3)
    with pytest.raises(ValueError, match="identical"):
        linear_stacker.stack_features([valid, valid[:5]])
    with pytest.raises(ValueError, match="normalized"):
        linear_stacker.stack_features([valid * 2])
    with pytest.raises(ValueError, match="align"):
        linear_stacker.LinearStacker.fit(np.ones((6, 3)), ["H"] * 5)
