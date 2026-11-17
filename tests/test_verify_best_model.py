import hashlib

import numpy as np
import pytest

from scripts import verify_best_model


def _manifest(member, match, digest):
    return {
        "candidate": {
            "members": [member],
            "validation_per_match_spellF1": {match: 0.0},
            "validation_probability_sha256": {match: {member: digest}},
        }
    }


def test_load_verified_probabilities_accepts_exact_hash(tmp_path):
    member = "outputs/model"
    match = "MATCH"
    directory = tmp_path / member
    directory.mkdir(parents=True)
    path = directory / ("proba_" + match + ".npy")
    probability = np.full((4, 3), 1 / 3)
    np.save(path, probability)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    loaded = verify_best_model.load_verified_probabilities(
        _manifest(member, match, digest), tmp_path
    )

    np.testing.assert_array_equal(loaded[match][0], probability)


def test_load_verified_probabilities_rejects_changed_artifact(tmp_path):
    member = "outputs/model"
    match = "MATCH"
    directory = tmp_path / member
    directory.mkdir(parents=True)
    path = directory / ("proba_" + match + ".npy")
    np.save(path, np.full((4, 3), 1 / 3))

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_best_model.load_verified_probabilities(
            _manifest(member, match, "0" * 64), tmp_path
        )


def test_load_verified_probabilities_rejects_path_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        verify_best_model.load_verified_probabilities(
            _manifest("../outside", "MATCH", "0" * 64), tmp_path
        )
