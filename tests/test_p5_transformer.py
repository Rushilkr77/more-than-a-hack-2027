import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.models import p5_transformer as p5


def _synthetic_sequence(seed: int = 0) -> p5.SequenceExample:
    rng = np.random.default_rng(seed)
    frames = 96
    period = np.r_[np.ones(55, dtype=np.int64), np.full(41, 2, dtype=np.int64)]
    frame = pd.DataFrame(
        {
            "shape": rng.normal(size=frames),
            "ball": rng.normal(size=frames),
        }
    )
    frame.loc[rng.random(frames) < 0.25, "ball"] = np.nan
    labels = np.where(
        frame["shape"] > 0.4,
        "H",
        np.where(frame["shape"] < -0.4, "A", "D"),
    )
    scaler = p5.fit_scaler(frame)
    return p5.SequenceExample(
        values=p5.transform(frame, scaler),
        labels=p5.encode(labels),
        period=period,
    )


def test_windows_never_cross_period_boundaries() -> None:
    period = np.r_[np.ones(17, dtype=np.int64), np.full(13, 2, dtype=np.int64)]

    spans = p5.windows(period, length=10, stride=5)

    assert spans
    assert all(np.unique(period[start:end]).size == 1 for start, end in spans)


@pytest.mark.parametrize("length,stride", [(0, 1), (5, 0), (-1, 2)])
def test_windows_reject_invalid_sizes(length: int, stride: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        p5.windows(np.ones(10, dtype=np.int64), length=length, stride=stride)


def test_scaler_preserves_missingness_with_observation_flags() -> None:
    frame = pd.DataFrame({"complete": [1.0, 2.0, 3.0], "partial": [4.0, np.nan, 6.0]})

    scaler = p5.fit_scaler(frame)
    values = p5.transform(frame, scaler)

    assert scaler.masked == ("partial",)
    assert values.shape == (3, 3)
    assert np.isfinite(values).all()
    assert values[:, -1].tolist() == [1.0, 0.0, 1.0]


@pytest.mark.parametrize(
    "column", ["frame", "frame_id", "match_id", "provider_id", "team_id", "track_id"]
)
def test_scaler_rejects_frame_identifiers(column: str) -> None:
    with pytest.raises(ValueError, match="alignment-only"):
        p5.fit_scaler(pd.DataFrame({column: [1, 2], "feature": [0.0, 1.0]}))


def test_transformer_uses_position_and_returns_per_frame_logits() -> None:
    torch.manual_seed(0)
    config = p5.TransformerConfig(
        input_features=3,
        model_width=16,
        heads=4,
        layers=1,
        feedforward_width=32,
        dropout=0.0,
        max_tokens=32,
    )
    model = p5.TemporalTransformer(config).eval()
    identical_tokens = torch.zeros(1, 8, 3)

    logits = model(identical_tokens)

    assert logits.shape == (1, 8, 3)
    assert not torch.allclose(logits[:, 0], logits[:, 1])


def test_default_model_stays_within_declared_capacity_band() -> None:
    model = p5.TemporalTransformer(p5.TransformerConfig(input_features=123))

    parameters = p5.parameter_count(model)

    assert 70_000 <= parameters <= 100_000


def test_equal_probability_blend_is_fixed_and_validated() -> None:
    first = np.asarray([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]], dtype=np.float32)
    second = np.asarray([[0.4, 0.4, 0.2], [0.4, 0.4, 0.2]], dtype=np.float32)

    result = p5.equal_probability_blend(first, second)

    np.testing.assert_allclose(result, (first + second) / 2)
    with pytest.raises(ValueError, match="identical shapes"):
        p5.equal_probability_blend(first, second[:1])


def test_full_frame_prediction_is_normalized() -> None:
    example = _synthetic_sequence()
    torch.manual_seed(1)
    model = p5.TemporalTransformer(
        p5.TransformerConfig(
            input_features=example.values.shape[1],
            model_width=16,
            heads=4,
            layers=1,
            feedforward_width=32,
            dropout=0.0,
            max_tokens=32,
        )
    ).eval()

    probabilities = p5.predict_proba(
        model,
        example.values,
        example.period,
        length=24,
        step=3,
    )

    assert probabilities.shape == (len(example.values), 3)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_synthetic_training_replays_exactly_for_same_seed() -> None:
    example = _synthetic_sequence()
    config = p5.TransformerConfig(
        input_features=example.values.shape[1],
        model_width=16,
        heads=4,
        layers=1,
        feedforward_width=32,
        dropout=0.0,
        max_tokens=32,
    )

    first, first_summary = p5.train(
        [example],
        [example],
        config=config,
        length=24,
        seed=7,
        max_epochs=2,
        batch_size=4,
        log=lambda _: None,
    )
    second, second_summary = p5.train(
        [example],
        [example],
        config=config,
        length=24,
        seed=7,
        max_epochs=2,
        batch_size=4,
        log=lambda _: None,
    )

    first_probabilities = p5.predict_proba(
        first, example.values, example.period, length=24, step=3
    )
    second_probabilities = p5.predict_proba(
        second, example.values, example.period, length=24, step=3
    )
    assert first_summary == second_summary
    assert np.array_equal(first_probabilities, second_probabilities)
