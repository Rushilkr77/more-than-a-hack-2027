import numpy as np
import pytest
import json

from src.features import model_inputs as mi
from src.models import p3_tcn
from src.common import sportec
from src.features import classification_inputs as ci
from tests.test_model_inputs import _slots, _table
from scripts import train_p3


def test_p3_config_locks_the_approved_long_context_window():
    config = p3_tcn.load_config()

    assert config.schemas == (mi.SCHEMA_BALL, mi.SCHEMA_PHASE)
    assert config.past_frames == 200
    assert config.future_frames == 52
    assert config.window_frames == 253
    assert config.core_frames == 256
    assert config.causal is True
    assert config.dilations == (1, 2, 4, 8, 16, 32)
    assert config.receptive_field == 253
    assert config.output_dir == "outputs/p3_long_context_253"


def test_schema_columns_are_the_exact_supplied_allow_lists():
    config = p3_tcn.load_config()

    assert p3_tcn.feature_columns(config, mi.SCHEMA_BALL) == mi.schema_columns(mi.SCHEMA_BALL)
    assert p3_tcn.feature_columns(config, mi.SCHEMA_PHASE) == mi.schema_columns(mi.SCHEMA_PHASE)
    assert len(p3_tcn.feature_columns(config, mi.SCHEMA_BALL)) == 60
    assert len(p3_tcn.feature_columns(config, mi.SCHEMA_PHASE)) == 80
    assert sum(name.startswith("gtag_") for name in p3_tcn.feature_columns(config, mi.SCHEMA_PHASE)) == 8


def test_train_only_scaler_centres_each_existing_column_and_imputes_missing_after_scaling():
    scaler = p3_tcn.P3Scaler.fit([np.array([[8.0], [10.0], [12.0], [14.0]], dtype=np.float32)])

    transformed = scaler.transform(np.array([[8.0], [11.0], [14.0], [np.nan]], dtype=np.float32))

    assert np.allclose(scaler.mean, [11.0])
    assert np.allclose(transformed[:3, 0], [-1.3416408, 0.0, 1.3416408])
    assert transformed[3, 0] == 0.0


def test_target_window_uses_exactly_eight_seconds_past_and_about_two_seconds_future():
    frames = np.arange(400, dtype=np.float32)[:, None]
    periods = np.ones(400, dtype=np.int8)

    windows = p3_tcn.target_windows(frames, periods, np.array([200]))

    assert windows.shape == (1, 1, 253)
    assert windows[0, 0].tolist() == list(range(253))


def test_target_window_never_reads_across_a_period_boundary():
    frames = np.arange(8, dtype=np.float32)[:, None]
    periods = np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int8)

    windows = p3_tcn.target_windows(frames, periods, np.array([4]), past_frames=2, future_frames=1)

    assert windows[0, 0].tolist() == [4.0, 4.0, 4.0, 5.0]


def test_period_chunk_produces_many_aligned_targets_without_crossing_a_period():
    frames = np.arange(8, dtype=np.float32)[:, None]
    periods = np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int8)

    window, valid_targets = p3_tcn.period_chunk(
        frames, periods, core_start=4, past_frames=2, future_frames=1, core_frames=3
    )

    assert valid_targets == 3
    assert window.shape == (1, 6)
    assert window[0].tolist() == [4.0, 4.0, 4.0, 5.0, 6.0, 7.0]


def test_packaged_tcn_emits_three_logits_for_the_target_frame():
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_tcn")
    config = p3_tcn.load_config()
    model = p3_tcn.P3Model(config, mi.SCHEMA_BALL)

    logits = model(torch.zeros(2, 60, config.window_frames))

    assert logits.shape == (2, 3, config.window_frames)
    assert p3_tcn.target_logits(logits).shape == (2, 3)


def test_feature_matrix_uses_the_exact_shared_schema_frame():
    table = _table(200)
    table["period"] = np.r_[np.ones(100), np.full(100, 2)]
    match = sportec.MatchData(
        table=table,
        slots=_slots(200),
        labels={"raw": np.full(200, "H"), "adjusted": np.full(200, "H")},
        pitch=(105.0, 68.0),
        fps=25,
    )
    config = p3_tcn.load_config()

    values, periods = p3_tcn.feature_matrix(match, config, mi.SCHEMA_BALL)
    expected = sportec.classification_inputs(
        match,
        ci.InputPolicy(schema=mi.SCHEMA_BALL, temporal_mode=ci.TemporalMode.OFFLINE_ACAUSAL),
    ).frame_features.loc[:, mi.schema_columns(mi.SCHEMA_BALL)].to_numpy(dtype=np.float32)

    assert np.allclose(values, expected, equal_nan=True)
    assert np.array_equal(periods, table["period"].to_numpy())


def test_artifact_round_trip_keeps_schema_scaler_and_smoother_penalty(tmp_path):
    torch = pytest.importorskip("torch")
    config = p3_tcn.load_config()
    scaler = p3_tcn.P3Scaler(mean=np.zeros(60, dtype=np.float32), scale=np.ones(60, dtype=np.float32))
    model = p3_tcn.P3Model(config, mi.SCHEMA_BALL)

    p3_tcn.save_artifact(tmp_path, model, mi.SCHEMA_BALL, scaler, switch_penalty=20)
    restored, schema, restored_scaler, penalty = p3_tcn.load_artifact(tmp_path, config)

    assert schema == mi.SCHEMA_BALL
    assert penalty == 20
    assert np.array_equal(restored_scaler.mean, scaler.mean)
    assert restored(torch.zeros(1, 60, config.window_frames)).shape == (1, 3, config.window_frames)


def test_selection_prefers_spell_f1_then_macro_f1():
    selected = p3_tcn.select_run([
        {"schema": mi.SCHEMA_BALL, "spellF1": 0.6, "macroF1": 0.7},
        {"schema": mi.SCHEMA_PHASE, "spellF1": 0.6, "macroF1": 0.8},
    ])

    assert selected["schema"] == mi.SCHEMA_PHASE


def test_train_script_runs_both_configured_schemas_by_default():
    config = p3_tcn.load_config()

    assert train_p3.schemas_to_run(config, None) == config.schemas
    assert train_p3.schemas_to_run(config, mi.SCHEMA_BALL) == (mi.SCHEMA_BALL,)


def test_training_log_is_machine_readable_and_flushed(capsys):
    train_p3.log_event("schema_started", schema=mi.SCHEMA_BALL, epoch=1)

    record = json.loads(capsys.readouterr().out)
    assert record == {"event": "schema_started", "schema": mi.SCHEMA_BALL, "epoch": 1}


def test_temporal_smoothing_loss_is_zero_for_constant_logits():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[[2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]])
    labels = torch.tensor([[0, 0, 0]])

    loss = train_p3.temporal_smoothing_loss(logits, labels, tau=4.0)

    assert float(loss) == pytest.approx(0.0)


def test_temporal_smoothing_loss_ignores_padded_pairs_and_truncates():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[[20.0, -20.0, 20.0], [-20.0, 20.0, -20.0], [0.0, 0.0, 0.0]]])
    labels = torch.tensor([[0, 1, -100]])

    loss = train_p3.temporal_smoothing_loss(logits, labels, tau=2.0)

    assert 0.0 < float(loss) <= 4.0
