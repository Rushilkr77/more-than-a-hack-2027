"""P3's exact-schema, offline temporal-window preparation."""

import json
import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from pytorch_tcn import TCN

from src.common import sportec
from src.features import classification_inputs as ci
from src.features import model_inputs as mi


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(ROOT, "configs", "p3_tcn.json")


@dataclass(frozen=True)
class P3Config:
    schemas: Tuple[str, ...]
    classes: Tuple[str, ...]
    fps: int
    past_frames: int
    future_frames: int
    core_frames: int
    channels: Tuple[int, ...]
    kernel_size: int
    dilations: Tuple[int, ...]
    dropout: float
    causal: bool
    input_shape: str
    use_norm: str
    activation: str
    kernel_initializer: str
    use_skip_connections: bool
    use_gate: bool
    output_activation: Optional[str]
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    seed: int

    @property
    def window_frames(self) -> int:
        return self.past_frames + 1 + self.future_frames

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)


def load_config(path: str = CONFIG_PATH) -> P3Config:
    """Load the single versioned P3 configuration."""
    with open(path) as file:
        values = json.load(file)
    config = P3Config(
        schemas=tuple(values["schemas"]),
        classes=tuple(values["classes"]),
        fps=int(values["fps"]),
        past_frames=int(values["past_frames"]),
        future_frames=int(values["future_frames"]),
        core_frames=int(values["core_frames"]),
        channels=tuple(values["channels"]),
        kernel_size=int(values["kernel_size"]),
        dilations=tuple(values["dilations"]),
        dropout=float(values["dropout"]),
        causal=bool(values["causal"]),
        input_shape=values["input_shape"],
        use_norm=values["use_norm"],
        activation=values["activation"],
        kernel_initializer=values["kernel_initializer"],
        use_skip_connections=bool(values["use_skip_connections"]),
        use_gate=bool(values["use_gate"]),
        output_activation=values["output_activation"],
        learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        batch_size=int(values["batch_size"]),
        max_epochs=int(values["max_epochs"]),
        early_stopping_patience=int(values["early_stopping_patience"]),
        seed=int(values["seed"]),
    )
    if config.schemas != (mi.SCHEMA_BALL, mi.SCHEMA_PHASE):
        raise ValueError("P3 must use exactly schema 03 and schema 04")
    if config.classes != ("H", "A", "D"):
        raise ValueError("P3 classes must be H, A, D")
    if not config.causal or config.window_frames != config.receptive_field:
        raise ValueError("P3 requires a causal TCN whose receptive field equals its configured window")
    if len(config.channels) != len(config.dilations):
        raise ValueError("P3 channels and dilations must have equal lengths")
    return config


def feature_columns(config: P3Config, schema: str) -> Tuple[str, ...]:
    """Return one supplied schema unchanged and in its defined order."""
    if schema not in config.schemas:
        raise ValueError("schema {!r} is not configured for P3".format(schema))
    return mi.schema_columns(schema)


def feature_matrix(match: sportec.MatchData, config: P3Config, schema: str) -> Tuple[np.ndarray, np.ndarray]:
    """Build one supplied flat schema through the shared offline input contract."""
    columns = feature_columns(config, schema)
    inputs = sportec.classification_inputs(
        match,
        ci.InputPolicy(schema=schema, temporal_mode=ci.TemporalMode.OFFLINE_ACAUSAL),
    )
    if tuple(inputs.frame_features.columns) != columns:
        raise ValueError("shared P3 input columns differ from the configured supplied schema")
    return inputs.frame_features.to_numpy(dtype=np.float32), inputs.period


@dataclass(frozen=True)
class P3Scaler:
    """Training-only standardisation without adding feature channels."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values_list: Iterable[np.ndarray]) -> "P3Scaler":
        total = None
        squared_total = None
        count = None
        for values in values_list:
            values = np.asarray(values, dtype=np.float64)
            if values.ndim != 2:
                raise ValueError("P3 scaler inputs must have shape [frames, features]")
            valid = np.isfinite(values)
            if total is None:
                total = np.zeros(values.shape[1], dtype=np.float64)
                squared_total = np.zeros(values.shape[1], dtype=np.float64)
                count = np.zeros(values.shape[1], dtype=np.int64)
            if values.shape[1] != len(total):
                raise ValueError("P3 scaler feature widths must match")
            finite = np.where(valid, values, 0.0)
            total += finite.sum(axis=0)
            squared_total += np.square(finite).sum(axis=0)
            count += valid.sum(axis=0)
        if total is None or np.any(count == 0):
            raise ValueError("every P3 feature needs at least one finite training value")
        mean = total / count
        variance = np.maximum(squared_total / count - np.square(mean), 0.0)
        scale = np.sqrt(variance)
        scale[~np.isfinite(scale) | (scale == 0.0)] = 1.0
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.mean):
            raise ValueError("P3 scaler input must match the fitted feature width")
        with np.errstate(invalid="ignore"):
            standardized = (values - self.mean) / self.scale
        return np.where(np.isfinite(standardized), standardized, 0.0).astype(np.float32)


def target_windows(
    values: np.ndarray,
    periods: np.ndarray,
    targets: Sequence[int],
    past_frames: Optional[int] = None,
    future_frames: Optional[int] = None,
) -> np.ndarray:
    """Build period-safe windows, padding each edge with its nearest same-period frame."""
    values = np.asarray(values, dtype=np.float32)
    periods = np.asarray(periods)
    targets = np.asarray(targets, dtype=int)
    if values.ndim != 2 or periods.shape != (len(values),):
        raise ValueError("P3 windows need values [frames, features] and one period per frame")
    if np.any(targets < 0) or np.any(targets >= len(values)):
        raise ValueError("P3 window targets must be valid frame positions")
    config = load_config()
    past = config.past_frames if past_frames is None else past_frames
    future = config.future_frames if future_frames is None else future_frames
    edges = np.flatnonzero(periods[1:] != periods[:-1]) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(values)]
    windows = np.empty((len(targets), values.shape[1], past + 1 + future), dtype=np.float32)
    for row, target in enumerate(targets):
        run = np.searchsorted(ends, target, side="right")
        positions = np.arange(target - past, target + future + 1)
        positions = np.clip(positions, starts[run], ends[run] - 1)
        windows[row] = values[positions].T
    return windows


def period_core_starts(periods: np.ndarray, core_frames: int) -> Tuple[int, ...]:
    """Return contiguous core starts without allowing a core to span a period."""
    periods = np.asarray(periods)
    if periods.ndim != 1 or core_frames <= 0:
        raise ValueError("P3 chunks need one period per frame and a positive core size")
    edges = np.flatnonzero(periods[1:] != periods[:-1]) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(periods)]
    return tuple(
        core_start
        for start, end in zip(starts, ends)
        for core_start in range(int(start), int(end), core_frames)
    )


def period_chunk(
    values: np.ndarray,
    periods: np.ndarray,
    core_start: int,
    past_frames: Optional[int] = None,
    future_frames: Optional[int] = None,
    core_frames: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Build one fixed-size dense core with only same-period context padding."""
    values = np.asarray(values, dtype=np.float32)
    periods = np.asarray(periods)
    if values.ndim != 2 or periods.shape != (len(values),):
        raise ValueError("P3 chunks need values [frames, features] and one period per frame")
    if core_start < 0 or core_start >= len(values):
        raise ValueError("P3 core start must be a valid frame position")
    config = load_config()
    past = config.past_frames if past_frames is None else past_frames
    future = config.future_frames if future_frames is None else future_frames
    core = config.core_frames if core_frames is None else core_frames
    edges = np.flatnonzero(periods[1:] != periods[:-1]) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(values)]
    run = np.searchsorted(ends, core_start, side="right")
    start, end = starts[run], ends[run]
    positions = np.arange(core_start - past, core_start + core + future)
    positions = np.clip(positions, start, end - 1)
    return values[positions].T, int(min(core, end - core_start))


class P3Model(torch.nn.Module):
    """Thin task wrapper around the maintained pytorch-tcn implementation."""

    def __init__(self, config: P3Config, schema: str):
        super().__init__()
        self.schema = schema
        self.feature_count = len(feature_columns(config, schema))
        self.tcn = TCN(
            num_inputs=self.feature_count,
            num_channels=config.channels,
            kernel_size=config.kernel_size,
            dilations=config.dilations,
            dropout=config.dropout,
            causal=config.causal,
            input_shape=config.input_shape,
            use_norm=config.use_norm,
            activation=config.activation,
            kernel_initializer=config.kernel_initializer,
            use_skip_connections=config.use_skip_connections,
            use_gate=config.use_gate,
            output_projection=len(config.classes),
            output_activation=config.output_activation,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != self.feature_count:
            raise ValueError("P3 model expects [batch, selected-schema features, frames]")
        return self.tcn(values)


def target_logits(logits: torch.Tensor) -> torch.Tensor:
    """Align the causal TCN's final output with the window's target frame."""
    if logits.ndim != 3:
        raise ValueError("P3 logits must have shape [batch, classes, frames]")
    return logits[:, :, -1]


def target_chunk_logits(logits: torch.Tensor, config: P3Config) -> torch.Tensor:
    """Select dense target outputs after their configured future look-ahead."""
    start = config.past_frames + config.future_frames
    stop = start + config.core_frames
    if logits.ndim != 3 or logits.shape[2] < stop:
        raise ValueError("P3 chunk logits do not cover the configured core frames")
    return logits[:, :, start:stop]


def save_artifact(
    directory: str,
    model: P3Model,
    schema: str,
    scaler: P3Scaler,
    switch_penalty: int,
) -> None:
    """Persist the exact model inputs and smoother setting needed for inference."""
    os.makedirs(directory, exist_ok=True)
    torch.save(
        {
            "schema": schema,
            "model_state": model.state_dict(),
            "mean": torch.from_numpy(scaler.mean),
            "scale": torch.from_numpy(scaler.scale),
            "switch_penalty": int(switch_penalty),
        },
        os.path.join(directory, "artifact.pt"),
    )


def load_artifact(directory: str, config: P3Config) -> Tuple[P3Model, str, P3Scaler, int]:
    """Restore a frozen P3 run under the currently versioned configuration."""
    payload = torch.load(os.path.join(directory, "artifact.pt"), map_location="cpu", weights_only=True)
    schema = payload["schema"]
    columns = feature_columns(config, schema)
    scaler = P3Scaler(
        mean=np.asarray(payload["mean"], dtype=np.float32),
        scale=np.asarray(payload["scale"], dtype=np.float32),
    )
    if scaler.mean.shape != (len(columns),) or scaler.scale.shape != (len(columns),):
        raise ValueError("P3 artifact scaler does not match its supplied schema")
    model = P3Model(config, schema)
    model.load_state_dict(payload["model_state"])
    return model, schema, scaler, int(payload["switch_penalty"])


def select_run(results: Sequence[Mapping[str, float]]) -> Mapping[str, float]:
    """Choose the fixed-schema submission candidate by spellF1 then macroF1."""
    if not results:
        raise ValueError("P3 selection requires at least one validation result")
    return max(results, key=lambda result: (result["spellF1"], result["macroF1"]))
