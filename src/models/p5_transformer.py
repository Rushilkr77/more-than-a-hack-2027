"""P5: low-capacity local-global Transformer over common flat inputs.

The model combines one shallow local temporal convolution with global
self-attention.  It consumes period-safe windows of the same standardised flat
features and explicit observation flags used by P4, then emits H/A/D logits for
every token.  The competition is offline, so attention is intentionally
acausal; no window may cross a period boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F


CLASSES: Tuple[str, ...] = ("H", "A", "D")
IGNORE = -100
ALIGNMENT_ONLY_COLUMNS = frozenset(
    (
        "frame",
        "frame_id",
        "match",
        "match_id",
        "competition",
        "competition_id",
        "provider",
        "provider_id",
        "team",
        "team_id",
        "track",
        "track_id",
        "slot",
        "slot_id",
    )
)


@dataclass(frozen=True)
class Scaler:
    """Training-only statistics and the columns exposed to P5."""

    columns: Tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    masked: Tuple[str, ...]


@dataclass(frozen=True)
class SequenceExample:
    """One continuous match/half collection at the selected frame step."""

    values: np.ndarray
    labels: np.ndarray
    period: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("values must have shape [frames, features]")
        if self.labels.ndim != 1 or self.period.ndim != 1:
            raise ValueError("labels and period must be one-dimensional")
        if not (len(self.values) == len(self.labels) == len(self.period)):
            raise ValueError("values, labels, and period must have equal frame counts")
        if len(self.values) == 0:
            raise ValueError("a sequence example cannot be empty")


@dataclass(frozen=True)
class TransformerConfig:
    """Fixed architecture boundary for the P5 local-global encoder."""

    input_features: int
    model_width: int = 64
    heads: int = 4
    layers: int = 2
    feedforward_width: int = 128
    dropout: float = 0.3
    local_kernel: int = 5
    max_tokens: int = 512

    def __post_init__(self) -> None:
        integer_fields = {
            "input_features": self.input_features,
            "model_width": self.model_width,
            "heads": self.heads,
            "layers": self.layers,
            "feedforward_width": self.feedforward_width,
            "local_kernel": self.local_kernel,
            "max_tokens": self.max_tokens,
        }
        invalid = [name for name, value in integer_fields.items() if value <= 0]
        if invalid:
            raise ValueError("positive values required for: {}".format(", ".join(invalid)))
        if self.model_width % self.heads != 0:
            raise ValueError("model_width must be divisible by heads")
        if self.local_kernel % 2 == 0:
            raise ValueError("local_kernel must be odd so frame alignment is preserved")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class TrainingSummary:
    """Minimal deterministic evidence returned by a synthetic or real fit."""

    seed: int
    epochs: int
    validation_cross_entropy: float
    parameters: int


def fit_scaler(frame: pd.DataFrame) -> Scaler:
    """Fit fixed columns and statistics on training matches only."""

    forbidden = sorted(ALIGNMENT_ONLY_COLUMNS.intersection(frame.columns))
    if forbidden:
        raise ValueError(
            "alignment-only columns cannot be model inputs: {}".format(", ".join(forbidden))
        )
    numeric = frame.astype("float64")
    columns = tuple(column for column in numeric.columns if numeric[column].notna().any())
    if not columns:
        raise ValueError("P5 cannot train when every selected feature is missing")
    selected = numeric.loc[:, list(columns)]
    mean = selected.mean().to_numpy(dtype="float64")
    std = selected.std().replace(0, 1).fillna(1).to_numpy(dtype="float64")
    masked = tuple(column for column in columns if selected[column].isna().any())
    return Scaler(columns=columns, mean=mean, std=std, masked=masked)


def transform(frame: pd.DataFrame, scaler: Scaler) -> np.ndarray:
    """Return finite float32 features followed by explicit observation flags."""

    missing = [column for column in scaler.columns if column not in frame.columns]
    if missing:
        raise ValueError("input is missing fitted columns: {}".format(", ".join(missing)))
    raw = frame.loc[:, list(scaler.columns)].to_numpy(dtype="float64")
    values = (raw - scaler.mean) / scaler.std
    if scaler.masked:
        observed = frame.loc[:, list(scaler.masked)].notna().to_numpy(dtype="float32")
    else:
        observed = np.empty((len(frame), 0), dtype="float32")
    return np.concatenate(
        (np.nan_to_num(values, nan=0.0).astype("float32"), observed),
        axis=1,
    )


def encode(labels: Iterable[str]) -> np.ndarray:
    """Encode H/A/D and retain unknown/ungraded labels as IGNORE."""

    codes = {label: index for index, label in enumerate(CLASSES)}
    return np.asarray([codes.get(label, IGNORE) for label in labels], dtype=np.int64)


def windows(period: np.ndarray, length: int, stride: int) -> List[Tuple[int, int]]:
    """Build fixed windows that never cross a period boundary."""

    if length <= 0 or stride <= 0:
        raise ValueError("window length and stride must be positive")
    periods = np.asarray(period)
    if periods.ndim != 1:
        raise ValueError("period must be one-dimensional")
    if len(periods) == 0:
        return []

    edges = np.flatnonzero(np.diff(periods) != 0) + 1
    spans: List[Tuple[int, int]] = []
    for start, end in zip(np.r_[0, edges], np.r_[edges, len(periods)]):
        start_i, end_i = int(start), int(end)
        if end_i - start_i <= length:
            spans.append((start_i, end_i))
        else:
            starts = list(range(start_i, end_i - length, stride)) + [end_i - length]
            spans.extend((window_start, window_start + length) for window_start in starts)
    return spans


def _sinusoidal_positions(max_tokens: int, width: int) -> Tensor:
    positions = torch.arange(max_tokens, dtype=torch.float32).unsqueeze(1)
    even_dimensions = torch.arange(0, width, 2, dtype=torch.float32)
    frequencies = torch.exp(even_dimensions * (-math.log(10_000.0) / width))
    encoding = torch.zeros(max_tokens, width, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: width // 2])
    return encoding


class TemporalTransformer(nn.Module):
    """Shallow local temporal residual followed by global self-attention."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_features, config.model_width)
        padding = config.local_kernel // 2
        self.local_depthwise = nn.Conv1d(
            config.model_width,
            config.model_width,
            kernel_size=config.local_kernel,
            padding=padding,
            groups=config.model_width,
        )
        self.local_pointwise = nn.Conv1d(config.model_width, config.model_width, kernel_size=1)
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_width,
            nhead=config.heads,
            dim_feedforward=config.feedforward_width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.layers,
            norm=nn.LayerNorm(config.model_width),
        )
        self.output = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.model_width, len(CLASSES)),
        )
        self.register_buffer(
            "position_encoding",
            _sinusoidal_positions(config.max_tokens, config.model_width),
            persistent=False,
        )

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3:
            raise ValueError("values must have shape [batch, frames, features]")
        if values.shape[-1] != self.config.input_features:
            raise ValueError(
                "expected {} input features, received {}".format(
                    self.config.input_features, values.shape[-1]
                )
            )
        if values.shape[1] > self.config.max_tokens:
            raise ValueError(
                "sequence has {} tokens but max_tokens is {}".format(
                    values.shape[1], self.config.max_tokens
                )
            )

        tokens = self.input_projection(values)
        local = self.local_depthwise(tokens.transpose(1, 2))
        local = self.local_pointwise(F.gelu(local)).transpose(1, 2)
        positions = self.position_encoding[: values.shape[1]].to(
            device=values.device, dtype=values.dtype
        )
        encoded = self.encoder(tokens + local + positions.unsqueeze(0))
        return self.output(encoded)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def equal_probability_blend(*members: np.ndarray) -> np.ndarray:
    """Average compatible H/A/D probability matrices with no fitted weight."""

    if len(members) < 2:
        raise ValueError("at least two probability members are required")
    arrays = [np.asarray(member, dtype=np.float32) for member in members]
    expected = arrays[0].shape
    if len(expected) != 2 or expected[1] != len(CLASSES):
        raise ValueError("probabilities must have shape [frames, 3]")
    if any(array.shape != expected for array in arrays[1:]):
        raise ValueError("probability members must have identical shapes")
    if any(not np.isfinite(array).all() or (array < 0).any() for array in arrays):
        raise ValueError("probability members must be finite and non-negative")
    if any(not np.allclose(array.sum(axis=1), 1.0, atol=1e-4) for array in arrays):
        raise ValueError("each probability row must sum to one")
    return np.mean(arrays, axis=0, dtype=np.float32)


Span = Tuple[int, Tuple[int, int]]


def _spans(examples: Sequence[SequenceExample], length: int, stride: int) -> List[Span]:
    return [
        (index, span)
        for index, example in enumerate(examples)
        for span in windows(example.period, length, stride)
    ]


def _batches(
    examples: Sequence[SequenceExample], spans: Sequence[Span], batch_size: int
) -> Iterator[Tuple[Tensor, Tensor]]:
    by_length: Dict[int, List[Tuple[int, int, int]]] = {}
    for index, (start, end) in spans:
        by_length.setdefault(end - start, []).append((index, start, end))
    for items in by_length.values():
        for offset in range(0, len(items), batch_size):
            chunk = items[offset : offset + batch_size]
            x = torch.from_numpy(
                np.stack([examples[index].values[start:end] for index, start, end in chunk])
            )
            y = torch.from_numpy(
                np.stack([examples[index].labels[start:end] for index, start, end in chunk])
            )
            yield x, y


def train(
    examples: Sequence[SequenceExample],
    validation_examples: Sequence[SequenceExample],
    *,
    config: TransformerConfig,
    length: int,
    seed: int = 0,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 40,
    patience: int = 5,
    log: Callable[[str], None] = print,
) -> Tuple[TemporalTransformer, TrainingSummary]:
    """Fit deterministically and early-stop on raw validation cross-entropy."""

    if not examples or not validation_examples:
        raise ValueError("training and validation examples are required")
    if batch_size <= 0 or max_epochs <= 0 or patience <= 0:
        raise ValueError("batch_size, max_epochs, and patience must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    feature_counts = {example.values.shape[1] for example in (*examples, *validation_examples)}
    if feature_counts != {config.input_features}:
        raise ValueError("all examples must match config.input_features")
    if length > config.max_tokens:
        raise ValueError("length cannot exceed config.max_tokens")

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    model = TemporalTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE)
    train_spans = _spans(examples, length, max(1, length // 2))
    validation_spans = _spans(validation_examples, length, length)
    if not train_spans or not validation_spans:
        raise ValueError("period-safe windowing produced no examples")

    best_loss = math.inf
    best_state: Dict[str, Tensor] = {}
    stale_epochs = 0
    epochs_run = 0
    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        order = [train_spans[index] for index in rng.permutation(len(train_spans))]
        for values, labels in _batches(examples, order, batch_size):
            optimizer.zero_grad()
            logits = model(values)
            loss = loss_function(logits.reshape(-1, len(CLASSES)), labels.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        total_loss = 0.0
        graded = 0
        with torch.no_grad():
            for values, labels in _batches(
                validation_examples, validation_spans, batch_size
            ):
                count = int((labels != IGNORE).sum())
                if count == 0:
                    continue
                loss = loss_function(
                    model(values).reshape(-1, len(CLASSES)), labels.reshape(-1)
                )
                total_loss += float(loss) * count
                graded += count
        if graded == 0:
            raise ValueError("validation examples contain no graded labels")
        validation_loss = total_loss / graded
        log("epoch {:2d} val_ce {:.6f}".format(epoch, validation_loss))
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            stale_epochs = 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if not best_state:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, TrainingSummary(
        seed=seed,
        epochs=epochs_run,
        validation_cross_entropy=best_loss,
        parameters=parameter_count(model),
    )


def predict_proba(
    model: TemporalTransformer,
    values: np.ndarray,
    period: np.ndarray,
    *,
    length: int,
    step: int,
) -> np.ndarray:
    """Stitch centre-preferred overlapping windows into full-frame probabilities."""

    features = np.asarray(values, dtype=np.float32)
    periods = np.asarray(period)
    if features.ndim != 2 or periods.ndim != 1 or len(features) != len(periods):
        raise ValueError("values and period must align on one frame axis")
    if step <= 0:
        raise ValueError("step must be positive")
    if len(features) == 0:
        return np.empty((0, len(CLASSES)), dtype=np.float32)

    output = np.zeros((len(features), len(CLASSES)), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for offset in range(min(step, len(features))):
            sampled_values = features[offset::step]
            sampled_periods = periods[offset::step]
            probabilities = np.zeros(
                (len(sampled_values), len(CLASSES)), dtype=np.float32
            )
            distance = np.full(len(sampled_values), np.inf)
            for start, end in windows(
                sampled_periods, length=length, stride=max(1, length // 2)
            ):
                logits = model(torch.from_numpy(sampled_values[start:end])[None])
                candidate = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                centre_distance = np.abs(
                    np.arange(start, end) - (start + end - 1) / 2.0
                )
                better = centre_distance < distance[start:end]
                probabilities[start:end][better] = candidate[better]
                distance[start:end][better] = centre_distance[better]
            if not np.isfinite(distance).all():
                raise RuntimeError("window stitching left frames without predictions")
            output[offset::step] = probabilities

    return output
