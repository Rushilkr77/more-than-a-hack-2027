"""Consolidated inputs for frame-level Home/Away/Dead classification.

The flat view preserves the interpretable PR 2 ablations.  The object view
preserves anonymous player and ball trajectories for a TCN plus a
permutation-invariant set aggregator.  Neither view reads labels, events,
provider tags, damage metadata, player identities, or match identities.

Coordinates use one fixed physical normalization, not statistics fitted on
any match.  Missing values remain NaN and are accompanied by masks and time
since observation, so model-specific adapters can impute without losing the
missingness signal.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from src.features import model_inputs as flat_inputs


PLAYER_VALUE_CHANNELS = (
    "x_home_norm",
    "y_center_norm",
    "vx_home_ms",
    "vy_ms",
    "speed_ms",
)
BALL_VALUE_CHANNELS = (
    "x_home_norm",
    "y_center_norm",
    "vx_home_ms",
    "vy_ms",
)


class TemporalMode(str, Enum):
    """Information available relative to the classified frame."""

    CAUSAL = "causal"
    OFFLINE_ACAUSAL = "offline_acausal"


@dataclass(frozen=True)
class InputPolicy:
    """Explicit experiment policy; prevents silent causal/acausal mixing."""

    schema: str
    temporal_mode: TemporalMode

    def __post_init__(self) -> None:
        flat_inputs.schema_columns(self.schema)
        if not isinstance(self.temporal_mode, TemporalMode):
            raise ValueError("temporal_mode must be a TemporalMode value")
        if self.temporal_mode == TemporalMode.CAUSAL and self.schema != flat_inputs.SCHEMA_CORE:
            raise ValueError(
                "schema {!r} contains acausal features; causal mode currently supports only {!r}".format(
                    self.schema, flat_inputs.SCHEMA_CORE
                )
            )


@dataclass(frozen=True)
class PlayerSetInputs:
    """Anonymous player-set trajectories; slot order has no semantic meaning."""

    values: np.ndarray
    observed: np.ndarray
    inside_pitch: np.ndarray
    seconds_since_seen_s: np.ndarray
    keeper_proxy: np.ndarray
    team: np.ndarray
    channels: Tuple[str, ...] = PLAYER_VALUE_CHANNELS

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or self.values.shape[2] != len(self.channels):
            raise ValueError("player values must have shape [frames, slots, channels]")
        shape = self.values.shape[:2]
        for name, value in (
            ("observed", self.observed),
            ("inside_pitch", self.inside_pitch),
            ("seconds_since_seen_s", self.seconds_since_seen_s),
            ("keeper_proxy", self.keeper_proxy),
        ):
            if value.shape != shape:
                raise ValueError("player {} must have shape [frames, slots]".format(name))
        if self.team.shape != (shape[1],):
            raise ValueError("player team must have one entry per anonymous slot")


@dataclass(frozen=True)
class BallTrajectoryInputs:
    """Observed ball trajectory with explicit quality and observation ages."""

    values: np.ndarray
    observed: np.ndarray
    inside_pitch: np.ndarray
    seconds_since_seen_s: np.ndarray
    seconds_until_seen_s: Optional[np.ndarray]
    channels: Tuple[str, ...] = BALL_VALUE_CHANNELS

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape[1] != len(self.channels):
            raise ValueError("ball values must have shape [frames, channels]")
        frames = self.values.shape[0]
        for name, value in (
            ("observed", self.observed),
            ("inside_pitch", self.inside_pitch),
            ("seconds_since_seen_s", self.seconds_since_seen_s),
        ):
            if value.shape != (frames,):
                raise ValueError("ball {} must have shape [frames]".format(name))
        if self.seconds_until_seen_s is not None and self.seconds_until_seen_s.shape != (frames,):
            raise ValueError("ball seconds_until_seen_s must have shape [frames]")


@dataclass(frozen=True)
class ObjectInputs:
    """Structured trajectory view shared by set/TCN model candidates."""

    frame_id: np.ndarray
    period: np.ndarray
    players: PlayerSetInputs
    ball: BallTrajectoryInputs
    temporal_mode: TemporalMode


@dataclass(frozen=True)
class ClassificationInputs:
    """One leak-safe hand-off containing flat and structured views."""

    frame_id: np.ndarray
    period: np.ndarray
    frame_features: pd.DataFrame
    players: PlayerSetInputs
    ball: BallTrajectoryInputs
    policy: InputPolicy


def _runs(period: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    edges = np.flatnonzero(np.diff(period) != 0) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(period)]
    return tuple((int(start), int(end)) for start, end in zip(starts, ends) if end > start)


def _validate_source(
    table: pd.DataFrame,
    slots: Mapping[str, np.ndarray],
    pitch: Tuple[float, float],
    fps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if len(pitch) != 2 or min(pitch) <= 0:
        raise ValueError("pitch must be a positive (length, width) pair")
    required_table = {
        "frame", "period", "home_attacks_right", "ball_visible", "ball_x_m", "ball_y_m"
    }
    missing_table = sorted(required_table - set(table.columns))
    if missing_table:
        raise ValueError("frame table is missing required columns: {}".format(", ".join(missing_table)))
    missing_slots = sorted({"x", "y", "s", "team"} - set(slots))
    if missing_slots:
        raise ValueError("slot arrays are missing: {}".format(", ".join(missing_slots)))
    if len(table) == 0:
        raise ValueError("frame table must not be empty")

    x = np.asarray(slots["x"], dtype=float)
    y = np.asarray(slots["y"], dtype=float)
    speed = np.asarray(slots["s"], dtype=float)
    team = np.asarray(slots["team"], dtype=np.int8)
    if x.ndim != 2 or y.shape != x.shape or speed.shape != x.shape or x.shape[0] != len(table):
        raise ValueError("slot x/y/s arrays must have shape [frames, slots]")
    if team.shape != (x.shape[1],) or set(team.tolist()) != {0, 1}:
        raise ValueError("slot team must contain both teams and one 0/1 value per slot")
    return x, y, speed, team


def _elapsed_from_observation(observed: np.ndarray, period: np.ndarray, fps: int) -> np.ndarray:
    out = np.full(observed.shape, np.nan, dtype=np.float32)
    step = 1.0 / float(fps)
    for start, end in _runs(period):
        age = np.zeros(observed.shape[1:] or (1,), dtype=np.float32)
        seen = np.zeros(observed.shape[1:] or (1,), dtype=bool)
        segment = observed[start:end].reshape((end - start,) + age.shape)
        target = out[start:end].reshape((end - start,) + age.shape)
        for row in range(end - start):
            age += step
            now = segment[row]
            age[now] = 0.0
            seen |= now
            target[row][seen] = age[seen]
    return out


def _until_observation(observed: np.ndarray, period: np.ndarray, fps: int) -> np.ndarray:
    reversed_observed = observed[::-1]
    reversed_period = period[::-1]
    return _elapsed_from_observation(reversed_observed, reversed_period, fps)[::-1].copy()


def _backward_velocity(
    first: np.ndarray,
    second: np.ndarray,
    observed: np.ndarray,
    period: np.ndarray,
    fps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    velocity_first = np.full(first.shape, np.nan, dtype=np.float32)
    velocity_second = np.full(second.shape, np.nan, dtype=np.float32)
    same_period = period[1:] == period[:-1]
    if observed.ndim == 2:
        same_period = same_period[:, None]
    valid = observed[1:] & observed[:-1] & same_period
    velocity_first[1:] = np.where(valid, (first[1:] - first[:-1]) * fps, np.nan)
    velocity_second[1:] = np.where(valid, (second[1:] - second[:-1]) * fps, np.nan)
    return velocity_first, velocity_second


def _keeper_proxy(
    home_axis_x: np.ndarray,
    inside_pitch: np.ndarray,
    team: np.ndarray,
    pitch_length: float,
) -> np.ndarray:
    proxy = np.zeros(home_axis_x.shape, dtype=bool)
    for team_value in (0, 1):
        slots = np.flatnonzero(team == team_value)
        advance = home_axis_x[:, slots] if team_value == 0 else pitch_length - home_axis_x[:, slots]
        valid = inside_pitch[:, slots]
        deepest = np.min(np.where(valid, advance, np.inf), axis=1)
        has_player = np.isfinite(deepest)
        is_deepest = valid & np.isclose(advance, deepest[:, None], rtol=0.0, atol=1e-6)
        proxy[:, slots] = is_deepest & has_player[:, None]
    return proxy


def make_object_inputs(
    table: pd.DataFrame,
    slots: Mapping[str, np.ndarray],
    pitch: Tuple[float, float],
    temporal_mode: TemporalMode,
    fps: int = 25,
) -> ObjectInputs:
    """Build anonymous player-set and ball trajectory views.

    The output is a representation contract, not a fitted transform.  Slot
    identities are not exposed.  A model must aggregate player slots with a
    permutation-invariant operation or a documented distance ordering.
    """
    if not isinstance(temporal_mode, TemporalMode):
        raise ValueError("temporal_mode must be a TemporalMode value")
    x, y, delivered_speed, team = _validate_source(table, slots, pitch, fps)
    length, width = pitch
    period = table["period"].to_numpy(dtype=np.int8)
    home_right = table["home_attacks_right"].to_numpy(dtype=bool)

    observed = np.isfinite(x) & np.isfinite(y)
    x = np.where(observed, x, np.nan)
    y = np.where(observed, y, np.nan)
    inside_pitch = observed & (x >= 0) & (x <= length) & (y >= 0) & (y <= width)
    home_axis_x = np.where(home_right[:, None], x, length - x)
    x_normalized = 2.0 * home_axis_x / length - 1.0
    y_normalized = 2.0 * y / width - 1.0
    vx, vy = _backward_velocity(home_axis_x, y, observed, period, fps)
    speed = np.where(
        observed & np.isfinite(delivered_speed) & (delivered_speed >= 0),
        delivered_speed,
        np.nan,
    )
    player_values = np.stack((x_normalized, y_normalized, vx, vy, speed), axis=2).astype(np.float32)
    players = PlayerSetInputs(
        values=player_values,
        observed=observed,
        inside_pitch=inside_pitch,
        seconds_since_seen_s=_elapsed_from_observation(observed, period, fps),
        keeper_proxy=_keeper_proxy(home_axis_x, inside_pitch, team, length),
        team=team.copy(),
    )

    ball_x = table["ball_x_m"].to_numpy(dtype=float)
    ball_y = table["ball_y_m"].to_numpy(dtype=float)
    ball_observed = (
        table["ball_visible"].to_numpy(dtype=bool) & np.isfinite(ball_x) & np.isfinite(ball_y)
    )
    ball_x = np.where(ball_observed, ball_x, np.nan)
    ball_y = np.where(ball_observed, ball_y, np.nan)
    ball_home_axis_x = np.where(home_right, ball_x, length - ball_x)
    ball_x_normalized = 2.0 * ball_home_axis_x / length - 1.0
    ball_y_normalized = 2.0 * ball_y / width - 1.0
    ball_vx, ball_vy = _backward_velocity(
        ball_home_axis_x, ball_y, ball_observed, period, fps
    )
    ball_values = np.column_stack(
        (ball_x_normalized, ball_y_normalized, ball_vx, ball_vy)
    ).astype(np.float32)
    ball = BallTrajectoryInputs(
        values=ball_values,
        observed=ball_observed,
        inside_pitch=(
            ball_observed
            & (ball_x >= 0)
            & (ball_x <= length)
            & (ball_y >= 0)
            & (ball_y <= width)
        ),
        seconds_since_seen_s=_elapsed_from_observation(ball_observed, period, fps),
        seconds_until_seen_s=(
            _until_observation(ball_observed, period, fps)
            if temporal_mode == TemporalMode.OFFLINE_ACAUSAL
            else None
        ),
    )
    return ObjectInputs(
        frame_id=table["frame"].to_numpy(dtype=np.int32),
        period=period,
        players=players,
        ball=ball,
        temporal_mode=temporal_mode,
    )


def make_classification_inputs(
    table: pd.DataFrame,
    slots: Mapping[str, np.ndarray],
    pitch: Tuple[float, float],
    policy: InputPolicy,
    fps: int = 25,
) -> ClassificationInputs:
    """Build the complete, leak-safe hand-off used by classifier candidates."""
    frame_features = flat_inputs.make_inputs(
        table,
        pitch,
        policy.schema,
        slots=slots,
        fps=fps,
    )
    objects = make_object_inputs(
        table,
        slots,
        pitch,
        policy.temporal_mode,
        fps=fps,
    )
    if not frame_features.index.equals(pd.Index(objects.frame_id, name="frame_id")):
        raise ValueError("flat and structured input rows do not align")
    return ClassificationInputs(
        frame_id=objects.frame_id,
        period=objects.period,
        frame_features=frame_features,
        players=objects.players,
        ball=objects.ball,
        policy=policy,
    )
