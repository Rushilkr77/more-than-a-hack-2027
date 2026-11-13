"""Leak-safe, architecture-agnostic model inputs.

The four schemas preserve the experiments designed in PR 2 while using the
merged frame-table contract.  ``make_inputs`` is a pure function of tracking
features and optional anonymous slot arrays.  It never reads labels, events,
damage metadata, or provider Smart Tagging.

The returned DataFrame uses ``frame_id`` as its index, not as a model feature.
Writing it with ``to_csv(..., index=True)`` produces the standalone CSV layout
from the original designs without allowing a trainer to accidentally fit on
absolute frame numbers.
"""

from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


SCHEMA_CORE = "01_core_shape"
SCHEMA_DYNAMICS = "02_shape_dynamics"
SCHEMA_BALL = "03_ball_shape_extended"
SCHEMA_PHASE = "04_full_phase_context"
SCHEMA_NAMES = (SCHEMA_CORE, SCHEMA_DYNAMICS, SCHEMA_BALL, SCHEMA_PHASE)

CORE_COLUMNS = (
    "period", "home_attacks_right", "n_home", "n_away",
    "home_shape_conf", "away_shape_conf", "home_shape_missing", "away_shape_missing",
    "home_dline_m", "home_cen_m", "home_fwd_m", "home_width_m", "home_depth_m", "home_speed_ms",
    "away_dline_m", "away_cen_m", "away_fwd_m", "away_width_m", "away_depth_m", "away_speed_ms",
)

DYNAMIC_COLUMNS = (
    "home_line_gap_change_3s", "away_line_gap_change_3s",
    "home_cen_change_3s", "away_cen_change_3s",
    "home_width_change_3s", "away_width_change_3s",
    "home_speed_change_3s", "away_speed_change_3s",
    "home_dline_fwd_max_cross_corr_5s", "away_dline_fwd_max_cross_corr_5s",
    "home_dline_fwd_lag_at_max_corr_s", "away_dline_fwd_lag_at_max_corr_s",
)

BALL_COLUMNS = (
    "ball_visible", "ball_x_m", "ball_y_m", "ball_home_adv_m",
    "home_ball_dist_m", "away_ball_dist_m", "ball_margin_m",
    "home_cen_to_ball_offset_m", "away_cen_to_ball_offset_m",
    "seconds_since_ball_seen", "seconds_until_ball_seen",
    "left_anchor_margin_m", "right_anchor_margin_m", "anchor_proxy_agrees", "gap_length_s",
    "home_tracks_ahead_ball_count", "home_tracks_behind_ball_count",
    "away_tracks_ahead_ball_count", "away_tracks_behind_ball_count",
    "home_local_support_count_5m", "away_local_support_count_5m",
    "home_compactness_behind_ball_m", "away_compactness_behind_ball_m",
)

# PR 5 arrived after the original schema headers.  These tracking-only signals
# belong in the ball-aware schemas, where they provide the evidence for D.
DEAD_BALL_COLUMNS = (
    "ball_n_obs_1s", "ball_range_back_1s_m", "ball_range_fwd_1s_m", "ball_speed_p90_ms",
    "ball_dist_touchline_m", "ball_dist_goalline_m", "ball_dist_corner_m",
    "ball_dist_goal_m", "ball_dist_centre_m", "pl_slow_share", "pl_fast_share",
    "pl_in_box_max_n", "pl_still_at_line_n", "home_own_half_share", "away_own_half_share",
    "dead_ball_conf", "dead_player_conf",
)

# Provider tag_* columns are deliberately absent.  These are generated from
# the same damaged tracking inputs on Sportec and BAR-ATH.
GENERATED_TAG_COLUMNS = (
    "gtag_poss", "gtag_phase", "gtag_transition", "gtag_long_ball",
    "gtag_final_third", "gtag_in_box", "gtag_players_in_box_n", "gtag_setpiece",
)

SCHEMA_COLUMNS = {
    SCHEMA_CORE: CORE_COLUMNS,
    SCHEMA_DYNAMICS: CORE_COLUMNS + DYNAMIC_COLUMNS,
    SCHEMA_BALL: CORE_COLUMNS + BALL_COLUMNS + DEAD_BALL_COLUMNS,
    SCHEMA_PHASE: CORE_COLUMNS + DYNAMIC_COLUMNS + BALL_COLUMNS + DEAD_BALL_COLUMNS + GENERATED_TAG_COLUMNS,
}


def schema_columns(schema: str) -> Tuple[str, ...]:
    """Return the explicit feature allow-list for one schema."""
    try:
        return SCHEMA_COLUMNS[schema]
    except KeyError as err:
        raise ValueError("unknown input schema {!r}; expected one of {}".format(schema, SCHEMA_NAMES)) from err


def _require_columns(table: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError("frame table is missing required columns: {}".format(", ".join(missing)))


def _runs(period: np.ndarray) -> List[Tuple[int, int]]:
    edges = np.flatnonzero(np.diff(period) != 0) + 1
    return list(zip(np.r_[0, edges].tolist(), np.r_[edges, len(period)].tolist()))


def _change_within_runs(values: np.ndarray, period: np.ndarray, frames: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float32)
    for start, end in _runs(period):
        if end - start <= frames:
            continue
        current = values[start + frames:end]
        previous = values[start:end - frames]
        valid = np.isfinite(current) & np.isfinite(previous)
        out[start + frames:end] = np.where(valid, current - previous, np.nan)
    return out


def _line_coordination(
    defensive_line: np.ndarray,
    forward_line: np.ndarray,
    period: np.ndarray,
    fps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trailing five-second line-motion correlation and its best +/-1 s lag.

    The correlation uses first differences rather than absolute line heights.
    Positive lag means the forward-line movement follows the defensive-line
    movement.  Every rolling calculation is isolated to one contiguous period.
    """
    n = len(defensive_line)
    best_corr = np.full(n, np.nan, dtype=np.float32)
    best_lag = np.full(n, np.nan, dtype=np.float32)
    window = max(2, int(round(5 * fps)))
    min_periods = max(2, int(round(window * 0.6)))
    step = max(1, int(round(0.2 * fps)))
    lag_frames = np.arange(-fps, fps + 1, step, dtype=int)

    for start, end in _runs(period):
        dline_motion = pd.Series(defensive_line[start:end], dtype="float64").diff()
        fwd_motion = pd.Series(forward_line[start:end], dtype="float64").diff()
        correlations = []
        for lag in lag_frames:
            # Positive lag: compare dline(t) with fwd(t + lag).
            correlations.append(
                dline_motion.rolling(window, min_periods=min_periods).corr(fwd_motion.shift(-int(lag))).to_numpy()
            )
        matrix = np.column_stack(correlations)
        valid = np.isfinite(matrix).any(axis=1)
        if not valid.any():
            continue
        safe = np.where(np.isfinite(matrix), matrix, -np.inf)
        choice = np.argmax(safe, axis=1)
        rows = np.flatnonzero(valid)
        best_corr[start + rows] = matrix[rows, choice[rows]].astype(np.float32)
        best_lag[start + rows] = (lag_frames[choice[rows]] / float(fps)).astype(np.float32)
    return best_corr, best_lag


def _dynamic_features(table: pd.DataFrame, fps: int) -> Dict[str, np.ndarray]:
    period = table["period"].to_numpy()
    frames = int(round(3 * fps))
    out: Dict[str, np.ndarray] = {}
    for team in ("home", "away"):
        for source, output in (
            ("line_gap_m", "line_gap_change_3s"),
            ("cen_m", "cen_change_3s"),
            ("width_m", "width_change_3s"),
            ("speed_ms", "speed_change_3s"),
        ):
            out["{}_{}".format(team, output)] = _change_within_runs(
                table["{}_{}".format(team, source)].to_numpy(dtype=float), period, frames
            )
        corr, lag = _line_coordination(
            table["{}_dline_m".format(team)].to_numpy(dtype=float),
            table["{}_fwd_m".format(team)].to_numpy(dtype=float),
            period,
            fps,
        )
        out["{}_dline_fwd_max_cross_corr_5s".format(team)] = corr
        out["{}_dline_fwd_lag_at_max_corr_s".format(team)] = lag
    return out


def _gap_features(table: pd.DataFrame, fps: int) -> Dict[str, np.ndarray]:
    visible = table["ball_visible"].to_numpy(dtype=bool)
    period = table["period"].to_numpy()
    margin = table["ball_margin_m"].to_numpy(dtype=float)
    nearest = table["nearest_team"].to_numpy(dtype=int)
    n = len(table)
    since = np.full(n, np.nan, dtype=np.float32)
    until = np.full(n, np.nan, dtype=np.float32)
    left_margin = np.full(n, np.nan, dtype=np.float32)
    right_margin = np.full(n, np.nan, dtype=np.float32)
    agrees = np.full(n, np.nan, dtype=np.float32)
    gap_length = np.zeros(n, dtype=np.float32)

    for start, end in _runs(period):
        local_visible = visible[start:end]
        rows = np.arange(start, end)
        previous = np.maximum.accumulate(np.where(local_visible, rows, -1))
        following = np.minimum.accumulate(np.where(local_visible, rows, n)[::-1])[::-1]
        valid_previous = previous >= start
        valid_following = following < end
        since[rows[valid_previous]] = (rows[valid_previous] - previous[valid_previous]) / float(fps)
        until[rows[valid_following]] = (following[valid_following] - rows[valid_following]) / float(fps)

        changes = np.diff(np.r_[True, local_visible, True].astype(np.int8))
        gap_starts = np.flatnonzero(changes == -1) + start
        gap_ends = np.flatnonzero(changes == 1) + start
        for gap_start, gap_end in zip(gap_starts, gap_ends):
            gap_length[gap_start:gap_end] = (gap_end - gap_start) / float(fps)
            left = gap_start - 1 if gap_start > start and visible[gap_start - 1] else None
            right = gap_end if gap_end < end and visible[gap_end] else None
            if left is not None:
                left_margin[gap_start:gap_end] = margin[left]
            if right is not None:
                right_margin[gap_start:gap_end] = margin[right]
            if left is not None and right is not None and nearest[left] >= 0 and nearest[right] >= 0:
                agrees[gap_start:gap_end] = float(nearest[left] == nearest[right])

    return {
        "seconds_since_ball_seen": since,
        "seconds_until_ball_seen": until,
        "left_anchor_margin_m": left_margin,
        "right_anchor_margin_m": right_margin,
        "anchor_proxy_agrees": agrees,
        "gap_length_s": gap_length,
    }


def _validate_slots(slots: Optional[Mapping[str, np.ndarray]], n_frames: int) -> Mapping[str, np.ndarray]:
    if slots is None:
        raise ValueError("schemas 03 and 04 require anonymous tracking slot arrays")
    missing = sorted({"x", "y", "team"} - set(slots))
    if missing:
        raise ValueError("slot arrays are missing: {}".format(", ".join(missing)))
    x, y, team = np.asarray(slots["x"]), np.asarray(slots["y"]), np.asarray(slots["team"])
    if x.ndim != 2 or y.shape != x.shape or x.shape[0] != n_frames:
        raise ValueError("slot x/y arrays must have shape [frames, slots]")
    if team.shape != (x.shape[1],):
        raise ValueError("slot team array must have one entry per slot")
    return slots


def _spatial_ball_features(
    table: pd.DataFrame,
    slots: Mapping[str, np.ndarray],
    pitch: Tuple[float, float],
) -> Dict[str, np.ndarray]:
    length, width = pitch
    x = np.asarray(slots["x"], dtype=float)
    y = np.asarray(slots["y"], dtype=float)
    slot_team = np.asarray(slots["team"], dtype=int)
    ball_x = table["ball_x_m"].to_numpy(dtype=float)
    ball_y = table["ball_y_m"].to_numpy(dtype=float)
    home_right = table["home_attacks_right"].to_numpy(dtype=bool)
    ball_visible = table["ball_visible"].to_numpy(dtype=bool) & np.isfinite(ball_x) & np.isfinite(ball_y)
    on_pitch = np.isfinite(x) & np.isfinite(y) & (x >= -1) & (x <= length + 1) & (y >= -1) & (y <= width + 1)
    out: Dict[str, np.ndarray] = {}

    for team, name in ((0, "home"), (1, "away")):
        mask = slot_team == team
        px, py = x[:, mask], y[:, mask]
        valid = on_pitch[:, mask]
        attacks_right = home_right if team == 0 else ~home_right
        player_advance = np.where(attacks_right[:, None], px, length - px)
        ball_advance = np.where(attacks_right, ball_x, length - ball_x)
        ahead = valid & ball_visible[:, None] & (player_advance > ball_advance[:, None])
        behind = valid & ball_visible[:, None] & (player_advance <= ball_advance[:, None])
        out["{}_tracks_ahead_ball_count".format(name)] = np.where(
            ball_visible, ahead.sum(axis=1), np.nan
        ).astype(np.float32)
        out["{}_tracks_behind_ball_count".format(name)] = np.where(
            ball_visible, behind.sum(axis=1), np.nan
        ).astype(np.float32)
        distance = np.hypot(px - ball_x[:, None], py - ball_y[:, None])
        support = valid & ball_visible[:, None] & (distance <= 5.0)
        out["{}_local_support_count_5m".format(name)] = np.where(
            ball_visible, support.sum(axis=1), np.nan
        ).astype(np.float32)

        behind_positions = np.where(behind, player_advance, np.nan)
        count = np.isfinite(behind_positions).sum(axis=1)
        compactness = np.full(len(table), np.nan, dtype=float)
        enough = count >= 2
        means = np.zeros(len(table), dtype=float)
        means[enough] = np.nansum(behind_positions[enough], axis=1) / count[enough]
        centred = behind_positions[enough] - means[enough, None]
        compactness[enough] = np.sqrt(np.nansum(centred * centred, axis=1) / count[enough])
        out["{}_compactness_behind_ball_m".format(name)] = np.where(
            ball_visible & enough, compactness, np.nan
        ).astype(np.float32)
    return out


def _ball_features(
    table: pd.DataFrame,
    slots: Mapping[str, np.ndarray],
    pitch: Tuple[float, float],
    fps: int,
) -> Dict[str, np.ndarray]:
    length, _ = pitch
    out = _gap_features(table, fps)
    out.update(_spatial_ball_features(table, slots, pitch))
    ball_home = table["ball_home_adv_m"].to_numpy(dtype=float)
    out["home_cen_to_ball_offset_m"] = (
        ball_home - table["home_cen_m"].to_numpy(dtype=float)
    ).astype(np.float32)
    out["away_cen_to_ball_offset_m"] = (
        (length - ball_home) - table["away_cen_m"].to_numpy(dtype=float)
    ).astype(np.float32)
    return out


def make_inputs(
    table: pd.DataFrame,
    pitch: Tuple[float, float],
    schema: str,
    slots: Optional[Mapping[str, np.ndarray]] = None,
    fps: int = 25,
) -> pd.DataFrame:
    """Build one standalone, leak-safe feature schema.

    ``slots`` is required only for schemas 03/04 because their designed local
    support and ahead/behind-ball fields cannot be recovered from aggregate
    frame-table columns.  Both Sportec and BAR-ATH expose the same anonymous
    slot-array contract.
    """
    columns = schema_columns(schema)
    if fps <= 0:
        raise ValueError("fps must be positive")
    if len(pitch) != 2 or min(pitch) <= 0:
        raise ValueError("pitch must be a positive (length, width) pair")

    required = {
        "frame", "period", "home_attacks_right", "n_home", "n_away",
        "home_shape_conf", "away_shape_conf",
        "home_dline_m", "home_cen_m", "home_fwd_m", "home_line_gap_m",
        "home_width_m", "home_depth_m", "home_speed_ms",
        "away_dline_m", "away_cen_m", "away_fwd_m", "away_line_gap_m",
        "away_width_m", "away_depth_m", "away_speed_ms",
    }
    if schema in (SCHEMA_BALL, SCHEMA_PHASE):
        required.update({
            "ball_visible", "ball_x_m", "ball_y_m", "ball_home_adv_m",
            "home_ball_dist_m", "away_ball_dist_m", "ball_margin_m", "nearest_team",
            *DEAD_BALL_COLUMNS,
        })
    if schema == SCHEMA_PHASE:
        required.update(GENERATED_TAG_COLUMNS)
    _require_columns(table, required)

    result = pd.DataFrame(index=pd.Index(table["frame"].to_numpy(dtype=np.int32), name="frame_id"))
    for column in CORE_COLUMNS:
        if column == "home_shape_missing":
            result[column] = (table["home_shape_conf"].to_numpy() <= 0) | table["home_dline_m"].isna().to_numpy()
        elif column == "away_shape_missing":
            result[column] = (table["away_shape_conf"].to_numpy() <= 0) | table["away_dline_m"].isna().to_numpy()
        else:
            result[column] = table[column].to_numpy()

    if schema in (SCHEMA_DYNAMICS, SCHEMA_PHASE):
        for name, values in _dynamic_features(table, fps).items():
            result[name] = values

    if schema in (SCHEMA_BALL, SCHEMA_PHASE):
        valid_slots = _validate_slots(slots, len(table))
        for name in (
            "ball_visible", "ball_x_m", "ball_y_m", "ball_home_adv_m",
            "home_ball_dist_m", "away_ball_dist_m", "ball_margin_m",
            *DEAD_BALL_COLUMNS,
        ):
            result[name] = table[name].to_numpy()
        for name, values in _ball_features(table, valid_slots, pitch, fps).items():
            result[name] = values

    if schema == SCHEMA_PHASE:
        for name in GENERATED_TAG_COLUMNS:
            result[name] = table[name].to_numpy()

    # Select from the immutable allow-list so future frame-table columns cannot
    # silently become training features.
    return result.loc[:, list(columns)]


def standalone_frame(inputs: pd.DataFrame) -> pd.DataFrame:
    """Return the CSV-friendly form with frame_id restored as a column."""
    if inputs.index.name != "frame_id":
        raise ValueError("input frame index must be named frame_id")
    return inputs.reset_index()
