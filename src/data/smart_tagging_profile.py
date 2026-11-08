"""Profile Smart Tagging intervals against provider-neutral tracking measurements."""

from collections import defaultdict

import numpy as np


HOME, AWAY = 0, 1
PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_WIDTH_M = 40.32

TEAM_METRICS = {
    "team_defensive_line_position_m": "defensive_line_position_m",
    "team_midfield_line_position_m": "midfield_line_position_m",
    "team_forward_line_position_m": "forward_line_position_m",
    "team_width_m": "team_width_m",
    "team_defensive_line_velocity_mps": "defensive_line_velocity_mps",
    "team_midfield_line_velocity_mps": "midfield_line_velocity_mps",
    "team_shape_quality": "shape_quality",
}

METRIC_DEFINITIONS = {
    **{name: "Median of {} over the interval.".format(column) for name, column in TEAM_METRICS.items()},
    "ball_visible_fraction": "Share of interval frames with a usable ball position.",
    "ball_median_advance_m": "Median ball distance from the tagged team's own goal line, metres.",
    "ball_forward_displacement_m": "Last usable ball advance minus first usable ball advance, metres.",
    "ball_attacking_third_fraction": "Share of ball-visible frames with ball in the tagged team's attacking third.",
    "ball_in_opposition_box_fraction": "Share of ball-visible frames with ball in the opponent penalty area.",
    "ball_central_channel_fraction": "Share of ball-visible frames within the central 40 percent of pitch width.",
    "players_in_opposition_box_median": "Median visible tagged-team tracks in the opponent penalty area.",
    "players_in_opposition_box_max": "Maximum visible tagged-team tracks in the opponent penalty area.",
}


def _number(value):
    return None if value is None or not np.isfinite(value) else float(value)


def _median(values):
    values = np.asarray(values, dtype=float)
    return _number(np.nanmedian(values)) if np.isfinite(values).any() else None


def _maximum(values):
    values = np.asarray(values, dtype=float)
    return _number(np.nanmax(values)) if np.isfinite(values).any() else None


def _summary(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(values):
        return {"observed_interval_count": 0, "median": None, "p25": None, "p75": None}
    return {"observed_interval_count": int(len(values)), "median": _number(np.median(values)),
            "p25": _number(np.quantile(values, 0.25)), "p75": _number(np.quantile(values, 0.75))}


def _team_direction(home_attacks_right, team):
    return home_attacks_right if team == HOME else ~home_attacks_right


def _measure(table, slots, start_frame, end_frame, team, pitch_length_m, pitch_width_m):
    """One fixed metric set for any half-open interval; unknown team keeps team-relative metrics null."""
    metrics = {name: None for name in METRIC_DEFINITIONS}
    part = table.iloc[start_frame:end_frame]
    if not len(part):
        return metrics
    if team is None:
        metrics["ball_visible_fraction"] = _number(part["ball_visible"].mean())
        return metrics

    prefix = "home" if team == HOME else "away"
    for metric, column in TEAM_METRICS.items():
        metrics[metric] = _median(part["{}_{}".format(prefix, column)])

    ball_visible = part["ball_visible"].to_numpy(dtype=bool)
    metrics["ball_visible_fraction"] = _number(ball_visible.mean())
    if ball_visible.any():
        ball_x = part["ball_x_m"].to_numpy(dtype=float)
        ball_y = part["ball_y_m"].to_numpy(dtype=float)
        attacks_right = _team_direction(part["home_attacks_right"].to_numpy(dtype=bool), team)
        ball_advance = np.where(attacks_right, ball_x, pitch_length_m - ball_x)
        observed_advance = ball_advance[ball_visible]
        metrics["ball_median_advance_m"] = _median(observed_advance)
        metrics["ball_forward_displacement_m"] = _number(observed_advance[-1] - observed_advance[0])
        metrics["ball_attacking_third_fraction"] = _number((observed_advance >= pitch_length_m * 2 / 3).mean())
        in_box = ((observed_advance >= pitch_length_m - PENALTY_AREA_DEPTH_M) &
                  (ball_y[ball_visible] >= (pitch_width_m - PENALTY_AREA_WIDTH_M) / 2) &
                  (ball_y[ball_visible] <= (pitch_width_m + PENALTY_AREA_WIDTH_M) / 2))
        metrics["ball_in_opposition_box_fraction"] = _number(in_box.mean())
        metrics["ball_central_channel_fraction"] = _number(
            (np.abs(ball_y[ball_visible] - pitch_width_m / 2) <= pitch_width_m * 0.2).mean())

    player_x = slots["x"][start_frame:end_frame, slots["team"] == team]
    player_y = slots["y"][start_frame:end_frame, slots["team"] == team]
    attacks_right = _team_direction(part["home_attacks_right"].to_numpy(dtype=bool), team)
    player_advance = np.where(attacks_right[:, None], player_x, pitch_length_m - player_x)
    in_box = ((player_advance >= pitch_length_m - PENALTY_AREA_DEPTH_M) &
              (player_y >= (pitch_width_m - PENALTY_AREA_WIDTH_M) / 2) &
              (player_y <= (pitch_width_m + PENALTY_AREA_WIDTH_M) / 2))
    players_in_box = in_box.sum(axis=1)
    metrics["players_in_opposition_box_median"] = _median(players_in_box)
    metrics["players_in_opposition_box_max"] = _maximum(players_in_box)
    return metrics


def _interval_bounds(event, frame_count):
    return max(0, int(event["start_frame"])), min(frame_count, int(event["end_frame"]))


def profile(table, slots, events, team_by_name, pitch_length_m, pitch_width_m, frames_per_second):
    """Return a JSON-safe, code-by-code profile of reference Smart Tagging intervals."""
    frame_count = len(table)
    normalized_teams = {name.strip().lower(): team for name, team in team_by_name.items()}
    baseline = {}
    for team, name in ((HOME, "home"), (AWAY, "away")):
        baseline[name] = {}
        for half in (1, 2):
            mask = (table["period"] == half) & table["graded"] & ~table["pre_kickoff"]
            frame_numbers = table.loc[mask, "frame"].to_numpy(dtype=int)
            if len(frame_numbers):
                baseline[name][str(half)] = _measure(
                    table, slots, int(frame_numbers[0]), int(frame_numbers[-1]) + 1, team,
                    pitch_length_m, pitch_width_m,
                )

    grouped = defaultdict(list)
    for source_index, event in enumerate(events, start=1):
        start, end = _interval_bounds(event, frame_count)
        if end <= start:
            continue
        team = normalized_teams.get((event.get("tags", {}).get("Team") or "").strip().lower())
        half = int(table.iloc[start]["period"]) if team is not None else None
        metrics = _measure(table, slots, start, end, team, pitch_length_m, pitch_width_m)
        reference = baseline.get("home" if team == HOME else "away", {}).get(str(half), {}) if team is not None else {}
        delta = {name: _number(value - reference[name]) if value is not None and reference.get(name) is not None else None
                 for name, value in metrics.items()}
        grouped[event["code"]].append({
            "source_index": source_index,
            "team": event.get("tags", {}).get("Team"),
            "half": event.get("tags", {}).get("Half"),
            "start_s": _number(event.get("start_s")), "end_s": _number(event.get("end_s")),
            "start_frame": start, "end_frame": end,
            "duration_s": _number((end - start) / frames_per_second),
            "metrics": metrics, "delta_from_team_half_baseline": delta,
        })

    codes = {}
    for code, intervals in sorted(grouped.items()):
        metric_summary = {name: _summary([row["metrics"][name] for row in intervals]) for name in METRIC_DEFINITIONS}
        baseline_delta = {name: _summary([row["delta_from_team_half_baseline"][name] for row in intervals])
                          for name in METRIC_DEFINITIONS}
        codes[code] = {
            "summary": {
                "interval_count": len(intervals),
                "team_known_interval_count": sum(row["team"] is not None and row["team"].strip().lower() in normalized_teams
                                                 for row in intervals),
                "duration_s": _summary([row["duration_s"] for row in intervals]),
                "metrics": metric_summary,
                "baseline_delta": baseline_delta,
            },
            "sportec_rule_template": {
                "method": "Use these interquartile ranges as candidate evidence, not as a production rule. "
                          "Require multiple available signals and lower confidence when coverage is poor.",
                "team_shape_relative_to_half_iqr": {name: baseline_delta[name] for name in TEAM_METRICS},
                "ball_box_iqr": {name: metric_summary[name] for name in METRIC_DEFINITIONS if name not in TEAM_METRICS},
            },
            "intervals": intervals,
        }
    return {
        "schema_version": 1,
        "metric_definitions": METRIC_DEFINITIONS,
        "method": {"intervals": "Smart Tagging half-open frame intervals", "baseline": "same team, same half, graded non-pre-kickoff frames",
                   "penalty_area": {"depth_m": PENALTY_AREA_DEPTH_M, "width_m": PENALTY_AREA_WIDTH_M}},
        "team_half_baseline": baseline,
        "codes": codes,
    }
