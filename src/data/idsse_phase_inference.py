"""Derive reproducible tactical-phase inputs from IDSSE tracking and event feeds."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from src.data.idsse_smart_tagging import Annotation
from src.tracking import HOME, AWAY, TrackingFeatureEngine, inferred_confidence


MINIMUM_PHASE_SECONDS = 5.0


def attack_directions_from_events(events_path, home_team_id, away_team_id):
    """Return (home, away) attack directions per half from IDSSE kick-off sides.

    Sportec pitch x increases from the physical left goal toward the right goal.
    A team on the left at kick-off therefore attacks toward increasing x.
    """
    directions = {}
    for event in ET.parse(Path(events_path)).getroot().findall("./Event"):
        if not len(event) or event[0].tag != "KickOff":
            continue
        kickoff = event[0]
        section = kickoff.get("GameSection")
        left, right = kickoff.get("TeamLeft"), kickoff.get("TeamRight")
        if section not in {"firstHalf", "secondHalf"} or {left, right} != {home_team_id, away_team_id}:
            continue
        directions[section] = (left == home_team_id, left == away_team_id)
    missing = {"firstHalf", "secondHalf"} - set(directions)
    if missing:
        raise ValueError("IDSSE kick-offs lack left/right side for: {}".format(", ".join(sorted(missing))))
    return directions


def _contiguous_runs(mask, minimum_frames):
    changes = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends) if end - start >= minimum_frames]


def _nearest_ball_team(tracking):
    ball_visible = np.isfinite(tracking.ball_x_m) & np.isfinite(tracking.ball_y_m)
    distance = np.hypot(tracking.player_x_m - tracking.ball_x_m[:, None],
                        tracking.player_y_m - tracking.ball_y_m[:, None])
    distance[~np.isfinite(distance)] = np.inf
    nearest_slot = distance.argmin(axis=1)
    valid = ball_visible & np.isfinite(distance[np.arange(len(distance)), nearest_slot])
    return np.where(valid, tracking.team_index_by_slot[nearest_slot], -1)


def _team_ball_advance(tracking, team):
    attacks_right = tracking.team_attacks_toward_increasing_x[:, team]
    if team == AWAY:
        attacks_right = ~tracking.team_attacks_toward_increasing_x[:, HOME]
    return np.where(attacks_right, tracking.ball_x_m, tracking.pitch_length_m - tracking.ball_x_m)


def _half_label(tracking, frame):
    observed = np.flatnonzero(np.isfinite(tracking.seconds_from_kickoff))
    if not len(observed):
        return "1st Half"
    first_direction = tracking.team_attacks_toward_increasing_x[observed[0], HOME]
    return "1st Half" if tracking.team_attacks_toward_increasing_x[frame, HOME] == first_direction else "2nd Half"


def _seconds(tracking, frame, end=False):
    value = tracking.seconds_from_kickoff[frame]
    if not np.isfinite(value):
        value = frame / tracking.frames_per_second
    return float(value + (1 / tracking.frames_per_second if end else 0))


def _record(tracking, code, team, team_names, start, end, shape_quality, rule_strength=0.65):
    confidence = inferred_confidence(rule_strength, shape_quality, multiplier=0.60)
    return Annotation(
        id="", start_s=_seconds(tracking, start), end_s=_seconds(tracking, end - 1, end=True), code=code,
        labels={"Team": team_names[team], "Half": _half_label(tracking, start), "Confidence": "{:.2f}".format(confidence)},
    )


def infer_team_phase_records(tracking, team_names):
    """Infer conservative phase candidates whose reference XML needs no extra sublabels.

    This is intentionally a first-pass weak-label generator. It uses only sustained
    five-second runs, ball-visible nearest-team possession proxy, attack-relative
    territory, and shape reliability. Ball/box codes requiring side or entry labels
    are excluded until their specific derivations are implemented.
    """
    if len(team_names) != 2:
        raise ValueError("team_names must contain home and away names")
    minimum_frames = int(round(MINIMUM_PHASE_SECONDS * tracking.frames_per_second))
    possession = _nearest_ball_team(tracking)
    features = TrackingFeatureEngine(tracking).build_features()
    records = []
    for team, prefix in ((HOME, "home"), (AWAY, "away")):
        advance = _team_ball_advance(tracking, team)
        quality = features["{}_shape_quality".format(prefix)]
        own_ball = possession == team
        candidates = {
            "BUILD UP": own_ball & (advance < tracking.pitch_length_m / 3),
            "PROGRESSION": own_ball & (advance >= tracking.pitch_length_m / 3) & (advance < tracking.pitch_length_m * 2 / 3),
            "CREATING CHANCES": own_ball & (advance >= tracking.pitch_length_m * 2 / 3),
            "DEFENDING IN DEFENSIVE THIRD": (possession == 1 - team) & (advance < tracking.pitch_length_m / 3),
            "DEFENDING IN MIDDLE THIRD": ((possession == 1 - team) & (advance >= tracking.pitch_length_m / 3) &
                                           (advance < tracking.pitch_length_m * 2 / 3)),
            "DEFENDING IN ATTACKING THIRD": (possession == 1 - team) & (advance >= tracking.pitch_length_m * 2 / 3),
        }
        forward_step = np.full(len(advance), np.nan)
        step = int(round(tracking.frames_per_second))
        forward_step[step:] = advance[step:] - advance[:-step]
        candidates["LONG BALL"] = own_ball & (forward_step >= tracking.pitch_length_m * 0.2)
        for code, mask in candidates.items():
            for start, end in _contiguous_runs(mask, minimum_frames):
                records.append(_record(tracking, code, team, team_names, start, end, np.nanmedian(quality[start:end])))

    return sorted(records, key=lambda record: (record.start_s, record.end_s, record.code, record.labels["Team"]))


def merge_records(direct_records, inferred_records):
    """Chronologically merge observed and inferred annotations with new XML-safe IDs."""
    records = sorted([*direct_records, *inferred_records],
                     key=lambda record: (record.start_s, record.end_s, record.code, record.labels.get("Team", "")))
    return [Annotation(str(index), record.start_s, record.end_s, record.code, record.labels)
            for index, record in enumerate(records, start=1)]
