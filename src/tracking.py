"""Provider-neutral tracking features used by tactical tag rules.

The engine measures team shape in each team's attacking direction.  It deliberately
does not infer possession, player roles, or tactical tags: those are separate rules
which can consume these repeatable measurements later.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET

import numpy as np


HOME, AWAY = 0, 1
TEAM_NAMES = ("home", "away")
MINIMUM_SHAPE_TRACKS = 7
DEFAULT_INFERRED_CONFIDENCE_MULTIPLIER = 0.60


@dataclass(frozen=True)
class Participant:
    """Provider participant metadata aligned to one tracking-array slot."""

    participant_key: str
    team_index: int
    identity_quality: str
    source_person_id: str | None = None
    source_team_id: str | None = None
    shirt_number: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    short_name: str | None = None
    playing_position: str | None = None
    starting: bool | None = None


@dataclass
class TrackingFrames:
    """Canonical player positions: rows are frames, columns are provider slots."""

    player_x_m: np.ndarray
    player_y_m: np.ndarray
    team_index_by_slot: np.ndarray
    team_attacks_toward_increasing_x: np.ndarray
    pitch_length_m: float
    pitch_width_m: float
    frames_per_second: float
    participants_by_slot: tuple[Participant, ...] | None = None
    ball_x_m: np.ndarray | None = None
    ball_y_m: np.ndarray | None = None
    seconds_from_kickoff: np.ndarray | None = None

    def __post_init__(self):
        self.player_x_m = np.asarray(self.player_x_m, dtype=float)
        self.player_y_m = np.asarray(self.player_y_m, dtype=float)
        self.team_index_by_slot = np.asarray(self.team_index_by_slot, dtype=np.int8)
        self.team_attacks_toward_increasing_x = np.asarray(self.team_attacks_toward_increasing_x, dtype=bool)
        if self.player_x_m.ndim != 2 or self.player_x_m.shape != self.player_y_m.shape:
            raise ValueError("player_x_m and player_y_m must be equal two-dimensional arrays")
        frames, slots = self.player_x_m.shape
        if self.ball_x_m is None and self.ball_y_m is None:
            self.ball_x_m = self.ball_y_m = np.full(frames, np.nan)
        elif self.ball_x_m is None or self.ball_y_m is None:
            raise ValueError("ball_x_m and ball_y_m must be provided together")
        else:
            self.ball_x_m = np.asarray(self.ball_x_m, dtype=float)
            self.ball_y_m = np.asarray(self.ball_y_m, dtype=float)
        if self.ball_x_m.shape != (frames,) or self.ball_y_m.shape != (frames,):
            raise ValueError("ball_x_m and ball_y_m must have shape (frames,)")
        if self.seconds_from_kickoff is None:
            self.seconds_from_kickoff = np.arange(frames, dtype=float) / self.frames_per_second
        else:
            self.seconds_from_kickoff = np.asarray(self.seconds_from_kickoff, dtype=float)
        if self.seconds_from_kickoff.shape != (frames,):
            raise ValueError("seconds_from_kickoff must have shape (frames,)")
        if self.team_index_by_slot.shape != (slots,) or not np.isin(self.team_index_by_slot, (HOME, AWAY)).all():
            raise ValueError("team_index_by_slot must assign every slot to home (0) or away (1)")
        if self.team_attacks_toward_increasing_x.shape != (frames, 2):
            raise ValueError("team_attacks_toward_increasing_x must have shape (frames, 2)")
        if min(self.pitch_length_m, self.pitch_width_m, self.frames_per_second) <= 0:
            raise ValueError("pitch dimensions and frames_per_second must be positive")
        if self.participants_by_slot is None:
            self.participants_by_slot = tuple(
                Participant("slot_{}".format(slot), int(team), "unknown")
                for slot, team in enumerate(self.team_index_by_slot))
        else:
            self.participants_by_slot = tuple(self.participants_by_slot)
        if len(self.participants_by_slot) != slots:
            raise ValueError("participants_by_slot must contain one participant per tracking slot")
        if any(participant.team_index != team for participant, team in zip(self.participants_by_slot, self.team_index_by_slot)):
            raise ValueError("participant team_index must agree with team_index_by_slot")


class MetricaATDAdapter:
    """Adapt already-loaded More Than A Hack ATD slots to :class:`TrackingFrames`."""

    def __init__(self, slots, home_attacks_toward_increasing_x, pitch_length_m, pitch_width_m, frames_per_second):
        self.slots = slots
        self.home_attacks_toward_increasing_x = np.asarray(home_attacks_toward_increasing_x, dtype=bool)
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.frames_per_second = frames_per_second

    def load(self):
        home_direction = self.home_attacks_toward_increasing_x
        participants = tuple(
            Participant("slot_{}".format(slot), int(team), "tracking_slot")
            for slot, team in enumerate(self.slots["team"]))
        return TrackingFrames(
            player_x_m=self.slots["x"],
            player_y_m=self.slots["y"],
            team_index_by_slot=self.slots["team"],
            team_attacks_toward_increasing_x=np.column_stack((home_direction, ~home_direction)),
            pitch_length_m=self.pitch_length_m,
            pitch_width_m=self.pitch_width_m,
            frames_per_second=self.frames_per_second,
            participants_by_slot=participants,
            ball_x_m=self.slots.get("ball_x"),
            ball_y_m=self.slots.get("ball_y"),
        )


class SportecTrackingAdapter:
    """Read Sportec position XML without guessing its attack directions.

    ``attacks_toward_increasing_x_by_section`` maps each XML GameSection to
    ``(home_direction, away_direction)``.  This is explicit because the source
    position XML alone does not establish a direction convention safely.
    """

    def __init__(self, positions_path, metadata_path, attacks_toward_increasing_x_by_section, frames_per_second=25.0):
        self.positions_path = Path(positions_path)
        self.metadata_path = Path(metadata_path)
        self.directions = attacks_toward_increasing_x_by_section
        self.frames_per_second = frames_per_second

    def load(self):
        home_team_id, away_team_id, length, width, metadata_participants, kickoff_time = self._metadata()
        slots, frame_numbers, sections = self._index(home_team_id, away_team_id)
        if not frame_numbers:
            raise ValueError("{}: no player frames found".format(self.positions_path))
        missing = set(sections) - set(self.directions)
        if missing:
            raise ValueError("attack direction missing for Sportec sections: {}".format(", ".join(sorted(missing))))

        first_frame, last_frame = min(frame_numbers), max(frame_numbers)
        n_frames = last_frame - first_frame + 1
        x = np.full((n_frames, len(slots)), np.nan)
        y = np.full_like(x, np.nan)
        ball_x = np.full(n_frames, np.nan)
        ball_y = np.full(n_frames, np.nan)
        seconds = np.full(n_frames, np.nan)
        directions = np.zeros((n_frames, 2), dtype=bool)
        observed_frame = np.zeros(n_frames, dtype=bool)
        slot_index = {slot: index for index, slot in enumerate(slots)}

        for frame_set in self._frame_sets():
            section = frame_set.get("GameSection")
            team_id = frame_set.get("TeamId")
            person_id = frame_set.get("PersonId")
            slot = (team_id, person_id)
            if team_id == "BALL":
                for frame in frame_set.findall("Frame"):
                    row = int(frame.get("N")) - first_frame
                    observed_frame[row] = True
                    directions[row] = self.directions[section]
                    if kickoff_time is not None and frame.get("T"):
                        seconds[row] = (self._timestamp(frame.get("T")) - kickoff_time).total_seconds()
                    if frame.get("X") is not None and frame.get("Y") is not None:
                        ball_x[row] = float(frame.get("X"))
                        ball_y[row] = float(frame.get("Y"))
                continue
            if slot not in slot_index:
                continue
            for frame in frame_set.findall("Frame"):
                row = int(frame.get("N")) - first_frame
                observed_frame[row] = True
                directions[row] = self.directions[section]
                if kickoff_time is not None and frame.get("T"):
                    seconds[row] = (self._timestamp(frame.get("T")) - kickoff_time).total_seconds()
                if frame.get("X") is not None and frame.get("Y") is not None:
                    x[row, slot_index[slot]] = float(frame.get("X"))
                    y[row, slot_index[slot]] = float(frame.get("Y"))

        # A dropped observation inherits the immediately preceding section direction.
        for row in range(1, n_frames):
            if not observed_frame[row]:
                directions[row] = directions[row - 1]
        participants = tuple(metadata_participants.get(
            slot,
            Participant(slot[1], HOME if slot[0] == home_team_id else AWAY, "observed_person_id",
                        source_person_id=slot[1], source_team_id=slot[0]),
        ) for slot in slots)
        return TrackingFrames(
            player_x_m=x,
            player_y_m=y,
            team_index_by_slot=np.array([HOME if team_id == home_team_id else AWAY for team_id, _ in slots]),
            team_attacks_toward_increasing_x=directions,
            pitch_length_m=length,
            pitch_width_m=width,
            frames_per_second=self.frames_per_second,
            participants_by_slot=participants,
            ball_x_m=ball_x,
            ball_y_m=ball_y,
            seconds_from_kickoff=seconds,
        )

    def _metadata(self):
        root = ET.parse(self.metadata_path).getroot()
        team_elements = {team.get("Role"): team for team in root.iter("Team")}
        teams = {role: team.get("TeamId") for role, team in team_elements.items()}
        if not teams.get("home") or not teams.get("guest"):
            raise ValueError("{}: expected home and guest teams".format(self.metadata_path))
        environment = next(root.iter("Environment"), None)
        if environment is None:
            raise ValueError("{}: pitch dimensions missing".format(self.metadata_path))
        participants = {}
        for team_index, role in ((HOME, "home"), (AWAY, "guest")):
            team = team_elements[role]
            team_id = teams[role]
            for player in team.findall("./Players/Player"):
                person_id = player.get("PersonId")
                if not person_id:
                    continue
                starting = player.get("Starting")
                participants[(team_id, person_id)] = Participant(
                    participant_key=person_id, team_index=team_index, identity_quality="verified_person",
                    source_person_id=person_id, source_team_id=team_id, shirt_number=player.get("ShirtNumber"),
                    first_name=player.get("FirstName"), last_name=player.get("LastName"),
                    short_name=player.get("Shortname"), playing_position=player.get("PlayingPosition"),
                    starting=None if starting is None else starting.strip().lower() == "true",
                )
        general = root.find("./MatchInformation/General")
        kickoff_time = self._timestamp(general.get("KickoffTime")) if general is not None and general.get("KickoffTime") else None
        return teams["home"], teams["guest"], float(environment.get("PitchX")), float(environment.get("PitchY")), participants, kickoff_time

    @staticmethod
    def _timestamp(value):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    def _frame_sets(self):
        for _, frame_set in ET.iterparse(self.positions_path, events=("end",)):
            if frame_set.tag == "FrameSet":
                yield frame_set
                frame_set.clear()

    def _index(self, home_team_id, away_team_id):
        slots, frame_numbers, sections = set(), set(), set()
        for frame_set in self._frame_sets():
            team_id, person_id = frame_set.get("TeamId"), frame_set.get("PersonId")
            if team_id == "BALL":
                sections.add(frame_set.get("GameSection"))
                frame_numbers.update(int(frame.get("N")) for frame in frame_set.findall("Frame"))
                continue
            if team_id not in (home_team_id, away_team_id) or not person_id:
                continue
            slots.add((team_id, person_id))
            sections.add(frame_set.get("GameSection"))
            frame_numbers.update(int(frame.get("N")) for frame in frame_set.findall("Frame"))
        return sorted(slots, key=lambda slot: (slot[0] != home_team_id, slot[1])), frame_numbers, sections


def inferred_confidence(rule_strength, observation_quality, multiplier=DEFAULT_INFERRED_CONFIDENCE_MULTIPLIER):
    """Confidence for inferred tags; direct source events should not call this."""
    if multiplier < 0:
        raise ValueError("multiplier must be non-negative")
    return float(np.clip(rule_strength, 0, 1) * np.clip(observation_quality, 0, 1) * multiplier)


class TrackingFeatureEngine:
    """Build the seven compact, attack-relative team-shape features per frame."""

    def __init__(self, tracking, velocity_half_window_frames=12):
        if velocity_half_window_frames < 1:
            raise ValueError("velocity_half_window_frames must be at least 1")
        self.tracking = tracking
        self.velocity_half_window_frames = velocity_half_window_frames

    def build_features(self):
        features = {}
        for team_index, team_name in enumerate(TEAM_NAMES):
            x = self.tracking.player_x_m[:, self.tracking.team_index_by_slot == team_index]
            y = self.tracking.player_y_m[:, self.tracking.team_index_by_slot == team_index]
            direction = self.tracking.team_attacks_toward_increasing_x[:, team_index]
            shape = self._team_shape(x, y, direction)
            features.update({"{}_{}_m".format(team_name, key): value for key, value in shape.items() if key != "shape_quality"})
            features["{}_shape_quality".format(team_name)] = shape["shape_quality"]
            for line in ("defensive_line_position", "midfield_line_position"):
                features["{}_{}_velocity_mps".format(team_name, line.replace("_position", ""))] = self._velocity(shape[line])
        return features

    def _team_shape(self, x, y, attacks_toward_increasing_x):
        if x.shape[1] == 0:
            missing = np.full(x.shape[0], np.nan)
            return {"defensive_line_position": missing, "midfield_line_position": missing,
                    "forward_line_position": missing, "team_width": missing,
                    "shape_quality": np.zeros(x.shape[0])}
        is_on_pitch = ((x >= -1) & (x <= self.tracking.pitch_length_m + 1) &
                       (y >= -1) & (y <= self.tracking.pitch_width_m + 1))
        metres_from_own_goal = np.where(attacks_toward_increasing_x[:, None], x, self.tracking.pitch_length_m - x)
        ordered_positions = np.sort(np.where(is_on_pitch, metres_from_own_goal, np.nan), axis=1)
        number_of_tracks = np.isfinite(ordered_positions).sum(axis=1)
        enough_tracks = number_of_tracks >= MINIMUM_SHAPE_TRACKS
        rows = np.arange(len(number_of_tracks))
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            defensive_line = np.nanmean(ordered_positions[:, 1:5], axis=1)
            midfield_line = np.nanmedian(ordered_positions[:, 1:], axis=1)
            forward_indices = np.stack([np.clip(number_of_tracks - offset, 0, ordered_positions.shape[1] - 1)
                                        for offset in (4, 3, 2)], axis=1)
            forward_line = np.nanmean(ordered_positions[rows[:, None], forward_indices], axis=1)
            team_width = np.nanquantile(np.where(is_on_pitch, y, np.nan), 0.9, axis=1) - np.nanquantile(
                np.where(is_on_pitch, y, np.nan), 0.1, axis=1)
        quality = np.where(enough_tracks, np.clip((number_of_tracks - 5) / 5.0, 0, 1), 0.0)
        return {
            "defensive_line_position": np.where(enough_tracks, defensive_line, np.nan),
            "midfield_line_position": np.where(enough_tracks, midfield_line, np.nan),
            "forward_line_position": np.where(enough_tracks, forward_line, np.nan),
            "team_width": np.where(enough_tracks, team_width, np.nan),
            "shape_quality": quality,
        }

    def _velocity(self, positions_m):
        result = np.full(len(positions_m), np.nan)
        half_window = self.velocity_half_window_frames
        if len(positions_m) <= 2 * half_window:
            return result
        before, after = positions_m[:-2 * half_window], positions_m[2 * half_window:]
        valid = np.isfinite(before) & np.isfinite(after)
        result[half_window:-half_window] = np.where(
            valid, (after - before) * self.tracking.frames_per_second / (2 * half_window), np.nan)
        return result
