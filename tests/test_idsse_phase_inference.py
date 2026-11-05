import numpy as np

from src.data.idsse_phase_inference import attack_directions_from_events, infer_team_phase_records, merge_records
from src.data.idsse_smart_tagging import Annotation
from src.tracking import TrackingFrames


def test_kickoff_team_left_and_right_define_half_attack_directions(tmp_path):
    events = tmp_path / "events.xml"
    events.write_text("""<PutDataRequest>
    <Event EventTime=\"2022-01-01T12:00:00+00:00\"><KickOff GameSection=\"firstHalf\" TeamLeft=\"A\" TeamRight=\"H\" /></Event>
    <Event EventTime=\"2022-01-01T13:00:00+00:00\"><KickOff GameSection=\"secondHalf\" TeamLeft=\"H\" TeamRight=\"A\" /></Event>
    </PutDataRequest>""")

    directions = attack_directions_from_events(events, "H", "A")

    assert directions == {"firstHalf": (False, True), "secondHalf": (True, False)}


def test_low_advancing_possession_emits_a_lower_confidence_build_up_interval():
    frames = 150  # six seconds at 25 fps: the minimum phase duration
    player_x = np.empty((frames, 14))
    player_y = np.tile([8, 16, 24, 32, 40, 48, 56] * 2, (frames, 1))
    player_x[:, :7] = [5, 10, 20, 30, 40, 50, 60]
    player_x[:, 7:] = [100, 95, 85, 75, 65, 55, 45]
    tracking = TrackingFrames(
        player_x_m=player_x, player_y_m=player_y, team_index_by_slot=np.array([0] * 7 + [1] * 7),
        team_attacks_toward_increasing_x=np.array([[True, False]] * frames),
        pitch_length_m=105, pitch_width_m=68, frames_per_second=25,
        ball_x_m=np.linspace(10, 30, frames), ball_y_m=np.full(frames, 34.0),
    )

    records = infer_team_phase_records(tracking, ("Home FC", "Away FC"))

    build_up = [record for record in records if record.code == "BUILD UP"]
    assert len(build_up) == 1
    assert build_up[0].labels["Team"] == "Home FC"
    assert 0 < float(build_up[0].labels["Confidence"]) < 1


def test_nearest_player_flicker_does_not_create_transition_labels():
    frames = 150
    player_x = np.empty((frames, 14))
    player_y = np.tile([8, 16, 24, 32, 40, 48, 56] * 2, (frames, 1))
    player_x[:, :7] = [5, 10, 20, 30, 40, 50, 60]
    player_x[:, 7:] = [100, 95, 85, 75, 65, 55, 45]
    ball_x = np.where(np.arange(frames) % 2, 10.0, 95.0)
    tracking = TrackingFrames(
        player_x_m=player_x, player_y_m=player_y, team_index_by_slot=np.array([0] * 7 + [1] * 7),
        team_attacks_toward_increasing_x=np.array([[True, False]] * frames),
        pitch_length_m=105, pitch_width_m=68, frames_per_second=25,
        ball_x_m=ball_x, ball_y_m=np.full(frames, 34.0),
    )

    records = infer_team_phase_records(tracking, ("Home FC", "Away FC"))

    assert not {"ATTACKING TRANSITION", "DEFENSIVE TRANSITION"} & {record.code for record in records}


def test_merge_records_sorts_and_reassigns_xml_ids():
    direct = [Annotation("9", 10.0, 15.0, "GOALS", {"Team": "Away FC", "Half": "1st Half", "Confidence": "1.00"})]
    inferred = [Annotation("", 5.0, 10.0, "BUILD UP", {"Team": "Home FC", "Half": "1st Half", "Confidence": "0.30"})]

    merged = merge_records(direct, inferred)

    assert [(record.id, record.code) for record in merged] == [("1", "BUILD UP"), ("2", "GOALS")]
