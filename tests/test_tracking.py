import numpy as np

from src.tracking import (MetricaATDAdapter, SportecTrackingAdapter,
                          TrackingFeatureEngine, TrackingFrames, inferred_confidence)


def test_team_shape_and_motion_are_attack_relative_for_both_teams():
    frames, slots = 3, 14
    player_x = np.full((frames, slots), np.nan)
    player_y = np.full((frames, slots), np.nan)
    # Both teams advance one metre per frame in their own attacking direction.
    for frame in range(frames):
        player_x[frame, :7] = np.array([5, 10, 20, 30, 40, 50, 60]) + frame
        player_x[frame, 7:] = np.array([100, 95, 85, 75, 65, 55, 45]) - frame
        player_y[frame] = np.tile([8, 16, 24, 32, 40, 48, 56], 2)
    tracking = TrackingFrames(
        player_x_m=player_x,
        player_y_m=player_y,
        team_index_by_slot=np.array([0] * 7 + [1] * 7),
        team_attacks_toward_increasing_x=np.array([[True, False]] * frames),
        pitch_length_m=105.0,
        pitch_width_m=68.0,
        frames_per_second=1.0,
    )

    features = TrackingFeatureEngine(tracking, velocity_half_window_frames=1).build_features()

    assert np.allclose(features["home_defensive_line_position_m"], [25, 26, 27])
    assert np.allclose(features["away_defensive_line_position_m"], [25, 26, 27])
    assert np.allclose(features["home_midfield_line_position_m"], [35, 36, 37])
    assert np.allclose(features["away_midfield_line_position_m"], [35, 36, 37])
    assert features["home_midfield_line_velocity_mps"][1] == 1.0
    assert features["away_defensive_line_velocity_mps"][1] == 1.0
    assert np.allclose(features["home_shape_quality"], 0.4)


def test_confidence_scales_only_inferred_evidence():
    assert inferred_confidence(rule_strength=0.5, observation_quality=0.8) == 0.24
    assert inferred_confidence(rule_strength=0.5, observation_quality=0.8, multiplier=0.6) == 0.24
    assert inferred_confidence(rule_strength=0.5, observation_quality=0.8, multiplier=1.2) == 0.48


def test_provider_adapters_emit_the_same_canonical_conventions(tmp_path):
    slots = {"x": np.array([[10.0, 95.0]]), "y": np.array([[20.0, 30.0]]), "team": np.array([0, 1])}
    metrica = MetricaATDAdapter(slots, [True], 105, 68, 25).load()
    assert metrica.team_attacks_toward_increasing_x.tolist() == [[True, False]]
    assert metrica.participants_by_slot[0].participant_key == "slot_0"
    assert metrica.participants_by_slot[0].identity_quality == "tracking_slot"

    metadata = tmp_path / "metadata.xml"
    metadata.write_text("""<PutDataRequest><MatchInformation><General KickoffTime=\"2022-01-01T12:00:00+00:00\" />
    <Environment PitchX=\"105\" PitchY=\"68\" />
    <Teams><Team TeamId=\"H\" Role=\"home\"><Players><Player PersonId=\"home-player\" ShirtNumber=\"9\" FirstName=\"Home\"
    LastName=\"Player\" Shortname=\"H. Player\" Starting=\"true\" PlayingPosition=\"FW\" /></Players></Team>
    <Team TeamId=\"A\" Role=\"guest\"><Players><Player PersonId=\"away-player\" ShirtNumber=\"1\" Starting=\"false\"
    PlayingPosition=\"GK\" /></Players></Team></Teams>
    </MatchInformation></PutDataRequest>""")
    positions = tmp_path / "positions.xml"
    positions.write_text("""<PutDataRequest><Positions>
    <FrameSet GameSection=\"firstHalf\" TeamId=\"H\" PersonId=\"home-player\"><Frame N=\"10\" T=\"2022-01-01T12:00:01+00:00\" X=\"10\" Y=\"20\" /></FrameSet>
    <FrameSet GameSection=\"firstHalf\" TeamId=\"A\" PersonId=\"away-player\"><Frame N=\"10\" T=\"2022-01-01T12:00:01+00:00\" X=\"95\" Y=\"30\" /></FrameSet>
    <FrameSet GameSection=\"firstHalf\" TeamId=\"BALL\" PersonId=\"ball\"><Frame N=\"10\" T=\"2022-01-01T12:00:01+00:00\" X=\"50\" Y=\"34\" /></FrameSet>
    </Positions></PutDataRequest>""")
    sportec = SportecTrackingAdapter(positions, metadata, {"firstHalf": (True, False)}).load()
    assert sportec.player_x_m.tolist() == [[10.0, 95.0]]
    assert sportec.team_index_by_slot.tolist() == [0, 1]
    assert sportec.team_attacks_toward_increasing_x.tolist() == [[True, False]]
    assert sportec.ball_x_m.tolist() == [50.0]
    assert sportec.ball_y_m.tolist() == [34.0]
    assert sportec.seconds_from_kickoff.tolist() == [1.0]
    assert sportec.participants_by_slot[0].source_person_id == "home-player"
    assert sportec.participants_by_slot[0].shirt_number == "9"
    assert sportec.participants_by_slot[0].playing_position == "FW"
    assert sportec.participants_by_slot[0].identity_quality == "verified_person"


def test_missing_team_slots_produce_missing_shape_not_an_index_error():
    tracking = TrackingFrames(
        player_x_m=np.array([[10.0]]), player_y_m=np.array([[20.0]]), team_index_by_slot=np.array([0]),
        team_attacks_toward_increasing_x=np.array([[True, False]]),
        pitch_length_m=105, pitch_width_m=68, frames_per_second=25,
    )
    features = TrackingFeatureEngine(tracking).build_features()
    assert np.isnan(features["away_defensive_line_position_m"][0])
    assert features["away_shape_quality"][0] == 0
