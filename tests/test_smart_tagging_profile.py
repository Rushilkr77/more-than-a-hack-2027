import numpy as np
import pandas as pd

from src.data.smart_tagging_profile import profile


def test_profile_uses_one_fixed_team_and_ball_box_measurement_set():
    table = pd.DataFrame({
        "frame": [0, 1, 2, 3], "period": [1, 1, 1, 1], "graded": [True] * 4,
        "pre_kickoff": [False] * 4, "home_attacks_right": [True] * 4,
        "ball_visible": [True] * 4, "ball_x_m": [80.0, 90.0, 95.0, 100.0],
        "ball_y_m": [34.0] * 4,
        "home_defensive_line_position_m": [20.0, 22.0, 24.0, 26.0],
        "home_midfield_line_position_m": [30.0, 32.0, 34.0, 36.0],
        "home_forward_line_position_m": [45.0, 47.0, 49.0, 51.0],
        "home_team_width_m": [30.0] * 4, "home_defensive_line_velocity_mps": [1.0] * 4,
        "home_midfield_line_velocity_mps": [1.5] * 4, "home_shape_quality": [1.0] * 4,
        "away_defensive_line_position_m": [50.0] * 4, "away_midfield_line_position_m": [55.0] * 4,
        "away_forward_line_position_m": [60.0] * 4, "away_team_width_m": [25.0] * 4,
        "away_defensive_line_velocity_mps": [0.0] * 4, "away_midfield_line_velocity_mps": [0.0] * 4,
        "away_shape_quality": [1.0] * 4,
    })
    slots = {
        "x": np.array([[85.0, 10.0], [85.0, 10.0], [85.0, 10.0], [85.0, 10.0]]),
        "y": np.array([[34.0, 34.0]] * 4),
        "team": np.array([0, 1]),
    }
    events = [{"code": "BUILD UP", "start_frame": 0, "end_frame": 4, "start_s": 0.0, "end_s": 0.16,
               "tags": {"Team": "Barcelona", "Half": "1st Half"}}]

    report = profile(table, slots, events, {"barcelona": 0, "atletic club": 1}, 100.0, 68.0, 25.0)

    instance = report["codes"]["BUILD UP"]["intervals"][0]
    assert set(instance["metrics"]) == set(report["metric_definitions"])
    assert instance["metrics"]["team_defensive_line_position_m"] == 23.0
    assert instance["metrics"]["ball_in_opposition_box_fraction"] == 0.75
    assert instance["metrics"]["players_in_opposition_box_median"] == 1.0
    assert instance["metrics"]["ball_forward_displacement_m"] == 20.0
    assert report["codes"]["BUILD UP"]["summary"]["interval_count"] == 1
    assert report["codes"]["BUILD UP"]["summary"]["team_known_interval_count"] == 1
    assert report["codes"]["BUILD UP"]["summary"]["baseline_delta"]["team_defensive_line_position_m"]["median"] == 0.0
    template = report["codes"]["BUILD UP"]["sportec_rule_template"]
    assert set(template["team_shape_relative_to_half_iqr"]) == {
        "team_defensive_line_position_m", "team_midfield_line_position_m", "team_forward_line_position_m",
        "team_width_m", "team_defensive_line_velocity_mps", "team_midfield_line_velocity_mps", "team_shape_quality",
    }
    assert "ball_in_opposition_box_fraction" in template["ball_box_iqr"]
