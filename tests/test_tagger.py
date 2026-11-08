"""Generated Smart Tagging: each rule fires on a hand-built case, and the tagger never sees labels."""

import inspect

import numpy as np
import pandas as pd

from scripts.smart_tagging_audit import audit_file
from src.data.idsse_smart_tagging import write_xml
from src.features import tagger

L, W, FPS, N = 105.0, 68.0, 25, 1500


def _table(**over):
    t = pd.DataFrame({
        "period": np.ones(N, np.int8), "graded": np.ones(N, bool), "pre_kickoff": np.zeros(N, bool),
        "home_attacks_right": np.ones(N, bool), "nearest_team": np.zeros(N, np.int8),
        "ball_margin_m": np.full(N, 3.0), "ball_x_m": np.full(N, 20.0), "ball_y_m": np.full(N, W / 2),
        "ball_home_adv_m": np.full(N, 20.0), "ball_range_back_1s_m": np.full(N, 10.0),
        "ball_range_fwd_1s_m": np.full(N, 10.0), "pl_slow_share": np.full(N, 0.1),
        "ball_dist_touchline_m": np.full(N, -20.0), "ball_dist_goalline_m": np.full(N, -20.0),
        "ball_dist_corner_m": np.full(N, 40.0), "ball_dist_goal_m": np.full(N, 40.0), "ball_dist_centre_m": np.full(N, 30.0),
    })
    for k, v in over.items():
        t[k] = v
    return t


def _slots(n_home_in_box=0):
    x = np.full((N, 22), 50.0)
    x[:, 11:11 + n_home_in_box] = L - 5          # home attacks right: opponent box at x >= L - 16.5
    return {"x": x, "y": np.full((N, 22), W / 2), "team": np.r_[np.zeros(11, np.int8), np.ones(11, np.int8)]}


def _slots_home_box(k):
    s = _slots()
    s["x"][:, :k] = L - 5
    return s


def codes(tags):
    return [(t.code, t.labels["Team"]) for t in tags]


def test_signature_has_no_label_inputs():
    assert list(inspect.signature(tagger.generate_tags).parameters) == ["table", "slots", "pitch", "fps"]


def test_build_up_and_mirror_from_possession_guess():
    tags, cols = tagger.generate_tags(_table(), _slots(), (L, W), FPS)
    assert ("BUILD UP", "Home") in codes(tags) and ("DEFENDING IN ATTACKING THIRD", "Away") in codes(tags)
    assert (cols["gtag_poss"] == 1).all() and (cols["gtag_phase"] == 1).all()


def test_possession_held_through_gaps_and_short_flicker_smoothed():
    nt = np.zeros(N, np.int8)
    nt[400:420] = 1                               # 0.8 s flicker to away
    nt[700:900] = -1                              # ball gap: hold home
    _, cols = tagger.generate_tags(_table(nearest_team=nt), _slots(), (L, W), FPS)
    assert (cols["gtag_poss"] == 1).all()


def test_contested_votes_ignored():
    nt = np.ones(N, np.int8)                      # away nearest everywhere ...
    nt[:100] = 0
    margin = np.full(N, 0.5)                      # ... but never clearly
    margin[:100] = 3.0
    _, cols = tagger.generate_tags(_table(nearest_team=nt, ball_margin_m=margin), _slots(), (L, W), FPS)
    assert (cols["gtag_poss"] == 1).all()         # only the confident home votes count


def test_long_ball_fires_on_30m_in_under_2_5s():
    bx = np.full(N, 20.0)
    bx[600:] = 60.0                               # 40 m in one frame is a jump (> 40 m/s): no
    bx[600:640] = np.linspace(20, 60, 40)         # 40 m over 1.6 s: yes
    tags, cols = tagger.generate_tags(_table(ball_x_m=bx), _slots(), (L, W), FPS)
    assert ("LONG BALL", "Home") in codes(tags) and cols["gtag_long_ball"].max() == 1


def test_throw_in_from_still_ball_at_touchline():
    back = np.full(N, 10.0)
    back[500:560] = 0.2
    fwd = np.full(N, 0.5)
    fwd[560:600] = 12.0
    touch = np.full(N, -20.0)
    touch[450:650] = -0.3
    tags, cols = tagger.generate_tags(_table(ball_range_back_1s_m=back, ball_range_fwd_1s_m=fwd,
                                             ball_dist_touchline_m=touch), _slots(), (L, W), FPS)
    sp = [t for t in tags if t.code == "SET PIECES"]
    assert len(sp) == 1 and sp[0].labels["Type"] == "Throw-in" and sp[0].labels["Team"] == "N/A"
    assert set(np.unique(cols["gtag_setpiece"])) == {0, 1}


def test_open_play_still_ball_needs_slow_players():
    back = np.full(N, 10.0)
    back[500:560] = 0.2
    fwd = np.full(N, 0.5)
    fwd[560:600] = 12.0
    tags, _ = tagger.generate_tags(_table(ball_range_back_1s_m=back, ball_range_fwd_1s_m=fwd), _slots(), (L, W), FPS)
    assert not [t for t in tags if t.code == "SET PIECES"]


def test_box_codes_and_players_in_box():
    adv = np.full(N, 95.0)                        # ball in home's attacking box all along
    tags, cols = tagger.generate_tags(_table(ball_home_adv_m=adv, ball_x_m=adv), _slots_home_box(4), (L, W), FPS)
    got = codes(tags)
    assert ("BALL IN FINAL THIRD", "Home") in got and ("BALL IN THE BOX", "Home") in got
    pib = [t for t in tags if t.code == "PLAYERS IN THE BOX"][0]
    assert pib.labels["Max Players in the box"] == "4" and cols["gtag_players_in_box_n"].max() == 4


def test_tags_never_cross_period_or_pre_kickoff():
    period = np.r_[np.ones(N // 2), np.full(N - N // 2, 2)].astype(np.int8)
    pre = np.zeros(N, bool)
    pre[:200] = True
    tags, cols = tagger.generate_tags(_table(period=period, pre_kickoff=pre,
                                             home_attacks_right=period == 1), _slots(), (L, W), FPS)
    assert (cols["gtag_poss"][:200] == 0).all()
    for t in tags:
        a, b = round(t.start_s * FPS), round(t.end_s * FPS) - 1
        assert period[a] == period[b] and a >= 200


def test_swap_sides_negates_signed_generated_tags():
    from src.features import frame_table as ft
    t = pd.DataFrame({c: np.ones(3, np.int8) for c, _, _ in tagger.GTAG_COLUMNS})
    for c in ft.COLUMNS:
        if c not in t:
            t[c] = np.zeros(3)
    t["home_attacks_right"] = np.ones(3, bool)
    sw = ft.swap_sides(t, length=105.0)
    assert (sw["gtag_poss"] == -1).all() and (sw["gtag_phase"] == 1).all() and (sw["gtag_setpiece"] == 1).all()


def test_xml_passes_audit(tmp_path):
    tags, _ = tagger.generate_tags(_table(), _slots(), (L, W), FPS)
    write_xml(tags, tmp_path / "t.xml")
    assert not audit_file(tmp_path / "t.xml").errors
