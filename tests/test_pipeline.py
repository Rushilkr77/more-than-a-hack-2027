"""Indexing, schema and submission tests. Synthetic 300-frame bundle always; real-data tests skip
unless the licensed bundle is present."""

import csv
import os

import numpy as np
import pandas as pd
import pytest

from games import GAME, GAMES, asset_paths, half_video_blocks
from scripts.build_submission import build_submission
from src.decoder import majority
from src.evidence import ball_nearest
from src.features import frame_table as ft

N = 300
BLOCKS = [(1, 50, 120), (2, 180, 250)]
KICKOFFS = {1: 50, 2: 200}
LENGTH, WIDTH = GAMES[GAME]["pitch"]
NAN = float("nan")

META = """<root><Teams><Team id="H"><Name>Home</Name><ProviderTeamsParameters><ProviderParameter><Name>attack_direction_first_half</Name><Value>left_to_right</Value></ProviderParameter></ProviderTeamsParameters></Team><Team id="A"><Name>Away</Name><ProviderTeamsParameters><ProviderParameter><Name>attack_direction_first_half</Name><Value>right_to_left</Value></ProviderParameter></ProviderTeamsParameters></Team></Teams>
<Players><Player id="1" teamId="H"/><Player id="2" teamId="H"/><Player id="3" teamId="A"/><Player id="4" teamId="A"/></Players></root>"""


def _tag(start, end, code, **tags):
    labels = "".join("<label><group>{}</group><text>{}</text></label>".format(k, v) for k, v in tags.items())
    return "<instance><start>{}</start><end>{}</end><code>{}</code>{}</instance>".format(start, end, code, labels)


TAGS = "<file>{}</file>".format("".join([
    _tag(2.0, 4.0, "SET PIECES", Team="Barcelona", Type="Kick Off (Start)", Half="1st Half"),
    _tag(3.0, 6.0, "BUILD UP", Team="Atletic Club"),
    _tag(8.0, 9.0, "SET PIECES", Team="N/A", Type="Throw-in"),
]))


def _raw_line(row):
    """frame_id row+1. Home slots 0,1 at x .2,.4; away 2,3 at x .6,.8; ball between slots 1 and 2."""
    players = [(.2, .5), (.4, .5), (.6, .5), (.8, .5)]
    ball = (.45, .5)                      # nearest home slot 1
    if row == 0:
        players[0] = None                 # one dropped track
    if row == 10:
        ball, players[1], players[2] = (.5, .5), (.4, .5), (.5, .6)   # tie in normalised units; away nearer in metres
    if 100 <= row < 110:
        ball = None                       # ball lost
    if row >= 150:
        ball = (.55, .5)                  # nearest away slot 2
    fmt = lambda p: "NaN,NaN,NaN" if p is None else "{:.6f},{:.6f},2.50".format(*p)  # noqa: E731
    return "{}:{}:{}".format(row + 1, ";".join(fmt(p) for p in players), fmt(ball))


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("bundle")
    (root / "tactical").mkdir()
    paths = asset_paths(data_dir=str(root))
    open(paths["atd_meta"], "w").write(META)
    open(paths["atd_raw"], "w").write("\n".join(_raw_line(r) for r in range(N)) + "\n")
    open(paths["smart_tagging"], "w").write(TAGS)
    cols = ["frame"] + ["h{}{}".format(i, j) for i in range(3) for j in range(3)] + ["reprojection_error", "valid"]
    with open(paths["homography"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in range(N):
            ok = not 90 <= r < 100
            w.writerow([r] + ([1] * 9 if ok else [NAN] * 9) + [0.3, int(ok)])
    return str(root)


@pytest.fixture(scope="module")
def table(bundle):
    return ft.build_table(data_dir=bundle, n_frames=N, blocks=BLOCKS, kickoffs=KICKOFFS)[0]


def test_official_constants():
    assert GAMES[GAME]["n_video_frames"] == 199319
    blocks = half_video_blocks(GAME)
    assert blocks == [(1, 11185, 90651), (2, 110808, 190197)]
    assert sum(b - a + 1 for _, a, b in blocks) == 158857
    assert ft._defaults(None, None)[1] == {1: 11185, 2: 113742}
    assert 113742 - 110808 == 2934     # pre-kickoff frames in the second graded block


def test_frame_index_is_frame_id_minus_one(table):
    assert len(table) == N and (table["frame"] == np.arange(N)).all()
    assert table.loc[0, "ball_x_m"] == pytest.approx(.45 * LENGTH, rel=1e-6)   # frame_id 1 -> frame 0
    assert table.loc[N - 1, "ball_x_m"] == pytest.approx(.55 * LENGTH, rel=1e-6)


def test_metres_not_normalised_units(table):
    row = table.loc[10]
    assert row.nearest_team == 1
    assert row.home_ball_dist_m == pytest.approx(.1 * LENGTH, rel=1e-5)
    assert row.away_ball_dist_m == pytest.approx(.1 * WIDTH, rel=1e-5)


def test_missing_ball_is_not_dead(table):
    lost = table.loc[100:109]
    assert not lost["ball_visible"].any() and (lost["nearest_team"] == -1).all()
    ev = ball_nearest.score(table).loc[100:109]
    assert ev[["score_H", "score_A", "score_D"]].isna().all().all() and (ev["confidence"] == 0).all()
    assert table.loc[110:149, "nearest_team"].eq(0).all() and table.loc[150:, "nearest_team"].eq(1).all()


def test_dropped_track_stats(table):
    assert table.loc[0, "n_home"] == 1 and table.loc[1, "n_home"] == 2 and table.loc[0, "n_away"] == 2
    assert table.loc[0, "home_cx_m"] == pytest.approx(.4 * LENGTH, rel=1e-5)
    assert table.loc[1, "home_depth_m"] == pytest.approx(.1 * LENGTH, rel=1e-5)   # std of {.2,.4}


def test_period_graded_pre_kickoff(table):
    assert table["graded"].sum() == 71 + 71
    assert [table.period[f] for f in (0, 120, 121, 179, 180, 299)] == [1, 1, 0, 0, 2, 2]
    assert table["pre_kickoff"].sum() == 20 and table.pre_kickoff[180:200].all() and not table.pre_kickoff[200]
    assert not (table.pre_kickoff & ~table.graded).any()


def test_homography_and_tags(table):
    assert not table.hom_valid[90:100].any() and table.hom_valid[:90].all()
    t = table.set_index("frame")
    assert (t.tag_n_active[60], t.tag_home_n[60], t.tag_away_n[60], t.tag_setpiece[60], t.tag_setpiece_team[60]) == (1, 1, 0, 5, 0)
    assert (t.tag_n_active[80], t.tag_home_n[80], t.tag_away_n[80]) == (2, 1, 1)
    assert (t.tag_n_active[99], t.tag_n_active[100]) == (2, 1)          # half-open end: 4.0 s == frame 100
    assert (t.tag_setpiece[210], t.tag_setpiece_team[210], t.tag_n_active[210]) == (1, -1, 1)   # N/A-team throw-in
    assert t.tag_setpiece[100] == 0 and t.tag_setpiece[225] == 0


def test_schema(table):
    assert list(table.columns) == ft.COLUMNS
    assert {n: str(table[n].dtype) for n in ft.COLUMNS} == {n: d for n, d, _ in ft.SCHEMA}
    assert len(set(ft.COLUMNS)) == len(ft.COLUMNS)


def test_cache_roundtrip_and_staleness(bundle, tmp_path, monkeypatch):
    monkeypatch.setenv("MTAH_CACHE_DIR", str(tmp_path / "cache"))
    kwargs = dict(n_frames=N, blocks=BLOCKS, kickoffs=KICKOFFS)
    args = repr(sorted((k, repr(v)) for k, v in kwargs.items()))
    assert not ft.is_fresh(bundle, args)                              # nothing cached yet
    built = ft.build(data_dir=bundle, **kwargs)
    assert ft.is_fresh(bundle, args) and not ft.is_fresh(bundle)       # build args are part of the key

    paths = asset_paths(data_dir=bundle)
    original = open(paths["smart_tagging"]).read()
    try:
        open(paths["smart_tagging"], "w").write(original + " ")        # input changed -> stale
        assert not ft.is_fresh(bundle, args)
    finally:
        open(paths["smart_tagging"], "w").write(original)
    assert ft.is_fresh(bundle, args)

    monkeypatch.setattr(ft, "code_sha256", lambda files=None, extra="": "different code")
    assert not ft.is_fresh(bundle, args)                               # code changed -> stale
    monkeypatch.undo()
    monkeypatch.setenv("MTAH_CACHE_DIR", str(tmp_path / "cache"))

    raw = paths["atd_raw"]
    os.rename(raw, raw + ".moved")
    try:
        with pytest.warns(UserWarning):
            assert ft.is_fresh(bundle, args)                           # raw absent: code-only check
        pd.testing.assert_frame_equal(pd.read_parquet(ft._paths()[0]), built)
    finally:
        os.rename(raw + ".moved", raw)


def test_baseline_submission_valid_and_deterministic(table, tmp_path):
    config = {"run_id": "t", "evidence": "ball_nearest", "decoder": {"name": "majority", "window": 5}}
    p1, s1 = build_submission(config, str(tmp_path / "a"), table, n_rows=N)
    p2, s2 = build_submission(config, str(tmp_path / "b"), table, n_rows=N)
    assert s1 == s2
    lines = open(p1).read().splitlines()
    assert lines[0] == "frame,label" and len(lines) == N + 1
    assert set(l.split(",")[1] for l in lines[1:]) <= {"H", "A", "D"}


def test_decoder_removes_flicker():
    ev = pd.DataFrame({"score_H": [1., 0., 1., 1., 1., 1., 1.], "score_A": [0., 1., 0., 0., 0., 0., 0.]})
    assert (majority.decode(ev, 0) == ["H", "A", "H", "H", "H", "H", "H"]).all()
    assert (majority.decode(ev, 2) == "H").all()


# -- real data (skipped unless the licensed bundle is available) -----------------------------

HAVE_DATA = all(os.path.isfile(p) for k, p in asset_paths().items() if k != "video")
real = pytest.mark.skipif(not HAVE_DATA, reason="licensed bundle not present (see DATA.md)")


@pytest.fixture(scope="module")
def real_table():
    return ft.load(rebuild=True)


@real
def test_real_shape_and_visibility(real_table):
    assert len(real_table) == 199319 and real_table["graded"].sum() == 158857
    assert real_table["ball_visible"].sum() == 113488                      # DATA.md
    assert real_table.loc[real_table.graded, "ball_visible"].sum() == 94214


@real
def test_real_track_counts(real_table):
    tracks = real_table.n_home + real_table.n_away
    assert tracks[real_table.graded].mean() == pytest.approx(20.5, abs=0.05)      # DATA.md: mean over graded frames
    assert tracks[real_table.period == 0].mean() < tracks[real_table.graded].mean()   # halftime is thinner, not empty


@real
def test_real_baseline_reproduces_published_csv(real_table, tmp_path):
    config = {"run_id": "baseline_smooth50", "evidence": "ball_nearest", "decoder": {"name": "majority", "window": 50}}
    path, _ = build_submission(config, str(tmp_path), real_table)
    mine = pd.read_csv(path)["label"]
    official = pd.read_csv(os.path.join(ft.ROOT, "evaluation", "results", "baseline_smooth50.csv"))["label"]
    assert (mine == official).mean() > 0.9999


# -- broadcast cross-check ---------------------------------------------------------------------

from src.features import broadcast_xcheck as bx  # noqa: E402


def _tracks(xs, team):
    x = np.array([xs], dtype=np.float64)
    return {"x": x, "y": np.full_like(x, 34.0), "team": np.array(team)}


def test_pair_cost_matches_and_rejects():
    tc = _tracks([10, 20, 30, 40], [0, 0, 1, 1])
    bc = _tracks([10.5, 20.5], [0, 0])
    assert bx.pair_cost(bc, tc, [0], [0])[0][0] == pytest.approx(0.5)
    far = _tracks([60, 70], [0, 0])
    assert bx.pair_cost(far, tc, [0], [0])[0][0] == 10.0             # capped, nothing near
    assert np.isnan(bx.pair_cost(bc, tc, [0], [5])[0][0])          # t out of range
    wrong_team = _tracks([10.5, 20.5], [1, 1])                       # nearest away track is 30 m: caps at 10, 9.5
    assert bx.pair_cost(wrong_team, tc, [0], [0])[0][0] == pytest.approx(9.75)


def test_grade_windows_tiers():
    n = 600
    ok = np.ones(n, dtype=bool)
    # all evidence good -> tier 1 in the middle; edges still have a full-enough window
    tier, conf, _, _ = bx.grade_windows(ok, ok, ok, ok)
    assert (tier == 1).all() and conf.min() >= 0.75
    # player evidence good, no co-visible ball -> capped at tier 2
    tier, conf, _, _ = bx.grade_windows(ok, ok, ~ok, ~ok)
    assert (tier == 2).all() and 0.5 <= conf.min() and conf.max() <= 0.75
    # nothing matches -> tier 3
    tier, _, _, _ = bx.grade_windows(~ok, ok, ~ok, ~ok)
    assert (tier == 3).all()
    # no evidence at all -> tier 4, conf NaN
    tier, conf, _, _ = bx.grade_windows(~ok, ~ok, ~ok, ~ok)
    assert (tier == 4).all() and np.isnan(conf).all()


BC_FILES = None
try:
    bx.broadcast_paths()
    BC_FILES = True
except IOError:
    pass


@pytest.mark.skipif(not (HAVE_DATA and BC_FILES), reason="broadcast bundle not present")
def test_real_offsets_are_stable():
    from src.data import atd
    tc, meta_raw = ft.load_slots(), bx.broadcast_paths()
    bc = atd.load_slots(meta_raw[0], meta_raw[1], bx.N_BC, GAMES[GAME]["pitch"])
    assert all(abs(a - b) <= 1 for a, b in zip(bx.refit_offsets(bc, tc), bx.OFFSETS))


# -- weighted decoder and optional filled evidence ---------------------------------------------

from src.evidence import ball_filled  # noqa: E402


def test_decoder_weights_and_anchor():
    ev = pd.DataFrame({"score_H": [1., 1., 1., 0., 0., 0., 0.], "score_A": [0., 0., 0., 1., 1., 1., 1.],
                       "weight": [1., 1., 1., .5, .5, .5, .5], "anchor": [1., 1., 1., 0., 0., 0., 0.]})
    assert (majority.decode(ev.drop(columns=["weight", "anchor"]), 6) == "A").all()      # 3 H votes vs 4 A votes
    assert (majority.decode(ev, 6) == "H").all()                                          # 3.0 vs 2.0 once fills weigh 0.5
    assert (majority.decode(ev, 6, min_anchor=4) == "D").all()                            # not enough trusted evidence
    fills_only = ev.assign(anchor=0.0)
    assert (majority.decode(fills_only, 6, min_anchor=1) == "D").all()                    # fills cannot decide alone


def test_ball_filled_marks_sources():
    table = pd.DataFrame({"frame": range(5), "nearest_team": [0, -1, -1, -1, -1],
                          "ball_visible": [True, False, False, False, False]})
    tc = {"x": np.array([[10., 90.]] * 5), "y": np.full((5, 2), 34.), "team": np.array([0, 1])}   # home at 10 m, away at 90 m
    xc = pd.DataFrame({"bc_ball_x_m": [np.nan, 88., 12., 89., np.nan], "bc_ball_y_m": 34.,
                       "tier": [1, 2, 1, 3, 4], "conf": [.9, .6, .8, .4, np.nan]})
    ev = ball_filled.score(table, (1, 2), xc, tc)
    assert list(ev.ball_source) == [0, 2, 1, -1, -1]                  # tier 3 / no ball never fill
    assert list(ev.weight) == [1.0, .6, .8, 0.0, 0.0] and list(ev.anchor) == [1., 0., 0., 0., 0.]
    assert list(ev.score_H.fillna(-1)) == [1.0, 0.0, 1.0, -1, -1]      # tier-2 fill at x=88 -> away; tier-1 at 12 -> home
    assert list(ev.confidence) == [1., 1., 1., 0., 0.] and ev.score_D.isna().all()
    only1 = ball_filled.score(table, (1,), xc, tc)
    assert list(only1.ball_source) == [0, -1, 1, -1, -1]
    # no fills at all must equal the baseline evidence votes
    none = ball_filled.score(table, (), xc, tc)
    assert (none.score_H.fillna(-1) == ball_nearest.score(table).score_H.fillna(-1)).all()


# -- orientation, shape, margin, hints, symmetry ----------------------------------------------

def test_orientation_margin_and_ball_territory(table):
    assert table.home_attacks_right[:121].all() and table.home_attacks_right[121:180].all()   # halftime gap keeps the first-half value
    assert not table.home_attacks_right[180:].any()
    assert table.loc[0, "ball_home_adv_m"] == pytest.approx(.45 * LENGTH, rel=1e-5) and table.loc[0, "ball_third_home"] == 1
    assert table.loc[250, "ball_home_adv_m"] == pytest.approx(LENGTH - .55 * LENGTH, rel=1e-5)   # second half: axis flips
    assert table.loc[10, "ball_margin_m"] == pytest.approx(abs(.1 * LENGTH - .1 * WIDTH), rel=1e-4)
    assert np.isnan(table.loc[100, "ball_margin_m"]) and table.loc[100, "ball_third_home"] == -1


def test_shape_columns_trim_ghost_and_flip():
    x = np.array([[5, 20, 22, 24, 26, 40, 45, 50, 70, 72, 99.]])
    y = np.full_like(x, 34.0)
    shape, conf = ft.shape_columns(x, y, np.array([True]), 105.3, 68.0)
    assert shape["dline_m"][0] == pytest.approx(23.0) and shape["cen_m"][0] == pytest.approx(46.8)
    assert shape["fwd_m"][0] == pytest.approx(64.0)            # 50, 70, 72: the 99 m ghost is trimmed
    assert shape["line_gap_m"][0] == pytest.approx(41.0) and conf[0] == 1.0
    flipped, _ = ft.shape_columns(x, y, np.array([False]), 105.3, 68.0)   # attacking the other way: distances from the other goal
    assert flipped["dline_m"][0] == pytest.approx(np.mean(np.sort(105.3 - x[0])[1:5]))
    few, conf = ft.shape_columns(x[:, :6], y[:, :6], np.array([True]), 105.3, 68.0)
    assert np.isnan(few["dline_m"][0]) and conf[0] == 0.0       # below MIN_SHAPE_TRACKS: NaN, zero confidence
    off = x.copy(); off[0, 8:] = 200.0                          # three tracks off the pitch are ignored
    assert ft.shape_columns(off, y, np.array([True]), 105.3, 68.0)[1][0] == pytest.approx(0.6)


def test_tag_possession_hint():
    ev = lambda a, b, code, team: {"start_frame": a, "end_frame": b, "code": code, "tags": {"Team": team}}   # noqa: E731
    events = [ev(0, 10, "BUILD UP", "Barcelona"), ev(10, 20, "DEFENDING IN MIDDLE THIRD", "Barcelona"),
              ev(30, 40, "BUILD UP", "Barcelona"), ev(30, 40, "BUILD UP", "Atletic Club"),     # conflict
              ev(50, 60, "BUILD UP", "N/A"), ev(70, 80, "LONG BALL", "Barcelona")]              # no team / not a hint code
    hint = ft._tag_columns(events, 100)[5]
    assert (hint[0:10] == 1).all() and (hint[10:20] == -1).all() and (hint[30:40] == 0).all()
    assert (hint[50:60] == 0).all() and (hint[70:80] == 0).all() and hint.dtype == np.int8


def test_swap_sides_is_involution_and_mirrors(table):
    sw = ft.swap_sides(table)
    pd.testing.assert_frame_equal(ft.swap_sides(sw), table)
    assert (sw.n_home == table.n_away).all() and (sw.home_cx_m.fillna(-1) == table.away_cx_m.fillna(-1)).all()
    ok = table.nearest_team >= 0
    assert (sw.nearest_team[ok] == 1 - table.nearest_team[ok]).all() and (sw.nearest_team[~ok] == -1).all()
    assert (sw.tag_poss_hint == -table.tag_poss_hint).all() and (sw.home_attacks_right != table.home_attacks_right).all()


def test_decoder_tie_rule_and_antisymmetry():
    h = np.array([1., 1., np.nan, 0., 0., 0., np.nan, 1., 0.])
    ev = pd.DataFrame({"score_H": h, "score_A": 1 - h})
    mirror = pd.DataFrame({"score_H": 1 - h, "score_A": h})
    swap = {"H": "A", "A": "H", "D": "D"}
    for w in (0, 1, 2):
        lab = majority.decode(ev, w, tie="previous")
        assert [swap[v] for v in lab] == list(majority.decode(mirror, w, tie="previous"))      # swapping teams swaps labels
    tied = pd.DataFrame({"score_H": [1., 0., np.nan], "score_A": [0., 1., np.nan]})
    assert list(majority.decode(tied, 1, tie="home")) == ["H", "H", "A"]          # ties go to home
    assert list(majority.decode(tied, 1, tie="previous")) == ["D", "D", "A"]      # no earlier label to hold -> D, never a team
    held = pd.DataFrame({"score_H": [1., 1., 1., 0., np.nan], "score_A": [0., 0., 0., 1., np.nan]})
    assert list(majority.decode(held, 1, tie="previous")) == ["H", "H", "H", "H", "A"]   # frame 3 is a tie: holds H
    with pytest.raises(ValueError):
        majority.decode(tied, 1, tie="other")


def test_grade_windows_reads_window_at_call_time(monkeypatch):
    n = 600
    ok = np.ones(n, dtype=bool)
    sparse = np.zeros(n, dtype=bool); sparse[::40] = True              # little ball evidence
    a = bx.grade_windows(ok, ok, sparse, sparse)[3]
    monkeypatch.setattr(bx, "WINDOW", 10)
    assert not np.array_equal(a, bx.grade_windows(ok, ok, sparse, sparse)[3])


@real
def test_real_orientation_and_phase_ordering(real_table):
    t = real_table
    for half, (a, b) in ((1, (11185, 90651)), (2, (110808, 190197))):
        s = t.iloc[a:b + 1]
        assert s.home_attacks_right.all() == (half == 1)
        assert (s.home_cx_m.mean() < s.away_cx_m.mean()) == (half == 1)       # XML direction agrees with measured positions
    from starter import load_smart_tagging as lst
    by_code = {}
    for e in lst.load():
        team = ft.TEAM_TAG.get((e["tags"].get("Team") or "").strip().lower())
        a, b = e["start_frame"], e["end_frame"]
        if team is None or b <= a or not t.graded[a:b].all():
            continue
        col = "home_dline_m" if team == 0 else "away_dline_m"
        by_code.setdefault(e["code"].upper(), []).append(np.nanmean(t[col][a:b]))
    order = ["DEFENDING IN DEFENSIVE THIRD", "BUILD UP", "PROGRESSION", "CREATING CHANCES"]
    means = [np.mean(by_code[c]) for c in order]
    assert means == sorted(means) and means[-1] - means[0] > 30               # line height rises from low block to attack
    assert real_table.home_shape_conf.mean() > 0.7 and (real_table.tag_poss_hint != 0).mean() > 0.05


def test_filled_configs_declare_broadcast_use():
    import json
    for name, expect in (("baseline", False), ("filled_t1", True), ("filled_t12", True)):
        cfg = json.load(open(os.path.join(ft.ROOT, "configs", name + ".json")))
        assert bool(cfg.get("uses_broadcast")) == expect
        assert (cfg["evidence"] == "ball_filled") == expect          # the marker cannot drift from the evidence choice


def test_manifest_and_slice_report(table, tmp_path):
    import json
    from scripts.slice_report import slice_scores
    config = {"run_id": "m", "evidence": "ball_nearest", "decoder": {"name": "majority", "window": 5}}
    path, sha = build_submission(config, str(tmp_path), table, n_rows=N)
    m = json.load(open(str(tmp_path / "m.manifest.json")))
    assert m["output_sha256"] == sha and m["uses_broadcast"] is False
    sh = m["graded_frame_shares"]
    assert sh["evidence_tactical_ball"] + sh["no_ball_evidence"] == pytest.approx(1.0, abs=1e-3)
    assert sh["label_H"] + sh["label_A"] + sh["label_D"] == pytest.approx(1.0, abs=1e-3)

    labels = np.array(pd.read_csv(path)["label"])
    res, tab = slice_scores(table, labels, labels)                     # identical prediction: perfect everywhere
    assert res["accuracy"] == 1.0 and (tab["accuracy"] == 1.0).all() and tab["share of errors"].sum() == 0
    pred = labels.copy(); pred[table.ball_visible.to_numpy() & table.graded.to_numpy()] = "D"
    _, tab = slice_scores(table, labels, pred)
    miss = tab[(tab["slice"] == "tactical ball") & (tab.bucket == "missing")].iloc[0]
    vis = tab[(tab["slice"] == "tactical ball") & (tab.bucket == "visible")].iloc[0]
    assert miss["accuracy"] == 1.0 and vis["accuracy"] < 1.0 and vis["share of errors"] == pytest.approx(1.0)   # damage is localised


def test_schema_doc_in_sync():
    from scripts.render_schema_doc import DOC, render
    assert open(DOC).read() == render()
