"""Sportec -> damaged BAR-ATH-shaped bundle: same columns as BAR-ATH, reproducible, no labels or identities leak."""

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

from src.features import frame_table as ft

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("bsb", os.path.join(ROOT, "scripts", "build_sportec_bundles.py"))
bsb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsb)

M, HOME, AWAY, N = "TESTMM", "CLU-H", "CLU-A", 1500       # 60 s per half


def _write_match(d):
    rng = np.random.default_rng(0)
    t0 = 1_700_000_000.0
    def iso(sec):  # noqa: E306
        return pd.Timestamp(sec, unit="s", tz="UTC").isoformat()
    sets = []
    for h, (sec, base) in enumerate((("firstHalf", 10000), ("secondHalf", 100000))):
        for team in (HOME, AWAY):
            for p in range(11):
                xy = np.cumsum(rng.normal(0, 0.06, (N, 2)), axis=0) + [(-20 if team == HOME else 20) * (1 - 2 * h), p * 5 - 25]
                v = np.r_[0, np.hypot(*np.diff(xy, axis=0).T) * 25]
                fr = "".join('<Frame N="{}" X="{:.2f}" Y="{:.2f}" S="{:.2f}"/>'.format(base + i, x, y, s * 3.6)
                             for i, ((x, y), s) in enumerate(zip(xy, v)))
                sets.append('<FrameSet GameSection="{}" TeamId="{}" PersonId="PERSON-{}-{}">{}</FrameSet>'.format(
                    sec, team, team, p, fr))
        status = (np.arange(N) % 500 > 50).astype(int)
        poss = np.where(np.arange(N) % 300 < 150, 1, 2)
        fr = "".join('<Frame N="{}" T="{}" X="{:.2f}" Y="{:.2f}" Z="{:.2f}" S="{:.2f}" BallPossession="{}" '
                     'BallStatus="{}"/>'.format(base + i, iso(t0 + h * 4000 + i / 25), 10 * np.sin(i / 50),
                                                 5 * np.cos(i / 70), abs(np.sin(i / 30)) * 3, 20, poss[i], status[i])
                     for i in range(N))
        sets.append('<FrameSet GameSection="{}" TeamId="BALL" PersonId="BALL">{}</FrameSet>'.format(sec, fr))
        sets.append('<FrameSet GameSection="{}" TeamId="referee" PersonId="REF"><Frame N="{}" X="0" Y="0" S="0"/>'
                    '</FrameSet>'.format(sec, base))
    with open(os.path.join(d, "DFL_04_03_positions_raw_observed_DFL-COM-X_DFL-MAT-{}.xml".format(M)), "w") as f:
        f.write("<PutDataRequest><Positions>{}</Positions></PutDataRequest>".format("".join(sets)))
    with open(os.path.join(d, "DFL_02_01_matchinformation_DFL-COM-X_DFL-MAT-{}.xml".format(M)), "w") as f:
        f.write('<PutDataRequest><MatchInformation><Teams><Team TeamId="{}" Role="home"/><Team TeamId="{}" '
                'Role="guest"/></Teams><Environment PitchX="105.00" PitchY="68.00"/></MatchInformation>'
                '</PutDataRequest>'.format(HOME, AWAY))
    with open(os.path.join(d, "DFL_03_02_events_raw_DFL-COM-X_DFL-MAT-{}.xml".format(M)), "w") as f:
        f.write('<PutDataRequest><Event EventTime="{}"><KickOff TeamLeft="{}" GameSection="firstHalf"/></Event>'
                '<Event EventTime="{}"><KickOff TeamLeft="{}" GameSection="secondHalf"/></Event>'
                '<Event EventTime="{}"><Play Team="{}"/></Event></PutDataRequest>'.format(
                    iso(t0), HOME, iso(t0 + 4000), AWAY, iso(t0 + 10), HOME))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("sportec")
    src, out, lab = (d / k for k in ("src", "out", "lab"))
    src.mkdir()
    _write_match(str(src))
    bsb.SRC_DIR, bsb.OUT_DIR, bsb.LABEL_DIR = str(src), str(out), str(lab)
    meta = bsb.build(M)
    return out / M, lab / M, meta


def test_bundle_has_only_tracking_files(built):
    bundle, labels, _ = built
    assert sorted(os.listdir(bundle / "tactical")) == [M + "_tactical_atd.txt", M + "_tactical_atd.xml"]
    assert sorted(os.listdir(labels)) == ["labels_adjusted.csv", "labels_raw.csv", "meta.json"]


def test_same_columns_as_bar_ath_and_no_labels(built):
    table, slots = ft.bundle_table(str(built[0]))
    assert list(table.columns) == ft.COLUMNS
    assert len(table) == 2 * N and table["graded"].all() and not table["pre_kickoff"].any()
    assert not any(c.startswith("tag_") or c.startswith("hom_") or c == "graded" for c in ft.model_columns())
    assert slots["x"].shape[1] == built[2]["slots"]


def test_no_identity_or_provider_fields_in_bundle(built):
    text = "".join(open(built[0] / "tactical" / f).read() for f in os.listdir(built[0] / "tactical"))
    for leak in ("PERSON-", HOME, AWAY, "BallPossession", "BallStatus", "Z="):
        assert leak not in text


def test_damage_hides_ball_and_breaks_tracks(built):
    table, _ = ft.bundle_table(str(built[0]))
    assert 0.25 < 1 - table["ball_visible"].mean() < 0.75
    assert ft.bundle_args(str(built[0] / "tactical" / (M + "_tactical_atd.xml")))["pitch"] == (105.0, 68.0)


def test_labels_raw_and_adjusted(built):
    raw = pd.read_csv(built[1] / "labels_raw.csv")["label"].to_numpy()
    adj = pd.read_csv(built[1] / "labels_adjusted.csv")["label"].to_numpy()
    assert set(raw) <= {"H", "A", "D"} and raw[0] == "D" and raw[100] == "H" and raw[200] == "A"
    assert (adj == "D").sum() == (raw == "D").sum()     # adjustment never touches dead ball


def test_reproducible(built, tmp_path):
    first = open(built[0] / "tactical" / (M + "_tactical_atd.txt")).read()
    bsb.OUT_DIR, bsb.LABEL_DIR = str(tmp_path / "o"), str(tmp_path / "l")
    bsb.build(M)
    assert open(tmp_path / "o" / M / "tactical" / (M + "_tactical_atd.txt")).read() == first


def test_swap_sides_uses_given_pitch(built):
    table, _ = ft.bundle_table(str(built[0]))
    sw = ft.swap_sides(table, length=105.0)
    s = (sw["ball_home_adv_m"] + table["ball_home_adv_m"]).dropna()
    assert np.allclose(s, 105.0, atol=1e-3)


def test_smooth_tracker_error_keeps_sd_and_removes_wobble():
    w = np.random.default_rng(0).normal(0, 0.15, 20000)
    s = bsb.smooth_noise(w, 25.0)
    assert 0.12 < s.std() < 0.18                          # same error size
    d2 = lambda v: np.median(np.abs(v[2:] - 2 * v[1:-1] + v[:-2]))  # noqa: E731
    assert d2(s) < d2(w) / 100                            # far less frame-to-frame wobble


def test_smooth_runs_never_crosses_gaps():
    v = np.r_[np.zeros(20), np.full(5, np.nan), np.full(20, 10.0)]
    out = bsb.smooth_runs(v, 3.0)
    assert np.isnan(out[20:25]).all()
    assert np.allclose(out[:20], 0) and np.allclose(out[25:], 10)
