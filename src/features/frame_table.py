"""Schema v0: one row per tactical frame, built once from the raw bundle and cached.

Every module loads this table (`load()`), never the raw ATD. Rebuild: `python -m src.features.frame_table`.
The column contract is SCHEMA below. Raw positions (cx_m, ball_x_m, ...) keep the ATD orientation; the shape and
ball-territory columns are in "metres from that team's own goal line" (bigger = more advanced), using
`home_attacks_right` (from the ATD XML, flipped in the second half). A missing ball is NaN, never D.

Notes for downstream users (all measured, see scripts/audit_data.py):
  * `ball_speed_atd_ms` is blank on ~91% of ball-visible frames and unreliable where present: do not use it.
  * Out-of-bounds ball positions (2.7% of visible) are not a reliable dead-ball cue; excess tracks (>11 per team,
    ~8% of frames) are on the pitch, so shape columns use trimmed statistics instead of a pitch mask.
  * Shape columns validate against Smart Tagging phases, NOT against `nearest_team` (which leans to defenders).
  * Ball gaps are long (85% of missing frames sit in runs > 10 s), so short-gap interpolation will not help.
"""

import hashlib
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, asset_paths, half_video_blocks, source_frames  # noqa: E402
from starter import load_smart_tagging  # noqa: E402
from src.data import atd  # noqa: E402
from src.data.atd import AWAY, HOME  # noqa: E402

SCHEMA_VERSION = "v0"

# Tagged-team text -> 0 home / 1 away. Lower-cased; N/A (and anything unlisted) means "no team".
TEAM_TAG = {"barcelona": HOME, "atletic club": AWAY, "athletic club": AWAY}
# SET PIECES "Type" tag -> code; unlisted types get SETPIECE_OTHER.
SETPIECE_CODES = {"throw-in": 1, "corner kick": 2, "goal kick": 3, "free kick": 4,
                  "kick off (start)": 5, "kick off (after goal)": 6}
SETPIECE_OTHER = 7
# Smart Tagging phase codes (lower-cased) where the tagged Team is in possession / defending. Automatic tags: a hint, not truth.
ATTACK_CODES = {"build up", "progression", "creating chances", "attacking transition", "ball in final third",
                "players in the box", "ball in the box"}
DEFEND_CODES = {"defending in defensive third", "defending in middle third", "defending in attacking third",
                "defensive transition"}
MIN_SHAPE_TRACKS = 7           # on-pitch tracks needed before any shape column is reported

TEAM_COLS = (("cx_m", "mean x of visible tracks, metres"),
             ("cy_m", "mean y of visible tracks, metres"),
             ("depth_m", "std of x (length-wise spread), metres"),
             ("width_m", "std of y (width-wise spread), metres"),
             ("speed_ms", "mean speed of visible tracks, m/s"))

SHAPE_COLS = (("dline_m", "defensive line: mean metres-from-own-goal of the 2nd-5th deepest on-pitch tracks (GK excluded)"),
              ("cen_m", "outfield centroid: mean metres-from-own-goal of all on-pitch tracks except the deepest"),
              ("fwd_m", "forward line: mean metres-from-own-goal of the 3 most advanced tracks after trimming the single most advanced"),
              ("line_gap_m", "fwd_m - dline_m: how stretched the team is between its lines"))

SCHEMA = [
    ("frame", "int32", "tactical video frame, 0-based; `frame_id - 1` in the ATD. Row i has frame == i"),
    ("period", "int8", "1 = frame <= end of first graded block (includes pre-match), "
                       "2 = frame >= start of second graded block (includes post-match), 0 = halftime gap"),
    ("graded", "bool", "frame is inside a graded block (158,857 frames in total)"),
    ("pre_kickoff", "bool", "graded but before that half's kickoff (second half: 110808..113741; evaluator labels these D)"),
    ("ball_visible", "bool", "ball has both coordinates in the ATD. Visibility is evidence quality, NOT dead-ball state"),
    ("ball_x_m", "float32", "ball x, metres, NaN if not visible"),
    ("ball_y_m", "float32", "ball y, metres, NaN if not visible"),
    ("ball_speed_atd_ms", "float32", "ATD ball speed field as delivered: ~91% NaN even when the ball is visible, unreliable, do not use"),
    ("n_home", "int8", "home tracks visible (slots are tracklets, not players; can exceed 11)"),
    ("n_away", "int8", "away tracks visible"),
] + [("{}_{}".format(t, c), "float32", "{}: {}".format(t, d))
     for t in ("home", "away") for c, d in TEAM_COLS] + [
    ("home_ball_dist_m", "float32", "distance from ball to nearest visible home track, metres; NaN if ball or tracks missing"),
    ("away_ball_dist_m", "float32", "distance from ball to nearest visible away track, metres; NaN if ball or tracks missing"),
    ("nearest_team", "int8", "team of the track nearest the ball: 0 home, 1 away, -1 unknown (the official baseline's signal)"),
    ("hom_valid", "bool", "tactical homography matrix exists for this frame"),
    ("hom_reproj_err", "float32", "homography reprojection error (lower is better); ~1.5 across halftime"),
    ("tag_n_active", "int16", "Smart Tagging intervals covering this frame (noisy evidence, not possession)"),
    ("tag_home_n", "int16", "active intervals tagged to the home team"),
    ("tag_away_n", "int16", "active intervals tagged to the away team (N/A-team intervals count for neither)"),
    ("tag_setpiece", "int8", "active SET PIECES type: 0 none, 1 throw-in, 2 corner, 3 goal kick, 4 free kick, "
                             "5 kick-off (start), 6 kick-off (after goal), 7 other"),
    ("tag_setpiece_team", "int8", "team of the active set piece: 0 home, 1 away, -1 none or N/A (every throw-in)"),
    # --- orientation, ball confidence, team shape, weak possession hint (appended; existing columns unchanged) ---
    ("home_attacks_right", "bool", "home attacks toward +x this half (ATD XML attack_direction_first_half, flipped in the second half; "
                                   "halftime gap uses the first-half value)"),
    ("ball_margin_m", "float32", "|nearest home dist - nearest away dist| to the ball; confidence for nearest_team "
                                 "(two cameras agree on the nearest team 67% of the time below 0.5 m, 99% above 8 m); NaN if either is missing"),
    ("ball_home_adv_m", "float32", "ball distance from the home goal line along the home attack axis; away = pitch length minus this; NaN if ball missing"),
    ("ball_third_home", "int8", "ball third from the home point of view: 0 home defensive, 1 middle, 2 home attacking; -1 unknown"),
] + [("{}_{}".format(t, c), "float32", "{}: {}".format(t, d))
     for t in ("home", "away") for c, d in SHAPE_COLS] + [
    ("home_shape_conf", "float32", "reliability 0..1 of the home shape columns: clip((on-pitch tracks - 5) / 5, 0, 1); 0 if fewer than 7 tracks (columns NaN)"),
    ("away_shape_conf", "float32", "reliability 0..1 of the away shape columns (same rule)"),
    ("tag_poss_hint", "int8", "weak possession hint from Smart Tagging phases: +1 home, -1 away, 0 none or conflicting (automatic tags, noisy)"),
]
COLUMNS = [name for name, _, _ in SCHEMA]

def _defaults(blocks, kickoffs):
    blocks = blocks or half_video_blocks(GAME)
    kickoffs = kickoffs or {1: load_smart_tagging.KICKOFF_VIDEO_FRAMES["1st Half"],
                            2: load_smart_tagging.KICKOFF_VIDEO_FRAMES["2nd Half"]}
    return blocks, kickoffs


def _team_stats(slots, mask, prefix, cols):
    x, y, s = (slots[k][:, mask] for k in "xys")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows are the NaN we want
        cols["n_" + prefix] = np.isfinite(x).sum(axis=1)
        cols[prefix + "_cx_m"] = np.nanmean(x, axis=1)
        cols[prefix + "_cy_m"] = np.nanmean(y, axis=1)
        cols[prefix + "_depth_m"] = np.nanstd(x, axis=1)
        cols[prefix + "_width_m"] = np.nanstd(y, axis=1)
        cols[prefix + "_speed_ms"] = np.nanmean(s, axis=1)
        dist = np.hypot(x - slots["ball_x"][:, None], y - slots["ball_y"][:, None])
        cols[prefix + "_ball_dist_m"] = np.nanmin(dist, axis=1) if dist.shape[1] else np.full(len(x), np.nan)


def attack_direction_first_half(meta_path):
    """True if the home (first-listed) team attacks toward +x in the first half, per the ATD metadata XML."""
    import xml.etree.ElementTree as ET
    team = next(iter(ET.parse(meta_path).getroot().iter("Team")))
    for param in team.iter("ProviderParameter"):
        if param.findtext("Name") == "attack_direction_first_half":
            return (param.findtext("Value") or "").strip().lower() == "left_to_right"
    raise ValueError("{}: no attack_direction_first_half for the home team".format(meta_path))


def shape_columns(x, y, attacks_right, length, width):
    """Own-goal-distance shape of one team. x, y: [frames, slots] metres. Returns dict of arrays + conf.

    Uses on-pitch tracks (pitch +/- 1 m); trimmed statistics (GK dropped, single most advanced track dropped)
    because duplicate/ghost tracks sit ON the pitch. NaN when fewer than MIN_SHAPE_TRACKS tracks.
    """
    on = (x >= -1) & (x <= length + 1) & (y >= -1) & (y <= width + 1)
    xa = np.where(attacks_right[:, None], x, length - x)
    srt = np.sort(np.where(on, xa, np.nan), axis=1)            # ascending, NaN last
    n = np.isfinite(srt).sum(axis=1)
    ok = n >= MIN_SHAPE_TRACKS
    rows = np.arange(len(n))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dline = np.nanmean(srt[:, 1:5], axis=1)
        cen = np.nanmean(srt[:, 1:], axis=1)
        top = np.stack([srt[rows, np.clip(n - k, 0, srt.shape[1] - 1)] for k in (4, 3, 2)], axis=1)   # 3 most advanced after trimming 1
        fwd = np.nanmean(top, axis=1)
    out = {"dline_m": dline, "cen_m": cen, "fwd_m": fwd}
    out = {k: np.where(ok, v, np.nan) for k, v in out.items()}
    out["line_gap_m"] = out["fwd_m"] - out["dline_m"]
    conf = np.where(ok, np.clip((n - 5) / 5.0, 0, 1), 0.0)
    return out, conf


def _tag_columns(events, n_frames):
    """Smart Tagging intervals (half-open frame ranges) -> per-frame counts and set-piece state."""
    n_active, n_home, n_away = (np.zeros(n_frames + 1, dtype=np.int32) for _ in range(3))
    setpiece = np.zeros(n_frames, dtype=np.int8)
    setpiece_team = np.full(n_frames, -1, dtype=np.int8)
    votes = np.zeros((2, n_frames + 1), dtype=np.int32)       # [home-in-possession votes, away-in-possession votes]
    for e in events:
        a, b = max(e["start_frame"], 0), min(e["end_frame"], n_frames)
        if a >= b:
            continue
        team = TEAM_TAG.get((e["tags"].get("Team") or "").strip().lower())
        n_active[a] += 1
        n_active[b] -= 1
        if team is not None:
            (n_home if team == HOME else n_away)[a] += 1
            (n_home if team == HOME else n_away)[b] -= 1
        code = (e["code"] or "").strip().lower()
        if team is not None and (code in ATTACK_CODES or code in DEFEND_CODES):
            home_has_ball = (team == HOME) == (code in ATTACK_CODES)
            votes[0 if home_has_ball else 1][a] += 1
            votes[0 if home_has_ball else 1][b] -= 1
        if code == load_smart_tagging.KICKOFF_CODE:
            kind = (e["tags"].get("Type") or "").strip().lower()
            setpiece[a:b] = SETPIECE_CODES.get(kind, SETPIECE_OTHER)
            setpiece_team[a:b] = -1 if team is None else team
    cum = lambda d: np.cumsum(d)[:n_frames]  # noqa: E731
    vh, va = cum(votes[0]), cum(votes[1])
    hint = np.where((vh > 0) & (va == 0), 1, np.where((va > 0) & (vh == 0), -1, 0)).astype(np.int8)
    return cum(n_active), cum(n_home), cum(n_away), setpiece, setpiece_team, hint


def build_table(data_dir=None, n_frames=None, blocks=None, kickoffs=None):
    """(frame table, slots) from the raw bundle. Pure function of the input files."""
    n_frames = n_frames or source_frames()
    blocks, kickoffs = _defaults(blocks, kickoffs)
    paths = asset_paths(data_dir=data_dir)
    slots = atd.load_slots(paths["atd_meta"], paths["atd_raw"], n_frames, GAMES[GAME]["pitch"])

    frame = np.arange(n_frames)
    cols = {"frame": frame}
    first, second = blocks[0], blocks[1]
    cols["period"] = np.where(frame <= first[2], 1, np.where(frame >= second[1], 2, 0))
    graded = np.zeros(n_frames, dtype=bool)
    pre = np.zeros(n_frames, dtype=bool)
    for half, a, b in blocks:
        graded[a:b + 1] = True
        pre[a:min(kickoffs[half], b + 1)] = True
    cols["graded"], cols["pre_kickoff"] = graded, pre

    cols["ball_visible"] = np.isfinite(slots["ball_x"])
    cols["ball_x_m"], cols["ball_y_m"], cols["ball_speed_atd_ms"] = (slots[k] for k in ("ball_x", "ball_y", "ball_s"))
    _team_stats(slots, slots["team"] == HOME, "home", cols)
    _team_stats(slots, slots["team"] == AWAY, "away", cols)

    h, a = cols["home_ball_dist_m"], cols["away_ball_dist_m"]
    cols["nearest_team"] = np.where(np.isfinite(h) & (~np.isfinite(a) | (h <= a)), 0,
                                    np.where(np.isfinite(a), 1, -1))

    hom = pd.read_csv(paths["homography"], usecols=["frame", "reprojection_error", "valid"])
    if len(hom) != n_frames or not (hom["frame"].to_numpy() == frame).all():
        raise ValueError("homography rows are not frames 0..{}".format(n_frames - 1))
    cols["hom_valid"] = hom["valid"].to_numpy().astype(bool)
    cols["hom_reproj_err"] = hom["reprojection_error"].to_numpy()

    events = load_smart_tagging.load(data_dir=data_dir)
    (cols["tag_n_active"], cols["tag_home_n"], cols["tag_away_n"],
     cols["tag_setpiece"], cols["tag_setpiece_team"], cols["tag_poss_hint"]) = _tag_columns(events, n_frames)

    # orientation, ball confidence, shape (own-goal-distance coordinates)
    length, width = GAMES[GAME]["pitch"]
    h1_right = attack_direction_first_half(paths["atd_meta"])
    right = np.where(cols["period"] == 2, not h1_right, h1_right)
    cols["home_attacks_right"] = right
    cols["ball_margin_m"] = np.abs(h - a)
    bx_ = cols["ball_x_m"]
    adv = np.where(right, bx_, length - bx_)
    cols["ball_home_adv_m"] = adv
    cols["ball_third_home"] = np.where(np.isfinite(adv), np.minimum((np.nan_to_num(adv) // (length / 3)), 2), -1)
    for k, name in ((HOME, "home"), (AWAY, "away")):
        m = slots["team"] == k
        team_right = right if k == HOME else ~right
        shape, conf = shape_columns(slots["x"][:, m], slots["y"][:, m], team_right, length, width)
        for c, v in shape.items():
            cols["{}_{}".format(name, c)] = v
        cols[name + "_shape_conf"] = conf

    table = pd.DataFrame({name: np.asarray(cols[name]).astype(dtype) for name, dtype, _ in SCHEMA})
    slot_arrays = {"x": slots["x"].astype(np.float32), "y": slots["y"].astype(np.float32),
                   "s": slots["s"].astype(np.float32), "team": slots["team"]}
    return table, slot_arrays


def swap_sides(table):
    """Mirror the table with home and away exchanged (training-time augmentation / symmetry tests)."""
    t = table.copy()
    for col in table.columns:
        if col.startswith("home_") and col != "home_attacks_right":
            other = "away_" + col[len("home_"):]
            if other in table.columns:
                t[col], t[other] = table[other].to_numpy(), table[col].to_numpy()
    t["n_home"], t["n_away"] = table["n_away"].to_numpy(), table["n_home"].to_numpy()
    t["tag_home_n"], t["tag_away_n"] = table["tag_away_n"].to_numpy(), table["tag_home_n"].to_numpy()
    flip = lambda v: np.where(v >= 0, 1 - v, v)  # noqa: E731
    t["nearest_team"] = flip(table["nearest_team"].to_numpy()).astype(table["nearest_team"].dtype)
    t["tag_setpiece_team"] = flip(table["tag_setpiece_team"].to_numpy()).astype(table["tag_setpiece_team"].dtype)
    t["tag_poss_hint"] = (-table["tag_poss_hint"].to_numpy()).astype(table["tag_poss_hint"].dtype)
    t["home_attacks_right"] = ~table["home_attacks_right"].to_numpy()
    t["ball_home_adv_m"] = (GAMES[GAME]["pitch"][0] - table["ball_home_adv_m"]).astype(table["ball_home_adv_m"].dtype)
    third = table["ball_third_home"].to_numpy()
    t["ball_third_home"] = np.where(third >= 0, 2 - third, third).astype(third.dtype)
    return t


def team_style(table, opening_frames=15000):
    """Small style table: one row per team x half, median shape over the graded block and over its first 10 minutes."""
    rows = []
    for half in (1, 2):
        g = table[(table["period"] == half) & table["graded"] & ~table["pre_kickoff"]]
        for scope, part in (("half", g), ("first 10 min", g.iloc[:opening_frames])):
            for team in ("home", "away"):
                r = {"team": team, "half": half, "scope": scope, "frames": len(part)}
                for c, _ in SHAPE_COLS:
                    r[c] = part["{}_{}".format(team, c)].median()
                r["width_m"] = part["{}_width_m".format(team)].median()
                r["shape_conf"] = part["{}_shape_conf".format(team)].mean()
                rows.append(r)
    return pd.DataFrame(rows)


# -- cache -------------------------------------------------------------------------------------
# The cache invalidates itself: meta records a hash of the code that produced it, the build args and
# the input files. load() rebuilds when any of them differ, so nobody silently reads an old table.

CODE_FILES = ("src/features/frame_table.py", "src/data/atd.py", "games.py",
              "starter/load_smart_tagging.py", "starter/baseline.py")


def cache_dir():
    return os.environ.get("MTAH_CACHE_DIR") or os.path.join(ROOT, "cache")


def _paths():
    d = cache_dir()
    return (os.path.join(d, "frame_table_{}.parquet".format(SCHEMA_VERSION)),
            os.path.join(d, "slots_{}.npz".format(SCHEMA_VERSION)),
            os.path.join(d, "frame_table_{}.meta.json".format(SCHEMA_VERSION)))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha256(files=CODE_FILES, extra=""):
    h = hashlib.sha256(extra.encode())
    for rel in files:
        with open(os.path.join(ROOT, rel), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def _inputs(data_dir):
    paths = asset_paths(data_dir=data_dir)
    paths.pop("video")
    return paths


def _input_hashes(data_dir):
    return {k: sha256_file(p) for k, p in sorted(_inputs(data_dir).items())}


def build(data_dir=None, **kwargs):
    """Build from raw and (over)write the cache, recording how it was made."""
    table, slot_arrays = build_table(data_dir=data_dir, **kwargs)
    table_path, slots_path, meta_path = _paths()
    os.makedirs(cache_dir(), exist_ok=True)
    table.to_parquet(table_path, index=False)
    np.savez(slots_path, **slot_arrays)
    args = repr(sorted((k, repr(v)) for k, v in kwargs.items()))
    with open(meta_path, "w") as f:
        json.dump({"schema": SCHEMA_VERSION, "rows": len(table), "build_args": args,
                   "code_sha256": code_sha256(extra=args), "inputs_sha256": _input_hashes(data_dir)}, f, indent=2)
    return table


def is_fresh(data_dir=None, build_args="[]"):
    """True if the cache matches the current code, build args and (when present) input files."""
    table_path, slots_path, meta_path = _paths()
    if not (os.path.isfile(table_path) and os.path.isfile(slots_path) and os.path.isfile(meta_path)):
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("schema") != SCHEMA_VERSION or meta.get("code_sha256") != code_sha256(extra=build_args):
        return False
    try:
        return meta["inputs_sha256"] == _input_hashes(data_dir)
    except OSError:
        warnings.warn("raw inputs not available; frame-table cache checked against code only")
        return True


def load(rebuild=False, data_dir=None):
    """The frame table; rebuilt from the raw bundle if missing, stale, or rebuild=True."""
    if rebuild or not is_fresh(data_dir):
        return build(data_dir=data_dir)
    table = pd.read_parquet(_paths()[0])
    if list(table.columns) != COLUMNS:
        return build(data_dir=data_dir)
    return table


def load_slots(data_dir=None):
    """Cached per-slot arrays {x, y, s: [frames, slots] float32 (m, m, m/s), team: [slots]}; same freshness rule."""
    if not is_fresh(data_dir):
        build(data_dir=data_dir)
    with np.load(_paths()[1]) as z:
        return {k: z[k] for k in z.files}


def quality_report(table):
    g = table["graded"]
    lines = ["rows {}  graded {}".format(len(table), int(g.sum())),
             "ball visible: all {:.1f}%  graded {:.1f}%".format(
                 100 * table["ball_visible"].mean(), 100 * table.loc[g, "ball_visible"].mean()),
             "mean visible tracks: graded {:.1f}  halftime {:.1f}".format(
                 (table.n_home + table.n_away)[g].mean(),
                 (table.n_home + table.n_away)[table.period == 0].mean())]
    return "\n".join(lines)


if __name__ == "__main__":
    t = load(rebuild="--rebuild" in sys.argv)
    print(quality_report(t))
