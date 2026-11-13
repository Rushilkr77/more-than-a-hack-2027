"""Sportec (DFL/IDSSE) matches -> BAR-ATH-shaped damaged tracking bundles + separate label files.

    python scripts/build_sportec_bundles.py                  # all 7 matches in data/sportec/
    python scripts/build_sportec_bundles.py --match J03WMX   # one match

Per match, exactly ONE damaged version, from a fixed per-match seed (re-running gives identical bytes):

    data/sportec_bundles/<M>/tactical/<M>_tactical_atd.txt   same line format as BAR-ATH_tactical_atd.txt
    data/sportec_bundles/<M>/tactical/<M>_tactical_atd.xml   same EPTS metadata shape as BAR-ATH_tactical_atd.xml
    data/sportec_labels/<M>/labels_raw.csv                   frame,label  (H/A/D from BallStatus/BallPossession)
    data/sportec_labels/<M>/labels_adjusted.csv              frame,label  (organiser "controlled possession" grain)
    data/sportec_labels/<M>/meta.json                        seed, drawn damage params, unit checks

The bundle folder holds only what BAR-ATH's tactical folder holds for tracking. Labels and provenance live in a
separate folder that no input code reads. Rules (see docs/data-layer/SPORTEC.md):
  * Only fields BAR-ATH has reach the bundle: x, y, speed per slot and ball. No Z, acceleration, distance, referee,
    PersonId, shirt numbers, names, positions or club ids. Slots are anonymous tracklets ("Track n"), not players.
  * Damage is mechanism-based with broad per-match ranges. BAR-ATH's own visibility mask is never read; only the
    organiser-published headline figures (about 41% of the ball missing, gaps of about 11 s) set the range centres.
  * Ball gaps are placed from geometry only (ball height, crowding, box). BallStatus / BallPossession never
    influence damage, or missingness would leak the label.
"""

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
import zlib

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "data", "sportec")
OUT_DIR = os.path.join(ROOT, "data", "sportec_bundles")
LABEL_DIR = os.path.join(ROOT, "data", "sportec_labels")
FPS = 25
BASE_SEED = 2027
KMH = 3.6

# Broad damage ranges, drawn once per match. Centres come from organiser-published BAR-ATH figures (DATA.md) or
# from tracker mechanisms; widths are deliberately wide so no match copies BAR-ATH.
RANGES = {
    "ball_hidden_share": (0.35, 0.50),       # DATA.md: 41% of graded frames
    "ball_gap_median_s": (6.0, 16.0),        # pipeline notes: median gap about 11 s
    "ball_gap_bias": (0.0, 0.6),             # share of gaps placed by geometry weights (rest uniform); capped
                                             # at 0.6: at 0.75 diagnostic A3 showed gaps avoiding dead ball
    "ball_speed_kept": (0.05, 0.15),         # ATD ball speed is mostly blank
    "tracklet_mean_s": (30.0, 120.0),        # mean tracklet length before an identity break
    "dropout_rate_per_s": (0.01, 0.06),      # per-player occlusion bursts (first build: too few, diagnostic B2)
    "dropout_median_s": (0.5, 3.0),
    "speed_warmup_s": (0.6, 1.4),            # speed blank at the start of every tracklet (tracker warm-up)
    "speed_nan_share": (0.04, 0.10),         # random speed blanks after warm-up
    "duplicate_prob": (0.02, 0.15),          # per tracklet: a ghost copy in another slot, 1-15 s
    "position_noise_m": (0.05, 0.20),
    "slots_per_team": (21, 26),
}
# Tracker ball teleports: the ball is briefly put on a (random) player for 1-5 frames, the jumps behind BAR-ATH's
# 0.23% of > 40 m/s ball steps (bundles without them: 0.07-0.15%, the first P2 round). Own random stream (seed + 1), so the
# rest of the damage stays byte-identical. Player and team are random: no label information.
TELEPORTS_PER_MIN = (0.3, 2.0)
TELEPORT_FRAMES = (1, 5)
# Tracker error shape. A real tracker's position error drifts slowly and its output is smoothed; independent per-frame
# noise makes every path shake (frame-to-frame wobble 100x a smoothed tracker). So: player error keeps its sd
# (position_noise_m) but is low-frequency, ball error has its own sd and correlation, and tracks get light Gaussian
# output smoothing (positions and speed; the ball per visible run). Ranges are wide and chosen from the mechanism,
# not fitted to BAR-ATH; BAR-ATH is only checked to fall inside the spread of the 7 matches. Gap placement is not
# touched. Own random stream (seed + 2), so the rest of the damage stays byte-identical.
TRACKER_RANGES = {
    "player_noise_corr_s": (0.3, 2.0),       # correlation time of player position error
    "player_smooth_frames": (1.0, 4.0),      # Gaussian sigma of output smoothing
    "ball_noise_m": (0.05, 0.40),
    "ball_noise_corr_frames": (2.0, 10.0),
    "ball_smooth_frames": (1.0, 4.0),
}
GAP_LOG_SIGMA = 1.0
# Adjusted labels: a possession spell shorter than this with no control event (pass, shot, ball claiming) by that
# team is a touch / deflection, not a change of controlled possession (CHALLENGE.md). Rule-of-thumb, not fitted.
MIN_CONTROL_S = 2.0
CONTROL_EVENTS = ("Play", "ShotAtGoal", "BallClaiming")


_EXCESS_ISO_FRACTION = re.compile(r"(\.\d{6})\d+(?=(?:[+-]\d{2}:\d{2})?$)")


def ts(value: str) -> float:
    """Parse provider timestamps on every supported Python version.

    Sportec and synthetic fixtures may carry nanosecond fractions, while
    ``datetime.fromisoformat`` accepts at most microseconds on Python 3.9.
    Truncating only the excess fractional digits preserves the instant to the
    precision used by the 25 fps pipeline.
    """
    if not value:
        raise ValueError("timestamp is required")
    normalized = _EXCESS_ISO_FRACTION.sub(r"\1", value.replace("Z", "+00:00"))
    return dt.datetime.fromisoformat(normalized).timestamp()


def match_files(match):
    find = lambda kind: glob.glob(os.path.join(SRC_DIR, "DFL_{}_*DFL-MAT-{}.xml".format(kind, match)))[0]  # noqa: E731
    return find("02_01_matchinformation"), find("03_02_events_raw"), find("04_03_positions_raw_observed")


def read_info(path):
    root = ET.parse(path).getroot()
    teams = {t.get("Role"): t.get("TeamId") for t in root.iter("Team")}
    env = next(root.iter("Environment"))
    return teams["home"], teams["guest"], float(env.get("PitchX")), float(env.get("PitchY"))


def read_positions(path):
    """{section: {"ball": {N, X, Y, Z, S, status, poss, T}, (team, person): {N, X, Y, S}}}"""
    out = {}
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag != "FrameSet":
            continue
        sec, team = el.get("GameSection"), el.get("TeamId")
        if team == "referee":
            el.clear()
            continue
        frames = list(el)
        get = lambda k: np.array([float(f.get(k)) for f in frames])  # noqa: E731
        rec = {"N": get("N").astype(np.int64), "X": get("X"), "Y": get("Y"), "S": get("S")}
        if team == "BALL":
            rec.update(Z=get("Z"), status=get("BallStatus").astype(np.int8), poss=get("BallPossession").astype(np.int8),
                       T=np.array([ts(f.get("T")) for f in frames]))
            out.setdefault(sec, {})["ball"] = rec
        else:
            out.setdefault(sec, {})[(team, el.get("PersonId"))] = rec
        el.clear()
    return out


def read_events(path, home, away):
    """(attack_left_team per section, [(epoch, team 0/1)] control events)."""
    left, control = {}, []
    for ev in ET.parse(path).getroot().iter("Event"):
        for k in ev.iter("KickOff"):
            if k.get("GameSection") in ("firstHalf", "secondHalf"):
                left[k.get("GameSection")] = k.get("TeamLeft")
        for tag in CONTROL_EVENTS:
            for e in ev.iter(tag):
                team = e.get("Team")
                if team in (home, away) and ev.get("EventTime"):
                    control.append((ts(ev.get("EventTime")), 0 if team == home else 1))
    return left, control


def lognormal_frames(rng, median_s, sigma, size=None):
    return np.maximum(1, np.round(rng.lognormal(np.log(median_s * FPS), sigma, size))).astype(int)


def assemble(pos, home, length, width):
    """Clean per-frame arrays on one contiguous timeline (first half rows, then second half rows)."""
    sections = [s for s in ("firstHalf", "secondHalf") if s in pos]
    starts, rows, n = {}, {}, 0
    for s in sections:
        b = pos[s]["ball"]["N"]
        starts[s], rows[s] = b.min(), n
        n += int(b.max() - b.min() + 1)
    persons = sorted({k for s in sections for k in pos[s] if k != "ball"})
    P = len(persons)
    x, y, sp = (np.full((n, P), np.nan) for _ in range(3))
    ball = {k: np.full(n, np.nan) for k in ("x", "y", "z", "s", "t")}
    status, poss = np.zeros(n, np.int8), np.zeros(n, np.int8)
    for s in sections:
        off = rows[s] - starts[s]
        b = pos[s]["ball"]
        r = b["N"] + off
        ball["x"][r], ball["y"][r], ball["z"][r] = b["X"] + length / 2, b["Y"] + width / 2, b["Z"]
        ball["s"][r], ball["t"][r] = b["S"] / KMH, b["T"]
        status[r], poss[r] = b["status"], b["poss"]
        for j, key in enumerate(persons):
            if key in pos[s]:
                p = pos[s][key]
                r = p["N"] + off
                ok = (r >= rows[s]) & (r < rows[s] + (pos[s]["ball"]["N"].max() - starts[s] + 1))
                x[r[ok], j], y[r[ok], j], sp[r[ok], j] = p["X"][ok] + length / 2, p["Y"][ok] + width / 2, p["S"][ok] / KMH
    team = np.array([0 if t == home else 1 for t, _ in persons], np.int8)
    half_rows = [rows[s] for s in sections] + [n]
    return dict(x=x, y=y, s=sp, team=team, ball=ball, status=status, poss=poss, half_rows=half_rows, n=n)


def speed_unit_ratio(c):
    """median(S converted) / median(displacement speed): ~1.0 iff S was km/h. Guards the /3.6."""
    v = np.hypot(np.diff(c["x"], axis=0), np.diff(c["y"], axis=0)) * FPS
    s = c["s"][1:]
    m = np.isfinite(v) & np.isfinite(s) & (v > 0.5) & (v < 10)
    return float(np.median(s[m]) / np.median(v[m]))


def ball_gaps(rng, c, p, length, width):
    """Hidden-ball mask from geometry-weighted + uniform gap starts. Never looks at labels."""
    n = c["n"]
    bx, by, bz = c["ball"]["x"], c["ball"]["y"], c["ball"]["z"]
    d = np.hypot(c["x"] - bx[:, None], c["y"] - by[:, None])
    crowd = (np.nan_to_num(d, nan=99) < 3.0).sum(axis=1) >= 3
    in_box = ((bx < 16.5) | (bx > length - 16.5)) & (np.abs(by - width / 2) < 20.16)
    w = 1.0 + 2.0 * (np.nan_to_num(bz) > 1.5) + 2.0 * crowd + 1.0 * in_box
    w /= w.sum()
    hidden = np.zeros(n, bool)
    while hidden.mean() < p["ball_hidden_share"]:
        start = rng.choice(n, p=w) if rng.random() < p["ball_gap_bias"] else rng.integers(n)
        hidden[start:start + lognormal_frames(rng, p["ball_gap_median_s"], GAP_LOG_SIGMA)] = True
    return hidden


def tracklets(rng, c, p):
    """Players -> anonymous tracklet slots with dropouts, identity breaks, duplicates and speed blanks."""
    n, P = c["x"].shape
    vis = np.isfinite(c["x"]) & np.isfinite(c["y"])
    for j in range(P):                                   # occlusion bursts
        n_bursts = rng.poisson(p["dropout_rate_per_s"] * vis[:, j].sum() / FPS)
        for a in rng.integers(0, n, n_bursts):
            vis[a:a + lognormal_frames(rng, p["dropout_median_s"], 0.8), j] = False
    edges = set(c["half_rows"])
    segs = []                                            # (start, end, player, team)
    for j in range(P):
        v = np.r_[False, vis[:, j], False].astype(np.int8)
        for a, b in zip(np.flatnonzero(np.diff(v) == 1), np.flatnonzero(np.diff(v) == -1)):
            cut = a
            while cut < b:                               # identity breaks inside a visible run
                nxt = min(b, cut + max(1, int(rng.exponential(p["tracklet_mean_s"] * FPS))))
                nxt = min([nxt] + [e for e in edges if cut < e < nxt])
                segs.append((cut, nxt, j, c["team"][j]))
                cut = nxt
    ghosts = []
    for a, b, j, t in segs:                              # duplicates: short offset copies
        if rng.random() < p["duplicate_prob"] and b - a > FPS:
            ga = rng.integers(a, b - 1)
            gb = min(b, ga + int(rng.uniform(1, 15) * FPS))
            ghosts.append((ga, gb, j, t, rng.uniform(0.5, 2.0), rng.uniform(0, 2 * np.pi)))
    k = int(rng.integers(RANGES["slots_per_team"][0], RANGES["slots_per_team"][1] + 1))
    slot_team = np.r_[np.zeros(k, np.int8), np.ones(k, np.int8)]
    free_at = np.zeros(2 * k, np.int64)
    S = len(slot_team)
    x, y, s = (np.full((n, S), np.nan) for _ in range(3))
    items = [(a, b, j, t, 0.0, 0.0) for a, b, j, t in segs] + ghosts
    items.sort(key=lambda it: it[0])
    noise = p["position_noise_m"]
    warm = int(round(p["speed_warmup_s"] * FPS))
    for a, b, j, t, off, ang in items:
        free = np.flatnonzero((slot_team == t) & (free_at <= a))
        if not len(free):
            continue                                     # tracker ran out of slots: observation lost
        k_ = rng.choice(free)
        free_at[k_] = b + 1
        nc, sm = p["player_noise_corr_s"] * FPS, p["player_smooth_frames"]
        x[a:b, k_] = smooth_track(c["x"][a:b, j], sm) + off * np.cos(ang) + smooth_noise(rng.normal(0, noise, b - a), nc)
        y[a:b, k_] = smooth_track(c["y"][a:b, j], sm) + off * np.sin(ang) + smooth_noise(rng.normal(0, noise, b - a), nc)
        sp = smooth_track(c["s"][a:b, j], sm)
        sp[:warm] = np.nan
        sp[rng.random(b - a) < p["speed_nan_share"]] = np.nan
        s[a:b, k_] = sp
    return x, y, s, slot_team


def smooth_noise(w, sigma_frames):
    """Low-frequency version of white noise w with the same sd (Gaussian filter, rescaled)."""
    if sigma_frames <= 0 or len(w) < 2:
        return w
    return gaussian_filter1d(w, sigma_frames, mode="nearest") * np.sqrt(2 * np.sqrt(np.pi) * sigma_frames)


def smooth_track(v, sigma_frames):
    """Tracker output smoothing inside one tracklet (never across a break)."""
    return gaussian_filter1d(v, sigma_frames, mode="nearest") if sigma_frames > 0 and len(v) > 1 else v


def smooth_runs(v, sigma_frames):
    """smooth_track over each finite run of v; NaN stays NaN."""
    out = v.copy()
    f = np.r_[False, np.isfinite(v), False].astype(np.int8)
    for a, b in zip(np.flatnonzero(np.diff(f) == 1), np.flatnonzero(np.diff(f) == -1)):
        out[a:b] = smooth_track(v[a:b], sigma_frames)
    return out


def teleport(rng, bx, by, c):
    """In place: short ball jumps onto a random visible player. Returns the drawn rate (per minute of visible ball)."""
    rate = float(rng.uniform(*TELEPORTS_PER_MIN))
    vis = np.flatnonzero(np.isfinite(bx))
    for f in rng.choice(vis, size=rng.poisson(rate * len(vis) / (60 * FPS)), replace=False):
        players = np.flatnonzero(np.isfinite(c["x"][f]))
        if not len(players):
            continue
        j = rng.choice(players)
        e = min(len(bx), f + int(rng.integers(TELEPORT_FRAMES[0], TELEPORT_FRAMES[1] + 1)))
        seen = np.isfinite(bx[f:e])
        bx[f:e][seen], by[f:e][seen] = c["x"][f:e, j][seen], c["y"][f:e, j][seen]
    return rate


def labels(c, control, min_frames):
    """(raw, adjusted) label arrays. D = BallStatus 0; H/A = BallPossession 1/2 (1 = home, per DFL)."""
    raw = np.where(c["status"] == 0, "D", np.where(c["poss"] == 1, "H", np.where(c["poss"] == 2, "A", "D")))
    t = c["ball"]["t"]
    has_ctrl = np.zeros((2, len(raw)), bool)
    for when, team in control:
        i = int(np.searchsorted(t, when))
        if 0 <= i < len(raw) and abs(t[min(i, len(raw) - 1)] - when) < 1.0:
            has_ctrl[team, max(0, i - FPS // 2):i + FPS // 2] = True
    adj = raw.copy()
    i, prev = 0, None
    while i < len(adj):
        j = i
        while j < len(adj) and raw[j] == raw[i]:
            j += 1
        lab = raw[i]
        if lab == "D":
            prev = None
        elif prev is not None and lab != prev and j - i < min_frames and not has_ctrl["HA".index(lab), i:j].any():
            adj[i:j] = prev                              # touch / deflection: possession did not change
        else:
            prev = lab
        i = j
    return raw, adj


def fmt(v, spec):
    return "NaN" if not np.isfinite(v) else spec.format(v)


def write_txt(path, x, y, s, bx, by, bs, length, width):
    with open(path, "w") as f:
        for r in range(len(bx)):
            pl = ";".join("{},{},{}".format(fmt(x[r, k] / length, "{:.5f}"), fmt(y[r, k] / width, "{:.5f}"),
                                            fmt(s[r, k], "{:.6f}")) for k in range(x.shape[1]))
            f.write("{}:{}:{},{},{}\n".format(r + 1, pl, fmt(bx[r] / length, "{:.5f}"), fmt(by[r] / width, "{:.5f}"),
                                               fmt(bs[r], "{:.6f}")))


def write_meta(path, match, slot_team, home_left_first, length, width, half_rows):
    """EPTS metadata in the BAR-ATH layout. Speed channel keeps BAR-ATH's own (inaccurate) 'kph' label: the values
    are m/s in both files, exactly as BAR-ATH delivers them."""
    params = {"first_half_start": half_rows[0], "first_half_end": half_rows[1] - 1,
              "second_half_start": half_rows[1], "second_half_end": half_rows[2] - 1}
    pp = "".join("<ProviderParameter><Name>{}</Name><Value>{}</Value></ProviderParameter>".format(k, v)
                 for k, v in params.items())
    teams = "".join(
        '<Team id="{0}"><Name>{1}</Name><ProviderTeamsParameters><ProviderParameter><Name>attack_direction_first_half'
        '</Name><Value>{2}</Value></ProviderParameter></ProviderTeamsParameters></Team>'.format(
            tid, name, "left_to_right" if right else "right_to_left")
        for tid, name, right in (("team_home", "Home", home_left_first), ("team_away", "Away", not home_left_first)))
    players = "".join(
        '<Player id="{0}" teamId="{1}"><Name>Track {0}</Name><ShirtNumber>{0}</ShirtNumber></Player>'.format(
            i + 1, "team_home" if t == 0 else "team_away") for i, t in enumerate(slot_team))
    chans = "".join('<PlayerChannel channelId="{0}" id="player{1}_{2}" playerId="{1}"/>'.format(c, i + 1, nm)
                    for i in range(len(slot_team)) for c, nm in (("x", "x"), ("y", "y"), ("s", "speed")))
    regs = "".join('<SplitRegister separator=","><PlayerChannelRef playerChannelId="player{0}_x"/>'
                   '<PlayerChannelRef playerChannelId="player{0}_y"/><PlayerChannelRef playerChannelId="player{0}_speed"/>'
                   '</SplitRegister>'.format(i + 1) for i in range(len(slot_team)))
    xml = (
        '<?xml version="1.0" encoding="utf-8"?><main><Metadata><GlobalConfig><FileName>{m}_tactical_atd</FileName>'
        '<TrackingType>Optical</TrackingType><ProviderName>Sportec (damaged)</ProviderName><FrameRate>{fps}.0</FrameRate>'
        '<ProviderGlobalParameters>{pp}<ProviderParameter><Name>data_type</Name><Value>atd</Value></ProviderParameter>'
        '</ProviderGlobalParameters></GlobalConfig><Sessions><Session id="session1"><SessionType>Match</SessionType>'
        '<FieldSize><Width>{L}</Width><Height>{W}</Height></FieldSize></Session></Sessions><Teams>{teams}</Teams>'
        '<Players>{players}</Players><Devices><Device id="device1"><Name>Device 1</Name><Sensors>'
        '<Sensor id="position"><Name>Position</Name><Channels><Channel id="x"><Name>position_x</Name><Unit>normalized'
        '</Unit></Channel><Channel id="y"><Name>position_y</Name><Unit>normalized</Unit></Channel></Channels></Sensor>'
        '<Sensor id="speed"><Name>Speed in kilometers per hour</Name><Channels><Channel id="s"><Name>speed_kph</Name>'
        '<Unit>kph</Unit></Channel></Channels></Sensor></Sensors></Device></Devices><PlayerChannels>{chans}'
        '</PlayerChannels></Metadata><DataFormatSpecifications><DataFormatSpecification separator=":">'
        '<StringRegister name="frameCount"/><SplitRegister separator=";">{regs}</SplitRegister>'
        '<SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/>'
        '<BallChannelRef channelId="s"/></SplitRegister></DataFormatSpecification></DataFormatSpecifications></main>'
    ).format(m=match, fps=FPS, pp=pp, L=length, W=width, teams=teams, players=players, chans=chans, regs=regs)
    root = ET.fromstring(xml)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def seed_for(match):
    return BASE_SEED + zlib.crc32(match.encode()) % 100000


def build(match):
    info, events, positions = match_files(match)
    home, away, length, width = read_info(info)
    left, control = read_events(events, home, away)
    if left.get("secondHalf") == left.get("firstHalf"):
        raise ValueError("{}: same team on the left in both halves".format(match))
    c = assemble(read_positions(positions), home, length, width)
    ratio = speed_unit_ratio(c)
    if not 0.8 < ratio < 1.25:
        raise ValueError("{}: Sportec S is not km/h (ratio {:.2f})".format(match, ratio))

    seed = seed_for(match)
    rng = np.random.default_rng(seed)
    p = {k: float(rng.uniform(*v)) for k, v in RANGES.items() if k != "slots_per_team"}
    rng_t = np.random.default_rng(seed + 2)
    p.update({k: float(rng_t.uniform(*v)) for k, v in TRACKER_RANGES.items()})
    hidden = ball_gaps(rng, c, p, length, width)
    x, y, s, slot_team = tracklets(rng, c, p)
    order = rng.permutation(len(slot_team))              # slot order carries no team block or identity
    x, y, s, slot_team = x[:, order], y[:, order], s[:, order], slot_team[order]
    bn, bc, bsm = p["ball_noise_m"], p["ball_noise_corr_frames"], p["ball_smooth_frames"]
    bx = np.where(hidden, np.nan, smooth_runs(c["ball"]["x"], bsm) + smooth_noise(rng.normal(0, bn, c["n"]), bc))
    by = np.where(hidden, np.nan, smooth_runs(c["ball"]["y"], bsm) + smooth_noise(rng.normal(0, bn, c["n"]), bc))
    bs = np.where(~hidden & (rng.random(c["n"]) < p["ball_speed_kept"]), c["ball"]["s"], np.nan)
    p["teleports_per_min"] = teleport(np.random.default_rng(seed + 1), bx, by, c)

    folder = os.path.join(OUT_DIR, match, "tactical")
    os.makedirs(folder, exist_ok=True)
    write_txt(os.path.join(folder, match + "_tactical_atd.txt"), x, y, s, bx, by, bs, length, width)
    write_meta(os.path.join(folder, match + "_tactical_atd.xml"), match, slot_team, left["firstHalf"] == home,
               length, width, c["half_rows"])

    raw, adj = labels(c, control, int(MIN_CONTROL_S * FPS))
    lab = os.path.join(LABEL_DIR, match)
    os.makedirs(lab, exist_ok=True)
    for name, arr in (("labels_raw.csv", raw), ("labels_adjusted.csv", adj)):
        with open(os.path.join(lab, name), "w") as f:
            f.write("frame,label\n" + "".join("{},{}\n".format(i, v) for i, v in enumerate(arr)))
    meta = {"match": match, "seed": seed, "params": p, "slots": int(len(slot_team)), "n_frames": int(c["n"]),
            "half_rows": [int(v) for v in c["half_rows"]], "pitch": [length, width],
            "speed_unit_ratio_after_kmh_conversion": round(ratio, 3),
            "ball_hidden_share": round(float(hidden.mean()), 4), "min_control_s": MIN_CONTROL_S}
    with open(os.path.join(lab, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def all_matches():
    return sorted(os.path.basename(p).split("DFL-MAT-")[1][:-4]
                  for p in glob.glob(os.path.join(SRC_DIR, "DFL_04_03_positions_raw_observed_*.xml")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", action="append")
    for m in ap.parse_args().match or all_matches():
        print(json.dumps(build(m)), flush=True)
