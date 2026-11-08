"""Smart-Tagging-style tags generated from tracking only, identically for BAR-ATH and the Sportec bundles.

`generate_tags(table, slots, pitch, fps)` reads the frame table and slot arrays and nothing else: no labels, no
events, no provider possession, no Metrica tags. The model reads these tags (as the `gtag_*` columns) on every match,
so training (Sportec) and inference (BAR-ATH) see tags made the same way. Metrica's own Smart Tagging stays
smoother-only. Rules and sources: docs/data-layer/SPORTEC.md, section "Generated Smart Tagging".

Constants are rule-book geometry, physical units or values taken from published work. None is fitted to Metrica's
tags or to any label.
"""

import numpy as np
import pandas as pd

from src.data.idsse_smart_tagging import Annotation

HOME, AWAY = 0, 1
TEAM_NAME = {HOME: "Home", AWAY: "Away"}

MARGIN_CONFIDENT_M = 1.0      # nearest-team vote counts only when the nearest player is >= 1 m clearer (contested below)
SMOOTH_FRAMES = 50            # runs < 2 s take the majority of the +-2 s window (Li & Link 2026, k = 50 at 25 Hz)
PHASE_MIN_S = 5.0             # shortest Metrica Smart Tagging interval (format minimum)
TRANSITION_GAIN_M = 15.0      # forward ball progress after a regain that marks a transition (low-confidence code)
TRANSITION_WINDOW_S = 5.0     # the "5-second" regain/counter-press window (Bauer & Anzer 2021)
TRANSITION_TAG_S = 10.0
LONG_BALL_M = 30.0            # long ball: ball travels >= 30 m ...
LONG_BALL_MAX_S = 2.5         # ... within 2.5 s
BALL_MAX_MS = 40.0            # faster than any kick: a tracker jump, not a long ball
BRIDGE_S = 1.0                # ball gaps up to 1 s are bridged for territorial codes
BOX_DEPTH_M, BOX_WIDTH_M = 16.5, 40.32
BOX_MIN_S = 1.0
PLAYERS_IN_BOX_MIN = 3
STILL_M, KICK_M, SLOW_SHARE = 1.0, 5.0, 0.5    # restart: ball still (< 1 m in 1 s), most players slow, then kicked
SETPIECE_GAP_S = 5.0
SETPIECE_CODES = {"Throw-in": 1, "Corner Kick": 2, "Goal Kick": 3, "Free Kick": 4,
                  "Kick Off (Start)": 5, "Kick Off (After Goal)": 6}
SIGNED = ("gtag_poss", "gtag_transition", "gtag_long_ball", "gtag_final_third", "gtag_in_box", "gtag_players_in_box_n")
PHASE_CODES = (("BUILD UP", "DEFENDING IN ATTACKING THIRD"), ("PROGRESSION", "DEFENDING IN MIDDLE THIRD"),
               ("CREATING CHANCES", "DEFENDING IN DEFENSIVE THIRD"))

GTAG_COLUMNS = (
    ("gtag_poss", "int8", "generated: tagger's possession guess (confident nearest team, held through ball gaps, 2 s "
                          "majority smoothing): +1 home, -1 away, 0 unknown. A tracking proxy, not the label"),
    ("gtag_phase", "int8", "generated: in-possession phase of the gtag_poss team while a >=5 s phase tag is active: "
                           "1 build up, 2 progression, 3 creating chances, 0 none"),
    ("gtag_transition", "int8", "generated: attacking transition (regain + >=15 m ball progress within 5 s; 10 s tag): "
                                "+1 home, -1 away, 0 none. Low confidence"),
    ("gtag_long_ball", "int8", "generated: long ball (>=30 m within 2.5 s; 5 s tag) by the gtag_poss team: +1 home, "
                               "-1 away, 0 none"),
    ("gtag_final_third", "int8", "generated: ball in a team's attacking third >=5 s (territorial, possession-free): "
                                 "+1 home's attacking third, -1 away's, 0 none"),
    ("gtag_in_box", "int8", "generated: ball in a team's attacking penalty area >=1 s (tag padded to 5 s): +1 home, "
                            "-1 away, 0 none"),
    ("gtag_players_in_box_n", "int8", "generated: while >=3 tracks of one team are in the opponent box for >=5 s, that "
                                      "count signed (+ home, - away); 0 otherwise. Tracklet duplicates inflate it"),
    ("gtag_setpiece", "int8", "generated: restart from tracking (ball still in the last 3 s, then kicked; open-field free "
                              "kicks also need most players slow) for 5 s; type by location: 1 throw-in, 2 corner, 3 goal kick, 4 free kick, 5 kick-off (start), "
                              "6 kick-off (after goal), 0 none. Team unknown"),
)


def _runs(mask, min_len=1):
    d = np.diff(np.r_[0, np.asarray(mask, np.int8), 0])
    return [(a, b) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a >= min_len]


def _wruns(mask, windows, min_len=1):
    """Runs of `mask` cut at every window edge (so no run spans a period change or the pre-kickoff stretch)."""
    return [(w0 + a, w0 + b) for w0, w1 in windows for a, b in _runs(mask[w0:w1], min_len)]


def _windows(table):
    """Match windows: graded frames after kickoff, split at period changes. Tags never span a window edge."""
    live = table["graded"].to_numpy() & ~table["pre_kickoff"].to_numpy()
    period = table["period"].to_numpy()
    out = []
    for a, b in _runs(live):
        cuts = [a] + [i + 1 for i in range(a, b - 1) if period[i] != period[i + 1]] + [b]
        out += list(zip(cuts[:-1], cuts[1:]))
    return out


def _hold(v, windows, limit=None):
    """Forward-fill NaN inside each window (optionally at most `limit` frames)."""
    out = np.full(len(v), np.nan)
    for a, b in windows:
        out[a:b] = pd.Series(v[a:b]).ffill(limit=limit).to_numpy()
    return out


def possession_guess(table, windows, fps):
    """+0 home / 1 away / -1 unknown per frame, from tracking only."""
    vote = table["nearest_team"].to_numpy().astype(float)
    vote[(vote < 0) | ~(table["ball_margin_m"].to_numpy() >= MARGIN_CONFIDENT_M)] = np.nan
    held = _hold(vote, windows)
    poss = np.where(np.isfinite(held), held, -1).astype(int)
    k = int(round(SMOOTH_FRAMES * fps / 25))
    for a, b in windows:
        seg = poss[a:b]
        for _ in range(2):                                   # short runs absorbed by their neighbourhood
            edges = np.flatnonzero(np.diff(seg)) + 1
            starts, ends = np.r_[0, edges], np.r_[edges, len(seg)]
            for s, e in zip(starts, ends):
                if e - s < k and seg[s] >= 0:
                    win = seg[max(0, s - k):min(len(seg), e + k)]
                    win = win[win >= 0]
                    seg[s:e] = np.bincount(win).argmax() if len(win) else seg[s]
        poss[a:b] = seg
    return poss


def _half_label(period):
    return "1st Half" if period == 1 else "2nd Half"


def _ann(code, a, b, fps, period, team=None, **labels):
    lab = {"Team": TEAM_NAME[team] if team is not None else "N/A", "Half": _half_label(period)}
    lab.update({k.replace("_", " "): v for k, v in labels.items() if v is not None})
    return Annotation("", a / fps, b / fps, code, lab)


def _max_players_label(n):
    return None if n <= 0 else ("7+" if n >= 7 else str(int(n)))


def generate_tags(table, slots, pitch, fps):
    """(annotations sorted by start, {gtag column: array}) from the frame table + slot arrays. Never reads labels."""
    n = len(table)
    length, width = pitch
    windows = _windows(table)
    period = table["period"].to_numpy()
    right = table["home_attacks_right"].to_numpy()
    poss = possession_guess(table, windows, fps)
    cols = {name: np.zeros(n, np.int8) for name, _, _ in GTAG_COLUMNS}
    cols["gtag_poss"] = np.where(poss == HOME, 1, np.where(poss == AWAY, -1, 0)).astype(np.int8)
    sign = {HOME: 1, AWAY: -1}
    tags = []
    adv_home = table["ball_home_adv_m"].to_numpy()
    adv_held = _hold(adv_home, windows)
    adv_bridged = _hold(adv_home, windows, limit=int(BRIDGE_S * fps))
    by = _hold(table["ball_y_m"].to_numpy(), windows, limit=int(BRIDGE_S * fps))
    min_phase = int(PHASE_MIN_S * fps)
    in_window = np.zeros(n, bool)
    win_end = np.zeros(n, int)                               # padded tags stop at their window's end
    for a, b in windows:
        in_window[a:b] = True
        win_end[a:b] = b

    # attacking direction per team and the ball's lateral position seen from the attacker (left = +y facing +x)
    team_right = {HOME: right, AWAY: ~right}

    for T in (HOME, AWAY):
        other = 1 - T
        adv = adv_held if T == HOME else length - adv_held
        third = np.where(np.isfinite(adv), np.minimum(adv // (length / 3), 2), -1)
        for k, (code, mirror) in enumerate(PHASE_CODES):
            for a, b in _wruns((poss == T) & (third == k), windows, min_phase):
                tags += [_ann(code, a, b, fps, period[a], T), _ann(mirror, a, b, fps, period[a], other)]
                cols["gtag_phase"][a:b] = k + 1

        # territorial, possession-free: final third and box are the attacking end of T
        adv_b = adv_bridged if T == HOME else length - adv_bridged
        lat = np.where(team_right[T], by, width - by)
        for a, b in _wruns(adv_b >= 2 * length / 3, windows, min_phase):
            m = np.nanmedian(lat[a:b])
            side = "Left" if m > 2 * width / 3 else ("Right" if m < width / 3 else "Central")
            tags.append(_ann("BALL IN FINAL THIRD", a, b, fps, period[a], T, Side=side,
                             Max_Players_in_the_box=_max_players_label(_in_box(slots, T, right, a, b, pitch).max())))
            cols["gtag_final_third"][a:b] = sign[T]
        box = in_window & (adv_b >= length - BOX_DEPTH_M) & (np.abs(by - width / 2) <= BOX_WIDTH_M / 2)
        for a, b in _wruns(box, windows, int(BOX_MIN_S * fps)):
            b5 = min(max(b, a + min_phase), win_end[a])
            tags.append(_ann("BALL IN THE BOX", a, min(b5, n), fps, period[a], T,
                             Direction_of_ball_entry=_entry_direction(table, a, fps),
                             Max_Players_in_the_box=_max_players_label(_in_box(slots, T, right, a, b, pitch).max())))
            cols["gtag_in_box"][a:min(b5, n)] = sign[T]
        crowd = _in_box(slots, T, right, 0, n, pitch)
        for a, b in _wruns(crowd >= PLAYERS_IN_BOX_MIN, windows, min_phase):
            tags.append(_ann("PLAYERS IN THE BOX", a, b, fps, period[a], T,
                             Max_Players_in_the_box=_max_players_label(crowd[a:b].max())))
            cols["gtag_players_in_box_n"][a:b] = np.clip(sign[T] * crowd[a:b], -127, 127)

    # transitions: possession guess switches to T and T's ball advance gains >= 15 m within 5 s
    w5, w10 = int(TRANSITION_WINDOW_S * fps), int(TRANSITION_TAG_S * fps)
    for a, b in windows:
        seg = poss[a:b]
        for i in np.flatnonzero((seg[1:] != seg[:-1]) & (seg[1:] >= 0) & (seg[:-1] >= 0)) + 1:
            f, T = a + i, int(seg[i])
            adv = adv_held if T == HOME else length - adv_held
            ahead = adv[f:min(b, f + w5)]
            if np.isfinite(adv[f]) and np.isfinite(ahead).any() and np.nanmax(ahead) - adv[f] >= TRANSITION_GAIN_M:
                e = min(b, f + w10)
                tags += [_ann("ATTACKING TRANSITION", f, e, fps, period[f], T),
                         _ann("DEFENSIVE TRANSITION", f, e, fps, period[f], 1 - T)]
                cols["gtag_transition"][f:e] = sign[T]

    # long balls: visible ball travels >= 30 m within 2.5 s at a physical speed
    bx, byv = table["ball_x_m"].to_numpy(), table["ball_y_m"].to_numpy()
    hit = np.zeros(n, bool)
    for lag in range(1, int(LONG_BALL_MAX_S * fps) + 1):
        d = np.hypot(bx[lag:] - bx[:-lag], byv[lag:] - byv[:-lag])
        ok = (d >= LONG_BALL_M) & (d <= BALL_MAX_MS * lag / fps) & (period[lag:] == period[:-lag])
        hit[:-lag] |= np.nan_to_num(ok).astype(bool)
    last = -10 ** 9
    for f in np.flatnonzero(hit & in_window):
        if f - last < min_phase or poss[f] < 0:
            continue
        last = f
        e = min(win_end[f], f + min_phase)
        tags.append(_ann("LONG BALL", f, e, fps, period[f], int(poss[f])))
        cols["gtag_long_ball"][f:e] = sign[int(poss[f])]

    # set pieces: ball still with most players slow, then kicked; type from where it happened
    g = lambda c: table[c].to_numpy()  # noqa: E731
    # A restart is a still ball (any time in the previous 3 s) followed by a kick (within the next 1 s). The three
    # signals rarely peak on the same frame. Players-slow is required only for open-field free kicks, where a still
    # ball is otherwise ambiguous: at goal kicks and throw-ins players keep moving while the ball is placed.
    still = pd.Series(np.nan_to_num(g("ball_range_back_1s_m"), nan=99) < STILL_M).rolling(3 * fps, min_periods=1).max()
    kick = pd.Series(np.nan_to_num(g("ball_range_fwd_1s_m")) > KICK_M)
    slow = pd.Series(np.nan_to_num(g("pl_slow_share")) > SLOW_SHARE).rolling(3 * fps, min_periods=1).max()
    cand = in_window & still.to_numpy().astype(bool) & kick.to_numpy()
    last, kickoffs_seen = -10 ** 9, set()
    for f in np.flatnonzero(cand):
        if f - last < int(SETPIECE_GAP_S * fps):
            continue
        if g("ball_dist_corner_m")[f] <= 2.0:
            kind = "Corner Kick"
        elif g("ball_dist_touchline_m")[f] > -1.0:
            kind = "Throw-in"
        elif g("ball_dist_centre_m")[f] <= 2.0:
            kind = "Kick Off (After Goal)" if period[f] in kickoffs_seen else "Kick Off (Start)"
            kickoffs_seen.add(period[f])
        elif g("ball_dist_goalline_m")[f] > -6.0 and g("ball_dist_goal_m")[f] <= 20.0:
            kind = "Goal Kick"
        elif slow.iloc[f]:
            kind = "Free Kick"
        else:
            continue                                          # still ball in open play, nobody stopped: not a restart
        last = f
        e = min(win_end[f], f + min_phase)
        tags.append(_ann("SET PIECES", f, e, fps, period[f], None, Type=kind))
        cols["gtag_setpiece"][f:e] = SETPIECE_CODES[kind]

    tags.sort(key=lambda t: (t.start_s, t.end_s, t.code, t.labels["Team"]))
    tags = [Annotation(str(i), t.start_s, t.end_s, t.code, t.labels) for i, t in enumerate(tags, start=1)]
    return tags, cols


def _in_box(slots, T, home_right, a, b, pitch):
    """Visible T tracks inside the penalty area T attacks, frames [a, b)."""
    length, width = pitch
    x, y = slots["x"][a:b][:, slots["team"] == T], slots["y"][a:b][:, slots["team"] == T]
    att_right = home_right[a:b] if T == HOME else ~home_right[a:b]
    in_y = np.abs(y - width / 2) <= BOX_WIDTH_M / 2
    in_x = np.where(att_right[:, None], x >= length - BOX_DEPTH_M, x <= BOX_DEPTH_M) & (x >= -1) & (x <= length + 1)
    return (in_x & in_y & np.isfinite(x)).sum(axis=1)


def _entry_direction(table, f, fps):
    """Ball direction over the 0.5 s before box entry: along the pitch length = Horizontal, across = Vertical."""
    a = max(0, f - fps // 2)
    x, y = table["ball_x_m"].to_numpy()[a:f + 1], table["ball_y_m"].to_numpy()[a:f + 1]
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return None
    dx, dy = abs(x[ok][-1] - x[ok][0]), abs(y[ok][-1] - y[ok][0])
    return "Horizontal" if dx >= 2 * dy else ("Vertical" if dy >= 2 * dx else "Diagonal")
