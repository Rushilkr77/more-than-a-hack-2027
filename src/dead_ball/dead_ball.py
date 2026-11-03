#!/usr/bin/env python3
"""DRAFT -- per-frame dead-ball probability for the BAR-ATH tactical tracking (ATD) file.
NOT FOR SUBMISSION as it stands: the final calibration step (cal_*) and the play periods use hand labels / constants
read off the provided game. See README.md in this folder.


Reads   : <match>_atd.txt (frame:players:ball), <match>_atd.xml (teams/GKs), <match>_smart_tagging.xml
Writes  : <out>/frame_scores.csv      one row per frame
          <out>/dead_segments.csv     one row per contiguous dead-ball segment
Pure standard library. Thresholds live in CFG below (documented in README.md).

    python3 dead_ball.py [--data DIR] [--out DIR] [--no-tags] [--eval]
"""
import argparse, array, csv, math, os, re, sys

FPS = 25
PITCH_L, PITCH_W = 105.0, 68.0
NAN = float("nan")

CFG = dict(
    # --- player slowdown (goalkeepers excluded; speed channel is m/s, see SIGNALS.md) ---
    player_half_win=25,        # +-1 s
    slow_speed=1.0,            # m/s: below this a player is "standing/walking"
    fast_speed=3.0,            # m/s: above this a player is "running"
    slow_frac_lo=0.20, slow_frac_hi=0.55,   # ramp 0 -> 1
    fast_frac_hi=0.12, fast_frac_lo=0.02,   # ramp 0 -> 1 (lower is more dead)
    min_player_samples=150,    # valid player speeds in the window (~3 players x 51 frames)
    # --- ball (only used when visible) ---
    ball_half_win=12,          # +-0.5 s
    min_ball_samples=8,        # of 25 frames, else ball signal is "unavailable"
    still_hi=1.0, still_lo=3.0,   # window bbox diagonal in m: <=1 m full, >=3 m none
    live_lo=3.0, live_hi=8.0,     # >=8 m in 1 s is clearly live
    out_lo=0.3, out_hi=1.5,       # m beyond the touch/goal line
    # --- tags ---
    # A SET PIECES tag lasts 35 s (5 s for kick-offs) but marks the STOP, not the whole stoppage: the ball is
    # already out ~2-3 s before tag start. Evidence: ramps up over 6 s before start, full until start+3 s,
    # fades to 0 by start+8 s (or the tag end for kick-offs). After that only the data signals decide.
    tag_pre=6.0, tag_pre_full=1.0, tag_hold=3.0, tag_fade=8.0,
    split_gap_s=12.0,                # two tags this far apart inside one dead run are two stoppages
    # --- fusion ---
    w_tag=0.90, w_still=0.60, w_out=0.50, w_players=0.60, live_penalty=0.75,
    dead_thr=0.50,
    # --- calibration to a probability (v0.5): logistic fit of [evidence_score, mean evidence over +-5 s]
    # against human-labelled stoppages on the provided match. DRAFT: the competition rules forbid hand labels on the
    # provided game, so these coefficients and the play_periods below must be replaced before any submission.
    # The +-5 s mean lets long, quiet stoppages keep a high probability. Team width was dropped (did not transfer).
    cal_score=(0.254, 0.311), cal_m5=(0.253, 0.234), cal_w=(-1.465, 0.519, 0.933), m5_half_win=125,
    width_min_players=8,
    # --- segments ---
    min_seg_s=2.0, merge_gap_s=1.5,
    # --- play periods (video seconds; same clock as the tracking file) ---
    # Read off the broadcast clock/scoreboard in tactical.mp4 and the tracking player count, +-5 s:
    #   1st half: ball moves from the centre spot ~447.5 s (match clock starts ~449 s) .. whistle ~3558 s (clock 51:4x)
    #   half-time ~3558-4549 s (pitch watered, warm-ups, 4-20 tracked players); 2nd half clock 00 ~4551 s
    #   full time before 7625 s (players shaking hands). The tag file labels frames up to 4395 s "1st Half",
    #   so its extremes must NOT be used as period bounds.
    play_periods=((440.0, 3550.0), (4550.0, 7610.0)),
    halftime=(3550.0, 4550.0),   # per the user; frames here get reason_code HALFTIME, no score
    near_player_m=3.0,
)


def ramp(v, v0, v1):
    if v0 == v1:
        return 0.0
    return min(1.0, max(0.0, (v - v0) / (v1 - v0)))


# ----------------------------------------------------------------------------- loading
def load_tracking(path):
    X = array.array("f"); Y = array.array("f"); S = array.array("f")
    bx = array.array("f"); by = array.array("f"); bs = array.array("f")
    for line in open(path):
        _, p, b = line.rstrip("\n").split(":")
        for t in p.split(";"):
            x, y, s = t.split(","); X.append(float(x)); Y.append(float(y)); S.append(float(s))
        x, y, s = b.split(","); bx.append(float(x)); by.append(float(y)); bs.append(float(s))
    return X, Y, S, bx, by


def load_meta(path):
    s = open(path).read()
    teams = {m[0]: m[1] for m in re.findall(r'<Team id="([^"]+)">\s*<Name>([^<]*)</Name>', s)}
    direction = {}
    for tid, d in re.findall(r'<Team id="([^"]+)">.*?attack_direction_first_half</Name>\s*<Value>([^<]*)</Value>', s, re.S):
        direction[teams[tid]] = d
    players = re.findall(r'<Player id="(\d+)" teamId="([^"]+)">(.*?)</Player>', s, re.S)
    slot_team, gk = [], set()
    for pid, tid, body in players:
        slot_team.append(teams[tid])
        if "<Value>Goalkeeper</Value>" in body:
            gk.add(len(slot_team) - 1)
    return slot_team, gk, direction


def load_tags(path):
    s = open(path).read()
    out = []
    for inst in re.findall(r"<instance>(.*?)</instance>", s, re.S):
        g = lambda k: float(re.search(r"<%s>(.*?)</%s>" % (k, k), inst).group(1))
        out.append(dict(id=int(g("ID")), start=g("start"), end=g("end"),
                        code=re.search(r"<code>(.*?)</code>", inst).group(1),
                        L=dict(re.findall(r"<group>(.*?)</group>\s*<text>(.*?)</text>", inst))))
    return out


TYPE_CODE = {"Kick Off (Start)": "TAG_KICKOFF", "Kick Off (After Goal)": "TAG_KICKOFF_AFTER_GOAL",
             "Throw-in": "TAG_THROW_IN", "Goal Kick": "TAG_GOAL_KICK", "Corner Kick": "TAG_CORNER",
             "Free Kick": "TAG_FREE_KICK"}


# ----------------------------------------------------------------------------- scoring
def score(X, Y, S, bx, by, gk, tags, cfg, use_tags=True):
    N = len(bx)
    # tag evidence per frame
    tag_s = array.array("f", [0.0]) * N
    tag_ix = array.array("i", [-1]) * N
    sp = [t for t in tags if t["code"] == "SET PIECES"]
    for k, t in enumerate(sp):
        if not use_tags:
            break
        a0 = int((t["start"] - cfg["tag_pre"]) * FPS); a1 = int((t["start"] - cfg["tag_pre_full"]) * FPS)
        h1 = int((t["start"] + cfg["tag_hold"]) * FPS)
        b1 = int((t["start"] + min(cfg["tag_fade"], t["end"] - t["start"])) * FPS)
        for f in range(max(a0, 0), min(b1, N - 1) + 1):
            v = (f - a0) / max(a1 - a0, 1) if f < a1 else (1.0 if f <= h1 else (b1 - f) / max(b1 - h1, 1))
            if v > tag_s[f]:
                tag_s[f] = v; tag_ix[f] = k
    # player prefix sums (goalkeepers excluded)
    PV = array.array("l", [0]); PS = array.array("l", [0]); PF = array.array("l", [0])
    v = s_ = fa = 0
    for f in range(N):
        base = f * 46
        for j in range(46):
            if j in gk:
                continue
            sp_ = S[base + j]
            if sp_ == sp_:
                v += 1
                if sp_ < cfg["slow_speed"]: s_ += 1
                elif sp_ > cfg["fast_speed"]: fa += 1
        PV.append(v); PS.append(s_); PF.append(fa)
    # team width: std of y (metres) over outfield players at every 5th frame, then +-1 s mean
    WH = []
    for f in range(0, N, 5):
        ys = [Y[f * 46 + j] * PITCH_W for j in range(46) if j not in gk and X[f * 46 + j] == X[f * 46 + j]]
        if len(ys) >= cfg["width_min_players"]:
            m_ = sum(ys) / len(ys); WH.append(math.sqrt(sum((v - m_) ** 2 for v in ys) / len(ys)))
        else:
            WH.append(None)
    out = []
    hp, hb = cfg["player_half_win"], cfg["ball_half_win"]
    for f in range(N):
        a, b = max(0, f - hp), min(N, f + hp + 1)
        nv = PV[b] - PV[a]
        if nv >= cfg["min_player_samples"]:
            slow = (PS[b] - PS[a]) / nv; fast = (PF[b] - PF[a]) / nv
            s_pl = 0.5 * ramp(slow, cfg["slow_frac_lo"], cfg["slow_frac_hi"]) + \
                   0.5 * ramp(fast, cfg["fast_frac_hi"], cfg["fast_frac_lo"])
            p_av = True
        else:
            slow = fast = None; s_pl = 0.0; p_av = False
        a, b = max(0, f - hb), min(N, f + hb + 1)
        n = 0; x0 = y0 = 9.0; x1 = y1 = -9.0; sx = sy = 0.0
        for g in range(a, b):
            x = bx[g]
            if x == x:
                y = by[g]; n += 1; sx += x; sy += y
                if x < x0: x0 = x
                if x > x1: x1 = x
                if y < y0: y0 = y
                if y > y1: y1 = y
        if n >= cfg["min_ball_samples"]:
            rng = math.hypot((x1 - x0) * PITCH_L, (y1 - y0) * PITCH_W)
            mx, my = sx / n * PITCH_L, sy / n * PITCH_W
            d_out = max(0.0, -mx, mx - PITCH_L, -my, my - PITCH_W)
            s_still = ramp(rng, cfg["still_lo"], cfg["still_hi"])
            live = ramp(rng, cfg["live_lo"], cfg["live_hi"])
            s_out = ramp(d_out, cfg["out_lo"], cfg["out_hi"]) if rng < cfg["live_lo"] else 0.0
            b_av = True
        else:
            rng = None; s_still = live = s_out = 0.0; b_av = False
        s_tag = tag_s[f]
        nontag = 1 - (1 - cfg["w_still"] * s_still) * (1 - cfg["w_out"] * s_out) * (1 - cfg["w_players"] * s_pl)
        nontag *= (1 - cfg["live_penalty"] * live)      # negative evidence ONLY when the ball is seen moving
        sc = 1 - (1 - cfg["w_tag"] * s_tag) * (1 - nontag)
        cov = 0.5 * min(1.0, nv / 300.0) + 0.5 * min(1.0, (n if b_av else 0) / 12.0)
        if s_tag >= 0.5:
            cov = max(cov, 0.9)
        conf = cov * (0.5 + 0.5 * abs(2 * sc - 1))
        prob = None; width = None
        out.append(dict(prob=prob, width=width, score=sc, conf=conf, s_tag=s_tag, tag_ix=tag_ix[f], s_still=s_still, s_out=s_out,
                        s_pl=s_pl, live=live, b_av=b_av, p_av=p_av, slow=slow, fast=fast, rng=rng,
                        bpos=(sx / n, sy / n) if b_av else None))
    # +-5 s mean evidence over in-play frames only, then the calibrated probability
    inplay = [any(a * FPS <= f <= b * FPS for a, b in cfg["play_periods"]) for f in range(N)]
    PS_ = [0.0]; PC_ = [0]
    for f in range(N):
        PS_.append(PS_[-1] + (out[f]["score"] if inplay[f] else 0.0)); PC_.append(PC_[-1] + (1 if inplay[f] else 0))
    hw = cfg["m5_half_win"]; b0, b1, b2 = cfg["cal_w"]
    for f in range(N):
        lo_, hi_ = max(0, f - hw), min(N, f + hw + 1); n_ = PC_[hi_] - PC_[lo_]
        m5 = (PS_[hi_] - PS_[lo_]) / n_ if n_ else out[f]["score"]
        zs = (out[f]["score"] - cfg["cal_score"][0]) / cfg["cal_score"][1]; zm = (m5 - cfg["cal_m5"][0]) / cfg["cal_m5"][1]
        out[f]["prob"] = 1 / (1 + math.exp(-(b0 + b1 * zs + b2 * zm))); out[f]["m5"] = m5
    return out, sp


def periods(tags, cfg):
    return list(cfg["play_periods"])


def reason(r, sp, cfg):
    if r["score"] >= cfg["dead_thr"]:
        if r["s_tag"] >= 0.5 and r["tag_ix"] >= 0:
            return TYPE_CODE.get(sp[r["tag_ix"]]["L"].get("Type"), "TAG_SET_PIECE")
        if r["s_still"] >= 0.5 and r["s_pl"] >= 0.5: return "BALL_STILL_PLAYERS_SLOW"
        if r["s_out"] >= 0.5: return "BALL_OUT_OF_PLAY"
        if r["s_still"] >= 0.5: return "BALL_STILL"
        return "PLAYERS_SLOW" if r["b_av"] else "PLAYERS_SLOW_NO_BALL"
    if r["live"] >= 0.5: return "LIVE_BALL_MOVING"
    if not r["b_av"]: return "NO_BALL_NO_EVIDENCE"
    return "LIVE_NO_DEAD_SIGNAL"


# ----------------------------------------------------------------------------- restart team / segments
def make_team_fn(tags, slot_team, gk, direction, X, Y, bx, by, cfg):
    h2_start = min([t["start"] for t in tags if t["L"].get("Half") == "2nd Half"] or [1e9])
    goals = [t for t in tags if t["code"] == "GOALS"]
    teams = sorted(set(slot_team))

    def attacker_of_side(t_s, right):      # team attacking the goal on the right (x~1) if right else left
        flip = t_s >= h2_start
        for tm in teams:
            d = direction[tm]
            goes_right = (d == "left_to_right") != flip
            if goes_right == right:
                return tm
        return ""

    def other(tm):
        return next((t for t in teams if t != tm), "")

    def nearest_team(f, x, y):
        best = None
        for j in range(46):
            if j in gk: continue
            px = X[f * 46 + j]
            if px == px:
                d = math.hypot((px - x) * PITCH_L, (Y[f * 46 + j] - y) * PITCH_W)
                if best is None or d < best[0]: best = (d, j)
        return slot_team[best[1]] if best and best[0] <= cfg["near_player_m"] else ""

    def geometry_type(x, y):
        edge_x = min(x, 1 - x) < 0.08
        if edge_x and abs(y - .5) > .30: return "CORNER_LIKE"
        if edge_x and abs(y - .5) <= .20: return "GOAL_KICK_LIKE"
        if min(y, 1 - y) < 0.04: return "THROW_IN_LIKE"
        return ""

    def fn(tag, t_s, f, spot):
        """returns (restart_team, source)"""
        if tag is not None:
            ty = tag["L"].get("Type")
            team = tag["L"].get("Team", "N/A")
            if team in teams: return team, "tag"
            if ty == "Kick Off (After Goal)":
                for g in goals:
                    if abs(g["end"] - tag["start"]) <= 3.0:
                        return other(g["L"]["Team"]), "goal_tag"
        if spot:
            x, y = spot; ty = geometry_type(x, y)
            if ty == "CORNER_LIKE": return attacker_of_side(t_s, x > .5), "geometry"
            if ty == "GOAL_KICK_LIKE": return other(attacker_of_side(t_s, x > .5)), "geometry"
            tm = nearest_team(f, x, y)
            if tm: return tm, "nearest_player"
        return "", "none"
    return fn, geometry_type


def segments(res, sp, periods_, cfg, team_fn, geo_type, N):
    thr = cfg["dead_thr"]
    inper = [any(a * FPS <= f <= b * FPS for a, b in periods_) for f in range(N)]
    flags = [inper[f] and res[f]["score"] >= thr for f in range(N)]
    raw = []; s = None
    for f in range(N + 1):
        on = f < N and flags[f]
        if on and s is None: s = f
        if not on and s is not None: raw.append([s, f - 1]); s = None
    merged = []
    for a, b in raw:
        if merged and a - merged[-1][1] <= cfg["merge_gap_s"] * FPS: merged[-1][1] = b
        else: merged.append([a, b])
    # a dead run containing several well-separated tags is several stoppages: split just before each later tag
    split = []
    for a, b in merged:
        ks = sorted({res[f]["tag_ix"] for f in range(a, b + 1) if res[f]["s_tag"] >= 0.5})
        cuts = [a]; last = None
        for k in ks:
            st = sp[k]["start"]
            if last is not None and st - last >= cfg["split_gap_s"]:
                cuts.append(max(a + 1, int((st - cfg["tag_pre_full"]) * FPS)))
            last = st
        cuts.append(b + 1)
        split += [[cuts[i], cuts[i + 1] - 1] for i in range(len(cuts) - 1)]
    segs = []
    for a, b in split:
        if (b - a + 1) < cfg["min_seg_s"] * FPS: continue
        tix = [res[f]["tag_ix"] for f in range(a, b + 1) if res[f]["s_tag"] >= 0.5]
        tag = sp[min(tix, key=lambda k: abs(sp[k]["start"] * FPS - a))] if tix else None
        vis = [(f, res[f]["bpos"]) for f in range(a, b + 1) if res[f]["bpos"]]
        stop_xy = vis[0][1] if vis else None
        tail = [v for v in vis if v[0] >= b - 2 * FPS] or vis[-1:]
        restart_xy = (sum(p[0] for _, p in tail) / len(tail), sum(p[1] for _, p in tail) / len(tail)) if tail else None
        rf = tail[len(tail) // 2][0] if tail else b
        team, src = team_fn(tag, a / FPS, rf, restart_xy)
        if team and src not in ("tag", "goal_tag"):
            team += "?"          # geometric guess (~60% right on tagged restarts), not a label
        kind = TYPE_CODE.get(tag["L"].get("Type"), "TAG_SET_PIECE") if tag else (geo_type(*restart_xy) if restart_xy else "") or "STOPPAGE_UNCLASSIFIED"
        segs.append(dict(start=a, end=b, tag=tag, kind=kind, stop_xy=stop_xy, restart_xy=restart_xy, team=team, team_src=src))
    return segs


# ----------------------------------------------------------------------------- main
def evaluate(res, res_nt, sp, tags, periods_, cfg):
    """Weak-label check: tagged restarts = positives, tagged open-play phases (away from restarts) = negatives."""
    N = len(res)
    near = set()
    for t in sp:
        for f in range(int((t["start"] - 15) * FPS), int((t["end"] + 15) * FPS)): near.add(f)
    inper = lambda f: any(a * FPS <= f <= b * FPS for a, b in periods_)
    pos = set()
    for t in sp:
        for f in range(int(t["start"] * FPS), int((t["start"] + 4) * FPS)):
            if inper(f): pos.add(f)
    live_codes = {"PROGRESSION", "ATTACKING TRANSITION", "DEFENSIVE TRANSITION", "LONG BALL", "CREATING CHANCES", "BUILD UP"}
    neg = set()
    for t in tags:
        if t["code"] in live_codes:
            for f in range(int(t["start"] * FPS), int(t["end"] * FPS)): neg.add(f)
    neg = {f for f in neg - near if inper(f)}
    thr = cfg["dead_thr"]
    lines = []
    for name, r in (("full model", res), ("without tags (data signals only)", res_nt)):
        tp = sum(1 for f in pos if f < N and r[f]["score"] >= thr) / len(pos)
        fp = sum(1 for f in neg if f < N and r[f]["score"] >= thr) / len(neg)
        lines.append("%-34s recall on tagged restarts %.1f%%   flagged in tagged open play %.1f%%" % (name, 100 * tp, 100 * fp))
    miss_pos = sum(1 for f in pos if f < N and not res[f]["b_av"]) / len(pos)
    miss_neg = sum(1 for f in neg if f < N and not res[f]["b_av"]) / len(neg)
    ins = [f for f in range(N) if inper(f)]
    miss_all = sum(1 for f in ins if not res[f]["b_av"]) / len(ins)
    lines.append("ball unavailable: tagged restarts %.1f%% | tagged open play %.1f%% | whole file %.1f%%" % (100 * miss_pos, 100 * miss_neg, 100 * miss_all))
    dead = [f for f in ins if res[f]["score"] >= thr]
    miss_dead = sum(1 for f in dead if not res[f]["b_av"]) / max(len(dead), 1)
    lines.append("share of flagged-dead frames with no ball: %.1f%%; P(dead | ball unavailable) = %.1f%% vs P(dead | ball visible) = %.1f%%" % (
        100 * miss_dead,
        100 * sum(1 for f in dead if not res[f]["b_av"]) / max(sum(1 for f in ins if not res[f]["b_av"]), 1),
        100 * sum(1 for f in dead if res[f]["b_av"]) / max(sum(1 for f in ins if res[f]["b_av"]), 1)))
    lines.append("in-play frames %d (%.0f s), flagged dead %d (%.1f%%)" % (len(ins), len(ins) / FPS, len(dead), 100 * len(dead) / len(ins)))
    return lines


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))                      # repository root
    data_root = os.environ.get("MTAH_DATA_DIR") or os.path.join(root, "data")
    ap.add_argument("--data", default=os.path.join(data_root, "tactical"), help="folder holding the three BAR-ATH tactical files")
    ap.add_argument("--out", default=os.path.join(root, "outputs", "dead_ball"), help="output folder (outputs/ is gitignored)")
    ap.add_argument("--no-tags", action="store_true", help="ignore SET PIECES tags (data signals only)")
    ap.add_argument("--eval", action="store_true", help="print weak-label evaluation")
    ap.add_argument("--periods", help="play periods in video seconds, e.g. 440:3558,4540:7612")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = dict(CFG)
    if a.periods:
        cfg["play_periods"] = tuple(tuple(float(v) for v in p.split(":")) for p in a.periods.split(","))
    X, Y, S, bx, by = load_tracking(os.path.join(a.data, "BAR-ATH_tactical_atd.txt"))
    slot_team, gk, direction = load_meta(os.path.join(a.data, "BAR-ATH_tactical_atd.xml"))
    tags = load_tags(os.path.join(a.data, "BAR-ATH_tactical_smart_tagging.xml"))
    N = len(bx)
    print("frames:", N, "| GK slots:", sorted(gk), "| attack dir 1H:", direction, file=sys.stderr)
    res, sp = score(X, Y, S, bx, by, gk, tags, cfg, use_tags=not a.no_tags)
    pr = periods(tags, cfg)
    team_fn, geo_type = make_team_fn(tags, slot_team, gk, direction, X, Y, bx, by, cfg)
    segs = segments(res, sp, pr, cfg, team_fn, geo_type, N)

    seg_of = {}
    for i, s in enumerate(segs, 1):
        for f in range(s["start"], s["end"] + 1): seg_of[f] = i
    with open(os.path.join(a.out, "frame_scores.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time_s", "p_dead", "score_dead", "evidence_score", "confidence", "reason_code", "restart_team", "segment_id", "point", "point_x", "point_y",
                    "kickoff_anchor", "setpiece_tag", "ball_stationary", "players_reset", "no_ball_evidence"])
        for f in range(N):
            t = f / FPS; r = res[f]
            if not any(p0 <= t <= p1 for p0, p1 in pr):
                code = "HALFTIME" if cfg["halftime"][0] < t < cfg["halftime"][1] else "OUT_OF_PERIOD"
                w.writerow([f + 1, "%.2f" % t, "", "", "", "", code] + [""] * 10); continue
            sid = seg_of.get(f, ""); team = point = px = py = ""
            if sid:
                s = segs[sid - 1]; team = s["team"]
                if f == s["start"] and s["stop_xy"]: point, (px, py) = "STOP", s["stop_xy"]
                elif f == s["start"]: point = "STOP"
                if f == s["end"]:
                    point = "RESTART"
                    if s["restart_xy"]: px, py = s["restart_xy"]
            ty = sp[r["tag_ix"]]["L"].get("Type", "") if r["tag_ix"] >= 0 and r["s_tag"] > 0 else ""
            # each flag is a 0-1 strength (not a yes/no)
            flags = ["%.3f" % v for v in (
                r["s_tag"] if ty.startswith("Kick Off") else 0.0,                       # kickoff_anchor: tag evidence ramp
                r["s_tag"] if ty in ("Free Kick", "Corner Kick") else 0.0,              # setpiece_tag: tag evidence ramp
                r["s_still"] if r["b_av"] else 0.0,                                     # ball_stationary: 1 = still, 0 = moving / not visible
                r["s_pl"],                                                              # players_reset: slow/regrouped players
                0.0 if r["b_av"] else 1.0 - max(r["s_tag"], r["s_pl"]))]                # no_ball_evidence: ball missing and nothing else
            w.writerow([f + 1, "%.2f" % t, "%.3f" % r["prob"], "%.3f" % r["prob"], "%.3f" % r["score"], "%.3f" % r["conf"], reason(r, sp, cfg), team, sid, point,
                        "%.3f" % px if px != "" else "", "%.3f" % py if py != "" else ""] + flags)
    with open(os.path.join(a.out, "dead_segments.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["segment_id", "stop_frame", "restart_frame", "stop_time_s", "restart_time_s", "duration_s", "kind", "tag_id",
                    "restart_team", "team_source", "stop_x", "stop_y", "restart_x", "restart_y", "mean_score", "mean_confidence"])
        for i, s in enumerate(segs, 1):
            fs = range(s["start"], s["end"] + 1)
            f2 = lambda p, k: "%.3f" % p[k] if p else ""
            w.writerow([i, s["start"] + 1, s["end"] + 1, "%.2f" % (s["start"] / FPS), "%.2f" % (s["end"] / FPS), "%.1f" % ((s["end"] - s["start"] + 1) / FPS),
                        s["kind"], s["tag"]["id"] if s["tag"] else "", s["team"], s["team_src"],
                        f2(s["stop_xy"], 0), f2(s["stop_xy"], 1), f2(s["restart_xy"], 0), f2(s["restart_xy"], 1),
                        "%.3f" % (sum(res[f]["score"] for f in fs) / len(fs)), "%.3f" % (sum(res[f]["conf"] for f in fs) / len(fs))])
    print("segments:", len(segs), "| dead frames:", sum(1 for f in range(N) if f in seg_of), "->", a.out, file=sys.stderr)
    if a.eval:
        res_nt, _ = score(X, Y, S, bx, by, gk, tags, cfg, use_tags=False)
        for l in evaluate(res, res_nt, sp, tags, pr, cfg): print(l)


if __name__ == "__main__":
    main()
