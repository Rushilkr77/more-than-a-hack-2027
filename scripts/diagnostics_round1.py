"""Risk-register markers for the Sportec bundles (docs/data-layer/SPORTEC.md), with a FLAG column.

    python scripts/diagnostics_round1.py                          # input-side markers (A2 A3 B1-B8 C1), now
    python scripts/diagnostics_round1.py --pred outputs/p2/smoothed --model-only outputs/p2/raw
                                                                  # + post-training markers (A6 C1 D1 D3 D7)

--pred / --model-only point at folders holding <MATCH>.csv (frame,label) predictions for the Sportec matches.
BAR-ATH appears only as an INPUT reference row for the B markers: compare to find mechanism bugs, never tune numbers
to it. Labels are read only from data/sportec_labels/.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from evaluation import possession_metrics as pm  # noqa: E402
from src.features import frame_table as ft  # noqa: E402

BUNDLES = os.path.join(ROOT, "data", "sportec_bundles")
LABELS = os.path.join(ROOT, "data", "sportec_labels")
SPLIT = os.path.join(ROOT, "configs", "sportec_split.json")


def gap_lengths_s(visible):
    v = np.r_[1, visible.astype(np.int8), 1]
    d = np.diff(v)
    return (np.flatnonzero(d == 1) - np.flatnonzero(d == -1)) / 25.0


def mutual_info_bits(x, y, bins=10):
    """MI between a column (quantile-binned, NaN = own bin) and the label."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    b = np.full(len(x), bins)
    if ok.sum() > bins and np.unique(x[ok]).size > 1:
        edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, bins + 1)[1:-1]))
        b[ok] = np.searchsorted(edges, x[ok])
    joint = pd.crosstab(b, y).to_numpy().astype(float)
    p = joint / joint.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum())


def input_markers(name, table, slots, labels=None):
    r = {"match": name}
    vis = table["ball_visible"].to_numpy()
    gaps = gap_lengths_s(vis)
    r["B8_ball_visible"] = vis.mean()
    r["B8_gap_median_s"] = float(np.median(gaps)) if len(gaps) else np.nan
    n = (table["n_home"].to_numpy(), table["n_away"].to_numpy())
    r["B2_tracks_team_p10"] = float(np.percentile(np.r_[n[0], n[1]], 10))
    r["B2_tracks_team_p50"] = float(np.percentile(np.r_[n[0], n[1]], 50))
    r["B2_tracks_team_max"] = int(max(n[0].max(), n[1].max()))
    r["B2_share_over_13"] = float(((n[0] > 13) | (n[1] > 13)).mean())    # beyond what training damage produces
    r["B2_shape_conf_mean"] = float(table["home_shape_conf"].mean())
    seen = np.isfinite(slots["x"])
    sp = slots["s"][seen]
    r["B1_speed_median_ms"] = float(np.nanmedian(sp))
    r["B1_speed_nan_share"] = float(np.mean(~np.isfinite(sp)))
    bx = table["ball_x_m"].to_numpy()
    r["B3_ball_x_range_m"] = "{:.1f}..{:.1f}".format(np.nanmin(bx), np.nanmax(bx))
    r["dead_ball_conf_mean"] = float(table["dead_ball_conf"].mean())
    if labels is None:
        return r
    lab = labels["raw"]
    for k in "HAD":
        m = lab == k
        r["A3_hidden_{}".format(k)] = float((~vis[m]).mean()) if m.any() else np.nan
    r["A3_hidden_skew"] = max(abs(r["A3_hidden_" + k] - (1 - vis.mean())) for k in "HAD")
    cols = ft.model_columns(list(table.columns))
    r["B5_constant_model_cols"] = ",".join(c for c in cols if table[c].nunique(dropna=False) <= 1) or "-"
    # Leak screen on live frames, H vs A (dead-ball columns legitimately predict D, so D is left out). H/A carries at
    # most 1 bit; honest single columns (team width, nearest_team) reach ~0.1-0.3. A label proxy would approach 1.
    alive = lab != "D"
    mi = {c: mutual_info_bits(table[c].to_numpy()[alive], lab[alive]) for c in cols}
    top = max(mi, key=mi.get)
    r["A2_top_mi_HA"] = "{} {:.2f}".format(top, mi[top])
    r["A2_max_mi_HA_bits"] = mi[top]
    r["A2_nearest_team_mi_HA_bits"] = mi["nearest_team"]
    right = table["home_attacks_right"].to_numpy()
    hx, ax = table["home_cx_m"].to_numpy(), table["away_cx_m"].to_numpy()
    starts = [0, int(np.flatnonzero(np.diff(table["period"].to_numpy()))[0]) + 1]
    r["B7_direction_ok"] = all(((hx[a:a + 50] < ax[a:a + 50]).mean() > 0.5) == bool(right[a]) for a in starts)
    codes = {k: pm.label_codes(v) for k, v in labels.items()}
    runs = pm.valid_runs(codes["raw"])
    for k, c in codes.items():
        st = pm.possession_stats(c, runs)
        r["C1_{}_spells".format(k)], r["C1_{}_mean_s".format(k)] = st["n_spells"], st["mean_duration_s"]
    r["label_share_HAD"] = "/".join("{:.2f}".format((lab == k).mean()) for k in "HAD")
    # A1: the generated-tag possession guess must look like a tracking proxy (~nearest_team quality), not the label
    guess = np.where(table["gtag_poss"].to_numpy() > 0, "H", np.where(table["gtag_poss"].to_numpy() < 0, "A", ""))
    known = alive & (guess != "")
    r["A1_gtag_poss_agree"] = float((guess[known] == lab[known]).mean()) if known.any() else np.nan
    nt = table["nearest_team"].to_numpy()
    ntl = np.where(nt == 0, "H", np.where(nt == 1, "A", ""))
    seen = alive & (ntl != "")
    r["A1_nearest_team_agree"] = float((ntl[seen] == lab[seen]).mean()) if seen.any() else np.nan
    return r


def tag_comparison(table, slots, pitch, fps):
    """F1 (report only, never a target): our generated tags vs Metrica's Smart Tagging on BAR-ATH, per code."""
    from starter import load_smart_tagging
    from src.features import tagger
    gen, _ = tagger.generate_tags(table, slots, pitch, fps)
    team_of = {"barcelona": "Home", "atletic club": "Away", "athletic club": "Away"}
    n = len(table)
    cover = {}
    for t in gen:
        key = (t.code, t.labels.get("Team"))
        cover.setdefault(key, np.zeros(n, bool))[int(t.start_s * fps):int(t.end_s * fps)] = True
    rows = {}
    for e in load_smart_tagging.load():
        code, team = e["code"], team_of.get((e["tags"].get("Team") or "").strip().lower(), "N/A")
        m = cover.get((code, team))
        r = rows.setdefault(code, {"metrica_n": 0, "metrica_median_s": [], "agree": []})
        r["metrica_n"] += 1
        r["metrica_median_s"].append((e["end_frame"] - e["start_frame"]) / fps)
        a, b = max(0, e["start_frame"]), min(n, e["end_frame"])
        r["agree"].append(m[a:b].mean() if m is not None and b > a else 0.0)
    for t in gen:
        r = rows.setdefault(t.code, {"metrica_n": 0, "metrica_median_s": [], "agree": []})
        r.setdefault("ours", []).append(t.end_s - t.start_s)
    out = pd.DataFrame({c: {"metrica_n": r["metrica_n"],
                            "metrica_median_s": np.median(r["metrica_median_s"]) if r["metrica_median_s"] else np.nan,
                            "ours_n": len(r.get("ours", [])),
                            "ours_median_s": np.median(r["ours"]) if r.get("ours") else np.nan,
                            "frame_agree": np.mean(r["agree"]) if r["agree"] else np.nan} for c, r in rows.items()}).T
    return out.sort_index()


def flags(r, ref):
    out = []
    if r.get("A2_max_mi_HA_bits", 0) > 0.6:
        out.append("A2 a column nearly determines H/A (label proxy?)")
    if r.get("A1_gtag_poss_agree", 0) > 0.9:
        out.append("A1 generated possession guess too close to the label")
    if r.get("A3_hidden_skew", 0) > 0.08:
        out.append("A3 gaps skew by label")
    if r.get("B5_constant_model_cols", "-") != "-":
        out.append("B5 constant model column")
    if r.get("B7_direction_ok") is False:
        out.append("B7 attack direction")
    if abs(r["B1_speed_median_ms"] / ref["B1_speed_median_ms"] - 1) > 0.5:
        out.append("B1 speed scale")
    if r["B2_tracks_team_max"] <= 11:
        out.append("B2 no excess tracks")
    if not 0.45 < r["B8_ball_visible"] < 0.70:
        out.append("B8 ball visible share")
    return "; ".join(out) or "ok"


def post_markers(match, table, labels, meta, pred_dir, raw_dir):
    gt = pm.label_codes(labels["adjusted"])
    runs = pm.valid_runs(gt)
    pred = pm.fill_abstentions(pm.label_codes(pm.read_label_csv(os.path.join(pred_dir, match + ".csv"))), runs)
    r = {"match": match, "spellF1": pm.spell_scores(gt, pred, runs)["mean_f1"],
         "macroF1": pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred)))}
    vis = table["ball_visible"].to_numpy()
    for name, m in (("visible", vis), ("hidden", ~vis)):        # A6: frame-level team accuracy by visibility
        alive = m & (gt != pm.D)
        r["A6_acc_" + name] = float((pred[alive] == gt[alive]).mean()) if alive.any() else np.nan
    # B2 gap check: BAR-ATH shows up to 19 tracks per team, our damage at most ~14. If accuracy already drops on
    # frames with ghost tracks (>11 per team), BAR-ATH's heavier ghost bursts will hurt more: widen duplicate_prob /
    # ghost duration in RANGES and rebuild.
    ghost = (table["n_home"].to_numpy() > 11) | (table["n_away"].to_numpy() > 11)
    for name, m in (("ghost", ghost), ("clean", ~ghost)):
        alive = m & (gt != pm.D)
        r["B2_acc_" + name] = float((pred[alive] == gt[alive]).mean()) if alive.any() else np.nan
    r["B2_ghost_frame_share"] = float(ghost.mean())
    st = pm.possession_stats(pred, runs)
    r["C1_pred_spells"], r["C1_pred_mean_s"] = st["n_spells"], st["mean_duration_s"]
    r["D3_hidden_share"], r["D3_gap_bias"] = meta["params"]["ball_hidden_share"], meta["params"]["ball_gap_bias"]
    if raw_dir:                                                  # D7: model alone vs smoothed
        raw = pm.fill_abstentions(pm.label_codes(pm.read_label_csv(os.path.join(raw_dir, match + ".csv"))), runs)
        r["D7_model_only_spellF1"] = pm.spell_scores(gt, raw, runs)["mean_f1"]
    return r


def pre_kickoff_check(table, pred_path):
    """E1: BAR-ATH's second-half wait before kickoff is graded and the evaluator labels it D. Sportec has no such
    stretch, so the model never trained on it. Any non-D there is a free error; fix: force D on `pre_kickoff` frames
    after the smoother (the organisers' definition, nothing learned from BAR-ATH)."""
    pred = pm.read_label_csv(pred_path)["label"].astype(str).str.upper().to_numpy()
    m = table["pre_kickoff"].to_numpy()
    share = float((pred[m] == "D").mean()) if m.any() else float("nan")
    return "E1 pre_kickoff frames {}  predicted D {:.3f}  {}".format(
        int(m.sum()), share, "ok" if share >= 0.99 else "FLAG: force D on pre_kickoff frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred")
    ap.add_argument("--model-only")
    ap.add_argument("--no-reference", action="store_true", help="skip the BAR-ATH input reference row")
    ap.add_argument("--bar-ath-pred", help="a BAR-ATH submission CSV: check E1 (pre-kickoff frames must be D)")
    a = ap.parse_args()
    matches = sorted(os.listdir(BUNDLES))
    rows, cache = [], {}
    ref = None
    if not a.no_reference:
        t, s = ft.load(), ft.load_slots()
        g = t["graded"].to_numpy()                               # match frames only, like the Sportec bundles
        ref = input_markers("BAR-ATH (input ref)", t[g].reset_index(drop=True),
                            {k: (v[g] if k != "team" else v) for k, v in s.items()})
        rows.append(ref)
    for m in matches:
        table, slots = ft.bundle_table(os.path.join(BUNDLES, m))
        labels = {k: pd.read_csv(os.path.join(LABELS, m, "labels_{}.csv".format(k)))["label"].to_numpy()
                  for k in ("raw", "adjusted")}
        with open(os.path.join(LABELS, m, "meta.json")) as f:
            meta = json.load(f)
        cache[m] = (table, labels, meta)
        r = input_markers(m, table, slots, labels)
        r["FLAG"] = flags(r, ref or r)
        rows.append(r)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).set_index("match").T.to_string(float_format=lambda v: "{:.3f}".format(v)))
    if not a.no_reference:
        from games import GAME, GAMES
        print("\nF1 generated tags vs Metrica Smart Tagging on BAR-ATH (report only; never tune the tagger to this)")
        print(tag_comparison(t, s, GAMES[GAME]["pitch"], GAMES[GAME]["fps"]).to_string(float_format=lambda v: "{:.2f}".format(v)))
    if a.bar_ath_pred:
        print(pre_kickoff_check(ft.load(), a.bar_ath_pred))
    if not a.pred:
        return
    post = [post_markers(m, *cache[m], a.pred, a.model_only) for m in matches
            if os.path.isfile(os.path.join(a.pred, m + ".csv"))]
    df = pd.DataFrame(post).set_index("match")
    if os.path.isfile(SPLIT):
        with open(SPLIT) as f:
            split = json.load(f)
        role = {m: k for k, ms in split.items() if isinstance(ms, list) for m in ms}
        df["role"] = [role.get(m, "?") for m in df.index]
        v, t = df.loc[df.role == "validate", "spellF1"].mean(), df.loc[df.role == "test", "spellF1"].mean()
        print("D1 validate {:.3f} vs test {:.3f}  gap {:+.3f}{}".format(v, t, t - v, "  FLAG" if abs(t - v) > 0.04 else ""))
    if len(df) > 2:
        print("D3 corr(spellF1, hidden share) {:+.2f}  corr(spellF1, gap bias) {:+.2f}".format(
            df.spellF1.corr(df.D3_hidden_share), df.spellF1.corr(df.D3_gap_bias)))
    df["B2_FLAG"] = (df.B2_acc_ghost < df.B2_acc_clean - 0.05).map({True: "ghost frames hurt: widen duplicates", False: "ok"})
    df["A6_FLAG"] =(df.A6_acc_hidden >= df.A6_acc_visible - 0.01).map({True: "hidden≈visible: check leak", False: "ok"})
    print(df.T.to_string(float_format=lambda v: "{:.3f}".format(v)))


if __name__ == "__main__":
    main()
