"""Train P4's bidirectional GRU on the common leak-safe input contract.

Protocol (same common layer as every approach): train on raw labels every `step`-th frame plus a home/away-mirrored
copy of each training match, early-stop on the validation matches, tune the common Viterbi penalty on adjusted
validation labels, score with `sportec.score`, compare variants with `keep_rule`. All choices are made on Sportec
validation only; `--bar-ath` is a report-only sanity check on the public 5 minutes (RULES.md rule 2).

Defaults are the final P4 (docs/models/P4.md): schema 04, gtag off, 30 s window, step 5, 2 extra damaged copies.

    python scripts/build_p4_aug.py --copies 2                     # P4-local damaged training copies (once)
    python scripts/train_p4.py --seed 0 --seed 1 --seed 2         # three seed runs: outputs/p4/04_g0_w30_s5_aug2_seed*
    python scripts/train_p4.py --ensemble --bar-ath               # final form: outputs/p4/04_g0_w30_s5_aug2_ens
    python scripts/train_p4.py --schema 03 --gtag on --aug 0 --window 10 --seed 1   # an ablation variant
    python scripts/train_p4.py --compare 04_g1_w30_s5_aug2_ens 04_g0_w30_s5_aug2_ens
"""

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation import possession_metrics as pm
from src.common import keep_rule
from src.common import sportec
from src.decoder.viterbi import smooth
from src.features import classification_inputs as ci
from src.features import frame_table as ft
from src.features import model_inputs as mi
from src.models import p4_gru
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_p4_aug

OUT = os.path.join(ROOT, "outputs", "p4")
SCHEMAS = {"03": mi.SCHEMA_BALL, "04": mi.SCHEMA_PHASE}
BAR_ATH_META = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_atd.xml")
BAR_ATH_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def load_aug(k, match: sportec.MatchData, name) -> sportec.MatchData:
    """P4-local extra damaged copy k of a training match (scripts/build_p4_aug.py). Labels come from the original
    match: damage never changes them."""
    table, slots = ft.bundle_table(build_p4_aug.bundle_dir(k, name))
    return sportec.MatchData(table=table, slots=slots, labels=match.labels, pitch=match.pitch, fps=match.fps)


def policy(schema):
    return ci.InputPolicy(schema=SCHEMAS[schema], temporal_mode=ci.TemporalMode.OFFLINE_ACAUSAL)


def feats(inputs: ci.ClassificationInputs, gtag: bool) -> pd.DataFrame:
    X = inputs.frame_features
    return X if gtag else X[[column for column in X if not column.startswith("gtag_")]]


def visibility_report(truth, pred, visible):
    """Frame-level slices by ball visibility (SPORTEC.md check A6): live-frame team accuracy, D recall and precision."""
    rows = {}
    for name, keep in (("visible", visible), ("missing", ~visible)):
        t, p = truth[keep], pred[keep]
        live = np.isin(t, ["H", "A"])
        rows[name] = {
            "share": round(float(keep.mean()), 3),
            "live_acc": round(float((t[live] == p[live]).mean()), 3) if live.any() else None,
            "D_recall": round(float((p[t == "D"] == "D").mean()), 3) if (t == "D").any() else None,
            "D_precision": round(float((t[p == "D"] == "D").mean()), 3) if (p == "D").any() else None,
        }
    return rows


def bar_ath(members, args, penalty, save=None):
    """Report-only: frozen model(s) on the public 5-minute BAR-ATH window. Never used to choose anything.
    members: [(model, scaler)]; probabilities are averaged across members (seed ensemble). save: folder for the
    averaged probabilities (scripts/blend.py reads them)."""
    root = ET.parse(BAR_ATH_META).getroot()
    pitch = (float(root.findtext(".//FieldSize/Width")), float(root.findtext(".//FieldSize/Height")))
    table, slots = ft.load(), ft.load_slots()
    inputs = ci.make_classification_inputs(table, slots, pitch, policy(args.schema), fps=25)
    proba = np.mean([p4_gru.predict_proba(model, p4_gru.transform(feats(inputs, args.gtag), scaler), inputs.period,
                                          args.window * 25 // args.step, args.step) for model, scaler in members], axis=0)
    if save:
        np.save(os.path.join(save, "proba_BAR-ATH.npy"), proba)
    gt = pd.read_csv(BAR_ATH_GT)["label"].to_numpy()
    rows = {}
    for output, labels in (("model only", p4_gru.CLASSES[proba.argmax(1)]),
                           ("smoothed", smooth(proba, table["period"], penalty))):
        gt_codes, pred_codes = pm.label_codes(gt), pm.label_codes(labels)
        low, high = pm.bootstrap_spell_f1(gt_codes, pred_codes, pm.valid_runs(gt_codes), 1000)
        rows[output] = {**sportec.score(gt, labels), "spellF1_low": low, "spellF1_high": high}
    return rows


def stem(args):
    """Run name without the seed, e.g. 04_g1_w30_s5_aug2."""
    tag = "_augonly{}".format(args.aug_only) if args.aug_only else "_aug{}".format(args.aug) if args.aug else ""
    if args.hidden != 64:
        tag += "_h{}".format(args.hidden)
    return "{}_g{}_w{}_s{}{}".format(args.schema, int(args.gtag), args.window, args.step, tag)


def run(args, split, data):
    t0 = time.time()
    aug_ks = [args.aug_only] if args.aug_only else list(range(1, args.aug + 1))
    name = "{}_seed{}".format(stem(args), args.seed)
    length = args.window * 25 // args.step

    def sequence(match):
        inputs = sportec.classification_inputs(match, policy(args.schema))
        return feats(inputs, args.gtag), inputs.period

    train_raw = []
    for match_name in split["train"]:
        copies = [data[match_name]] + [load_aug(k, data[match_name], match_name) for k in aug_ks]
        for match in copies + [sportec.mirrored(copy) for copy in copies]:
            X, period = sequence(match)
            train_raw.append((X.iloc[::args.step], match.labels["raw"][::args.step], period[::args.step]))
    scaler = p4_gru.fit_scaler(pd.concat([X for X, _, _ in train_raw], ignore_index=True))
    seqs = [(p4_gru.transform(X, scaler), p4_gru.encode(y), p) for X, y, p in train_raw]

    validation = {}
    for match_name in split["validate"]:
        X, period = sequence(data[match_name])
        validation[match_name] = (p4_gru.transform(X, scaler), period)
    val_seqs = [(x[::args.step], p4_gru.encode(data[m].labels["raw"][::args.step]), p[::args.step])
                for m, (x, p) in validation.items()]

    model = p4_gru.train(seqs, val_seqs, length, seed=args.seed, hidden=args.hidden, layers=args.layers)
    probabilities = {m: p4_gru.predict_proba(model, x, p, length, args.step) for m, (x, p) in validation.items()}

    penalty, curve = sportec.tune_penalty(probabilities, data)
    rows, visibility = [], {}
    for m, proba in probabilities.items():
        raw = p4_gru.CLASSES[proba.argmax(axis=1)]
        smoothed = smooth(proba, data[m].table["period"], penalty)
        sportec.write(os.path.join(OUT, name, "raw", m + ".csv"), raw)
        sportec.write(os.path.join(OUT, name, "smoothed", m + ".csv"), smoothed)
        np.save(os.path.join(OUT, name, "proba_" + m + ".npy"), proba)
        for output, labels in (("model only", raw), ("smoothed", smoothed)):
            rows.append({"version": name, "match": m, "output": output,
                         **sportec.score(data[m].labels["adjusted"], labels)})
        visibility[m] = visibility_report(data[m].labels["adjusted"], smoothed,
                                          data[m].table["ball_visible"].to_numpy(dtype=bool))
    metadata = {
        "version": name, "schema": SCHEMAS[args.schema], "gtag": args.gtag, "window_s": args.window,
        "step": args.step, "aug_copies": aug_ks, "seed": args.seed, "hidden": args.hidden, "layers": args.layers,
        "inputs": seqs[0][0].shape[1], "train_rows": int(sum(len(s[0]) for s in seqs)),
        "epochs": model.p4_epochs_, "val_ce": round(model.p4_val_ce_, 4), "switch_penalty": penalty,
        "penalty_curve": {str(k): round(v, 4) for k, v in curve.items()}, "visibility": visibility,
        "torch": torch.__version__, "seconds": round(time.time() - t0, 1),
    }
    if args.bar_ath:
        metadata["bar_ath_public_report_only"] = bar_ath([(model, scaler)], args, penalty)
    torch.save({"state_dict": model.state_dict(), "scaler": scaler, "meta": metadata},
               os.path.join(OUT, name, "model.pt"))
    with open(os.path.join(OUT, name, "meta.json"), "w") as file:
        json.dump(metadata, file, indent=2, default=float)
    return rows, metadata


def ensemble(args, split, data):
    """Final-submission form: average P(H), P(A), P(D) of the saved per-seed runs, re-tune the common penalty on
    validation, smooth once. Reuses each seed's proba_*.npy and model.pt, so nothing is retrained."""
    seeds = args.seed or [0, 1, 2]
    runs = [os.path.join(OUT, "{}_seed{}".format(stem(args), seed)) for seed in seeds]
    name = stem(args) + ("_ens" if len(seeds) == 3 else "_ens{}".format(len(seeds)))
    probabilities = {m: np.mean([np.load(os.path.join(r, "proba_" + m + ".npy")) for r in runs], axis=0)
                     for m in split["validate"]}
    penalty, curve = sportec.tune_penalty(probabilities, data)
    rows = []
    for m, proba in probabilities.items():
        raw = p4_gru.CLASSES[proba.argmax(axis=1)]
        smoothed = smooth(proba, data[m].table["period"], penalty)
        sportec.write(os.path.join(OUT, name, "raw", m + ".csv"), raw)
        sportec.write(os.path.join(OUT, name, "smoothed", m + ".csv"), smoothed)
        np.save(os.path.join(OUT, name, "proba_" + m + ".npy"), proba)
        for output, labels in (("model only", raw), ("smoothed", smoothed)):
            rows.append({"version": name, "match": m, "output": output,
                         **sportec.score(data[m].labels["adjusted"], labels)})
    metadata = {"version": name, "seeds": list(seeds), "members": [os.path.basename(r) for r in runs],
                "switch_penalty": penalty, "penalty_curve": {str(k): round(v, 4) for k, v in curve.items()},
                "torch": torch.__version__}
    members = []
    for r in runs if args.bar_ath or args.test or args.export else []:
        saved = torch.load(os.path.join(r, "model.pt"), weights_only=False)
        scaler, meta = saved["scaler"], saved["meta"]
        model = p4_gru.GRU(len(scaler["columns"]) + len(scaler["masked"]), meta["hidden"], meta["layers"])
        model.load_state_dict(saved["state_dict"])
        members.append((model.eval(), scaler))
    if args.bar_ath or args.export:
        metadata["bar_ath_public_report_only"] = bar_ath(members, args, penalty, os.path.join(OUT, name))
    if args.export and not args.test:
        # Test-match probabilities for scripts/blend.py. Saved, not scored: blend.py --final scores the test match once.
        for m in split["test"]:
            inputs = sportec.classification_inputs(sportec.load(m), policy(args.schema))
            np.save(os.path.join(OUT, name, "proba_" + m + ".npy"), np.mean(
                [p4_gru.predict_proba(model, p4_gru.transform(feats(inputs, args.gtag), scaler), inputs.period,
                                      args.window * 25 // args.step, args.step) for model, scaler in members], axis=0))
    if args.test:
        # Held-out Sportec test match: frozen models and the validation-tuned penalty; scored once, never tuned on.
        for m in split["test"]:
            match = sportec.load(m)
            inputs = sportec.classification_inputs(match, policy(args.schema))
            proba = np.mean([p4_gru.predict_proba(model, p4_gru.transform(feats(inputs, args.gtag), scaler),
                                                  inputs.period, args.window * 25 // args.step, args.step)
                             for model, scaler in members], axis=0)
            np.save(os.path.join(OUT, name, "proba_" + m + ".npy"), proba)
            raw = p4_gru.CLASSES[proba.argmax(axis=1)]
            smoothed = smooth(proba, match.table["period"], penalty)
            sportec.write(os.path.join(OUT, name, "test", "raw", m + ".csv"), raw)
            sportec.write(os.path.join(OUT, name, "test", "smoothed", m + ".csv"), smoothed)
            for output, labels in (("model only", raw), ("smoothed", smoothed)):
                rows.append({"version": name, "match": m + " (test)", "output": output,
                             **sportec.score(match.labels["adjusted"], labels)})
        metadata["test"] = [row for row in rows if row["match"].endswith("(test)")]
    elif os.path.exists(os.path.join(OUT, name, "meta.json")):
        with open(os.path.join(OUT, name, "meta.json")) as file:      # keep an earlier recorded test score
            metadata.update({k: v for k, v in json.load(file).items() if k == "test"})
    with open(os.path.join(OUT, name, "meta.json"), "w") as file:
        json.dump(metadata, file, indent=2, default=float)
    print(json.dumps(metadata, default=float))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))


def compare(base, candidate, split):
    gains = []
    for m in split["validate"]:
        truth = pm.label_codes(pd.read_csv(os.path.join(sportec.LABELS, m, "labels_adjusted.csv"))["label"])
        predictions = [pm.label_codes(pd.read_csv(os.path.join(OUT, v, "smoothed", m + ".csv"))["label"])
                       for v in (base, candidate)]
        gains.append(keep_rule.chunk_gains(truth, *predictions))
    result = keep_rule.keep_test(np.concatenate(gains))
    print("{} -> {}: {}".format(base, candidate, result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", choices=sorted(SCHEMAS), default="04")
    parser.add_argument("--gtag", type=lambda v: v.lower() in ("1", "on", "true"), default=False)
    parser.add_argument("--window", type=int, default=30, help="window length in seconds")
    parser.add_argument("--step", type=int, default=5, help="train on every step-th frame")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--aug", type=int, default=2, help="extra damaged training copies (build_p4_aug.py)")
    parser.add_argument("--aug-only", type=int, default=0, help="train with this one extra damaged copy only")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--bar-ath", action="store_true", help="report-only public BAR-ATH check")
    parser.add_argument("--ensemble", action="store_true",
                        help="average saved per-seed runs (default seeds 0, 1, 2) instead of training")
    parser.add_argument("--test", action="store_true",
                        help="with --ensemble: score the frozen ensemble on the held-out test match (once, at the end)")
    parser.add_argument("--export", action="store_true",
                        help="with --ensemble: save test-match and BAR-ATH probabilities for scripts/blend.py; the "
                             "test match is not scored")
    parser.add_argument("--compare", nargs=2, metavar=("BASE", "CANDIDATE"))
    args = parser.parse_args()
    split = sportec.split()
    if args.compare:
        return compare(*args.compare, split)
    if args.ensemble:
        return ensemble(args, split, {m: sportec.load(m) for m in split["validate"]})
    data = {m: sportec.load(m) for m in split["train"] + split["validate"]}
    rows = []
    for seed in args.seed or [0]:
        args.seed = seed
        result, metadata = run(args, split, data)
        rows += result
        print(json.dumps({k: v for k, v in metadata.items() if k != "visibility"}, default=float))
        print(json.dumps(metadata["visibility"]))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))


if __name__ == "__main__":
    main()
