#!/usr/bin/env python3
"""Train P6 (Deep-Sets player encoder + TCN) without opening the test match.

P6 v0 intentionally fixes the simplest unresolved choices: schema 03 (no
generated tags), all observed anonymous players, mean pooling, one ternary
H/A/D head, and the common Viterbi smoother.  Attention, nearest-five and a
two-head loss are later ablations only after this baseline is reproducible.
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.common.sportec import BUNDLES, LABELS, score, split as sportec_split, tune_penalty, write  # noqa: E402
from src.decoder.viterbi import smooth  # noqa: E402
from src.features import classification_inputs as ci  # noqa: E402
from src.features import frame_table as ft  # noqa: E402
from src.features import model_inputs as mi  # noqa: E402
from src.models.p6_hybrid_tcn import HybridBatch, HybridTCN, TrajectorySetTCN  # noqa: E402

CLASSES = np.array(["H", "A", "D"])
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}
VERSION = "sum_tcn_schema03_v0"


@dataclass(frozen=True)
class FlatScaler:
    """Train-only robust scaling; the appended mask preserves missingness."""

    median: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frames: Iterable[pd.DataFrame]) -> "FlatScaler":
        sampled = np.concatenate([frame.to_numpy(dtype=np.float32) for frame in frames], axis=0)
        median = np.nanmedian(sampled, axis=0).astype(np.float32)
        q25, q75 = np.nanpercentile(sampled, (25, 75), axis=0)
        scale = (q75 - q25).astype(np.float32)
        median = np.nan_to_num(median, nan=0.0)
        scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
        return cls(median=median, scale=scale)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.to_numpy(dtype=np.float32)
        observed = np.isfinite(values)
        filled = np.where(observed, values, self.median)
        scaled = np.clip((filled - self.median) / self.scale, -10.0, 10.0)
        return np.concatenate((scaled, observed.astype(np.float32)), axis=1).astype(np.float32)

    def as_dict(self) -> Dict[str, List[float]]:
        return {"median": self.median.tolist(), "scale": self.scale.tolist()}


@dataclass(frozen=True)
class PreparedMatch:
    name: str
    table: pd.DataFrame
    labels: Mapping[str, np.ndarray]
    flat: np.ndarray
    inputs: ci.ClassificationInputs


def _source(match: str) -> Tuple[pd.DataFrame, Mapping[str, np.ndarray], Mapping[str, object]]:
    bundle = os.path.join(BUNDLES, match)
    meta = os.path.join(bundle, "tactical", match + "_tactical_atd.xml")
    args = ft.bundle_args(meta)
    table, slots = ft.bundle_table(bundle)
    labels = {
        kind: pd.read_csv(os.path.join(LABELS, match, "labels_{}.csv".format(kind)))["label"].to_numpy()
        for kind in ("raw", "adjusted")
    }
    if any(len(values) != len(table) for values in labels.values()):
        raise ValueError("{} labels do not align with tracking".format(match))
    return table, slots, {"args": args, "labels": labels}


def fit_scaler(matches: Sequence[str], sample_step: int = 25) -> FlatScaler:
    frames: List[pd.DataFrame] = []
    for match in matches:
        table, slots, metadata = _source(match)
        args = metadata["args"]
        frame = mi.make_inputs(
            table, args["pitch"], mi.SCHEMA_BALL, slots=slots, fps=args["fps"]
        )
        frames.append(frame.iloc[::sample_step])
    return FlatScaler.fit(frames)


def prepare(match: str, scaler: FlatScaler) -> PreparedMatch:
    table, slots, metadata = _source(match)
    args = metadata["args"]
    policy = ci.InputPolicy(mi.SCHEMA_BALL, ci.TemporalMode.OFFLINE_ACAUSAL)
    inputs = ci.make_classification_inputs(table, slots, args["pitch"], policy, fps=args["fps"])
    return PreparedMatch(
        name=match,
        table=table,
        labels=metadata["labels"],
        flat=scaler.transform(inputs.frame_features),
        inputs=inputs,
    )


def _tensor(values: np.ndarray, dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.as_tensor(np.array(values, copy=True, order="C"), dtype=dtype)


def batch_from_ranges(data: PreparedMatch, starts: Sequence[int], frames: int) -> Tuple[HybridBatch, Tensor]:
    ix = np.stack([np.arange(start, start + frames) for start in starts])
    players, ball = data.inputs.players, data.inputs.ball
    team = np.broadcast_to(players.team, (len(starts), len(players.team)))
    batch = HybridBatch(
        flat=_tensor(data.flat[ix]),
        player_values=_tensor(players.values[ix]),
        player_observed=_tensor(players.observed[ix], torch.bool),
        player_inside=_tensor(players.inside_pitch[ix], torch.bool),
        player_age_s=_tensor(players.seconds_since_seen_s[ix]),
        player_keeper=_tensor(players.keeper_proxy[ix], torch.bool),
        player_team=_tensor(team, torch.long),
        ball_values=_tensor(ball.values[ix]),
        ball_observed=_tensor(ball.observed[ix], torch.bool),
        ball_inside=_tensor(ball.inside_pitch[ix], torch.bool),
        ball_since_s=_tensor(ball.seconds_since_seen_s[ix]),
        ball_until_s=_tensor(ball.seconds_until_seen_s[ix]),
    )
    labels = np.vectorize(CLASS_TO_INDEX.__getitem__)(data.labels["raw"][ix])
    return batch, _tensor(labels, torch.long)


def _period_ranges(period: np.ndarray) -> List[Tuple[int, int]]:
    edges = np.flatnonzero(np.diff(period) != 0) + 1
    return list(zip(np.r_[0, edges].astype(int), np.r_[edges, len(period)].astype(int)))


def sample_starts(data: PreparedMatch, frames: int, count: int, rng: np.random.Generator) -> np.ndarray:
    candidates = np.concatenate(
        [
            np.arange(start, end - frames + 1, 5, dtype=int)
            for start, end in _period_ranges(data.inputs.period)
            if end - start >= frames
        ]
    )
    if candidates.size == 0:
        raise ValueError("{} has no period long enough for {} frames".format(data.name, frames))
    return rng.choice(candidates, size=count, replace=count > len(candidates))


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: FlatScaler,
    matches: Sequence[str],
    device: torch.device,
    epoch: int,
    chunk_frames: int,
    chunks_per_match: int,
    batch_size: int,
    target_step: int,
) -> float:
    model.train()
    losses: List[float] = []
    history, future = model.history_frames, model.future_frames
    if chunk_frames <= history + future:
        raise ValueError("chunk_frames must exceed the TCN history and future context")
    rng = np.random.default_rng(1000 + epoch)
    for match in matches:
        data = prepare(match, scaler)
        starts = sample_starts(data, chunk_frames, chunks_per_match, rng)
        for offset in range(0, len(starts), batch_size):
            batch, labels = batch_from_ranges(data, starts[offset:offset + batch_size], chunk_frames)
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            stop = -future if future else None
            logits = model(batch)[:, history:stop:target_step]
            targets = labels[:, history:stop:target_step]
            loss = nn.functional.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        del data
    return float(np.mean(losses))


def predict(model: nn.Module, data: PreparedMatch, device: torch.device, core_frames: int = 2048) -> np.ndarray:
    model.eval()
    history, future = model.history_frames, model.future_frames
    probabilities = np.zeros((len(data.table), 3), dtype=np.float32)
    with torch.no_grad():
        for period_start, period_end in _period_ranges(data.inputs.period):
            for core_start in range(period_start, period_end, core_frames):
                core_end = min(period_end, core_start + core_frames)
                input_start = max(period_start, core_start - history)
                input_end = min(period_end, core_end + future)
                batch, _ = batch_from_ranges(data, [input_start], input_end - input_start)
                logits = model(batch.to(device))[0]
                left, right = core_start - input_start, core_end - input_start
                probabilities[core_start:core_end] = torch.softmax(logits[left:right], dim=-1).cpu().numpy()
    return probabilities


def validate(
    model: nn.Module,
    scaler: FlatScaler,
    matches: Sequence[str],
    device: torch.device,
) -> Tuple[float, float, int, Dict[str, np.ndarray], Dict[str, Tuple[object, object, float]], List[Dict[str, object]], Dict[int, float]]:
    proba: Dict[str, np.ndarray] = {}
    common_data: Dict[str, Tuple[object, object, float]] = {}
    prepared: Dict[str, PreparedMatch] = {}
    for match in matches:
        item = prepare(match, scaler)
        proba[match] = predict(model, item, device)
        prepared[match] = item
        common_data[match] = (item.table, item.labels, 0.0)
    penalty, curve = tune_penalty(proba, common_data)
    rows: List[Dict[str, object]] = []
    for match in matches:
        raw = CLASSES[proba[match].argmax(axis=1)]
        smoothed = smooth(proba[match], prepared[match].inputs.period, penalty)
        rows.append({"match": match, "output": "model only", **score(prepared[match].labels["adjusted"], raw)})
        rows.append({"match": match, "output": "smoothed", **score(prepared[match].labels["adjusted"], smoothed)})
    smoothed_rows = [row for row in rows if row["output"] == "smoothed"]
    return (
        float(np.mean([row["spellF1"] for row in smoothed_rows])),
        float(np.mean([row["macroF1"] for row in smoothed_rows])),
        penalty,
        proba,
        common_data,
        rows,
        curve,
    )


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--chunks-per-match", type=int, default=64)
    parser.add_argument("--chunk-frames", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-step", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-name", default=VERSION)
    parser.add_argument(
        "--architecture",
        choices=("pooling_first", "trajectory_first"),
        default="trajectory_first",
    )
    args = parser.parse_args(argv)
    if min(args.epochs, args.chunks_per_match, args.chunk_frames, args.batch_size, args.target_step) <= 0:
        raise ValueError("all count arguments must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    split = sportec_split()
    device = torch.device("cpu")
    started = time.time()
    output_dir = os.path.join(ROOT, "outputs", "p6", args.run_name)
    scaler = fit_scaler(split["train"])
    flat_features = 2 * len(mi.schema_columns(mi.SCHEMA_BALL))
    model_class = TrajectorySetTCN if args.architecture == "trajectory_first" else HybridTCN
    model = model_class(flat_features=flat_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    os.makedirs(output_dir, exist_ok=True)

    history: List[Dict[str, object]] = []
    best = (-np.inf, -np.inf)
    best_payload = None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model, optimizer, scaler, split["train"], device, epoch, args.chunk_frames,
            args.chunks_per_match, args.batch_size, args.target_step,
        )
        spell_f1, macro_f1, penalty, proba, common_data, rows, curve = validate(
            model, scaler, split["validate"], device
        )
        record = {
            "epoch": epoch,
            "train_loss": round(loss, 6),
            "spellF1": round(spell_f1, 6),
            "macroF1": round(macro_f1, 6),
            "switch_penalty": penalty,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if (spell_f1, macro_f1) > best:
            best = (spell_f1, macro_f1)
            best_payload = (penalty, proba, common_data, rows, curve)
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    if best_payload is None:
        raise RuntimeError("training produced no validation result")
    model.load_state_dict(torch.load(os.path.join(output_dir, "model.pt"), map_location=device, weights_only=True))
    penalty, proba, common_data, rows, curve = best_payload
    for match, values in proba.items():
        raw = CLASSES[values.argmax(axis=1)]
        smoothed = smooth(values, common_data[match][0]["period"], penalty)
        write(os.path.join(output_dir, "raw", match + ".csv"), raw)
        write(os.path.join(output_dir, "smoothed", match + ".csv"), smoothed)
        os.makedirs(os.path.join(output_dir, "proba"), exist_ok=True)
        np.save(os.path.join(output_dir, "proba", match + ".npy"), values)

    meta = {
        "version": args.run_name,
        "input_schema": mi.SCHEMA_BALL,
        "generated_tags": False,
        "temporal_mode": ci.TemporalMode.OFFLINE_ACAUSAL.value,
        "model": (
            "per-object asymmetric TCN + Deep-Sets mean pooling + ternary head"
            if args.architecture == "trajectory_first"
            else "Deep-Sets mean pooling + residual TCN + ternary head"
        ),
        "architecture": args.architecture,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "receptive_field_frames": model.receptive_field_frames,
        "history_frames": model.history_frames,
        "future_frames": model.future_frames,
        "test_matches_loaded": [],
        "best_spellF1": round(best[0], 6),
        "best_macroF1": round(best[1], 6),
        "switch_penalty": penalty,
        "penalty_curve": {str(key): round(value, 6) for key, value in curve.items()},
        "history": history,
        "config": vars(args),
        "seconds": round(time.time() - started, 1),
    }
    with open(os.path.join(output_dir, "scaler.json"), "w", encoding="utf-8") as handle:
        json.dump(scaler.as_dict(), handle)
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

