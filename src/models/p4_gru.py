"""P4: bidirectional GRU over the selected common flat input view.

Each frame's flat features are read as a sequence (one window at a time, never across halftime), so the model
learns its own temporal context instead of P2's hand-built rolling windows. Missing values follow GRU-D-lite
(DECISION_LOG IL-009): standardise with training statistics, NaN -> 0, plus a 0/1 observed column for every
feature that can be missing. Output is P(H), P(A), P(D) per frame for the common Viterbi smoother.

Deterministic on CPU for a given seed (RULES.md rule 3: the submission must regenerate byte-for-byte).
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

CLASSES = np.array(["H", "A", "D"])
IGNORE = -100


def fit_scaler(X: pd.DataFrame):
    """Training-only statistics. Columns that are all-missing in training are dropped."""
    X = X.astype("float64")
    columns = X.columns[X.notna().any()].tolist()
    if not columns:
        raise ValueError("P4 cannot train when every selected feature is missing")
    mean = X[columns].mean()
    std = X[columns].std().replace(0, 1).fillna(1)
    masked = [column for column in columns if X[column].isna().any()]
    return {"columns": columns, "mean": mean.to_numpy(), "std": std.to_numpy(), "masked": masked}


def transform(X: pd.DataFrame, scaler) -> np.ndarray:
    """[frames, features + masks] float32; NaN becomes 0 with its observed flag beside it."""
    values = (X[scaler["columns"]].to_numpy(dtype="float64") - scaler["mean"]) / scaler["std"]
    observed = X[scaler["masked"]].notna().to_numpy(dtype="float32")
    return np.concatenate([np.nan_to_num(values, nan=0.0).astype("float32"), observed], axis=1)


def encode(labels) -> np.ndarray:
    codes = {label: i for i, label in enumerate(CLASSES)}
    return np.array([codes.get(label, IGNORE) for label in labels], dtype=np.int64)


def windows(period: np.ndarray, length: int, stride: int):
    """(start, end) windows of `length` steps that never cross a period change. The last window of each half ends
    at the half's end; a half shorter than `length` is one shorter window."""
    edges = np.flatnonzero(np.diff(period) != 0) + 1
    out = []
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(period)]):
        if b - a <= length:
            out.append((int(a), int(b)))
            continue
        starts = list(range(a, b - length, stride)) + [b - length]
        out += [(int(s), int(s + length)) for s in starts]
    return out


class GRU(nn.Module):
    def __init__(self, n_in, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.inp = nn.Linear(n_in, hidden)
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, len(CLASSES)))

    def forward(self, x):
        return self.out(self.gru(torch.relu(self.inp(x)))[0])


def _batches(seqs, spans, batch_size):
    """Group equal-length windows so every batch is a dense tensor (no padding inside a bidirectional pass)."""
    by_len = {}
    for i, (s, e) in spans:
        by_len.setdefault(e - s, []).append((i, s, e))
    for items in by_len.values():
        for k in range(0, len(items), batch_size):
            chunk = items[k:k + batch_size]
            x = torch.from_numpy(np.stack([seqs[i][0][s:e] for i, s, e in chunk]))
            y = torch.from_numpy(np.stack([seqs[i][1][s:e] for i, s, e in chunk]))
            yield x, y


def _spans(seqs, length, stride):
    return [(i, w) for i, (_, _, period) in enumerate(seqs) for w in windows(period, length, stride)]


def train(seqs, val_seqs, length, seed=0, hidden=64, layers=2, dropout=0.3, lr=1e-3, batch_size=32,
          max_epochs=40, patience=5, log=print):
    """seqs / val_seqs: lists of (x [n, d] float32, y [n] int64 codes, period [n]) at the training frame step.
    Early-stops on validation cross-entropy (raw labels); returns the best model in eval mode."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    model = GRU(seqs[0][0].shape[1], hidden, layers, dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE)
    train_spans = _spans(seqs, length, max(1, length // 2))
    val_spans = _spans(val_seqs, length, length)
    best, best_state, stale = np.inf, None, 0
    for epoch in range(max_epochs):
        model.train()
        order = [train_spans[j] for j in rng.permutation(len(train_spans))]
        for x, y in _batches(seqs, order, batch_size):
            opt.zero_grad()
            loss = loss_fn(model(x).reshape(-1, len(CLASSES)), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for x, y in _batches(val_seqs, val_spans, batch_size):
                n = int((y != IGNORE).sum())
                total += float(loss_fn(model(x).reshape(-1, len(CLASSES)), y.reshape(-1))) * n
                count += n
        val = total / max(count, 1)
        log("epoch {:2d} val_ce {:.4f}".format(epoch, val))
        if val < best - 1e-4:
            best, stale = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    model.p4_epochs_ = epoch + 1
    model.p4_val_ce_ = best
    return model


def predict_proba(model, x, period, length, step):
    """P(H), P(A), P(D) for every frame of a full-rate sequence.

    The model was trained on every `step`-th frame, so each of the `step` phase offsets is run as its own sequence.
    Windows overlap by half; each frame keeps the prediction from the window whose centre is nearest."""
    x, period = np.asarray(x, dtype=np.float32), np.asarray(period)
    out = np.zeros((len(x), len(CLASSES)), dtype=np.float32)
    with torch.no_grad():
        for offset in range(step):
            xs, ps = x[offset::step], period[offset::step]
            proba = np.zeros((len(xs), len(CLASSES)), dtype=np.float32)
            dist = np.full(len(xs), np.inf)
            for s, e in windows(ps, length, max(1, length // 2)):
                p = torch.softmax(model(torch.from_numpy(xs[s:e])[None]), -1)[0].numpy()
                d = np.abs(np.arange(s, e) - (s + e - 1) / 2)
                better = d < dist[s:e]
                proba[s:e][better], dist[s:e][better] = p[better], d[better]
            out[offset::step] = proba
    return out
