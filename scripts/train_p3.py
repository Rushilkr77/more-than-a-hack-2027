"""Train P3's two fixed exact-schema TCN runs; never load the held-out test."""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.common import sportec
from src.models import p3_tcn


OUT = os.path.join(ROOT, "outputs", "p3")


def schemas_to_run(config, schema):
    """Run both approved schemas unless one exact schema was requested."""
    if schema is None:
        return config.schemas
    if schema not in config.schemas:
        raise ValueError("schema {!r} is not configured for P3".format(schema))
    return (schema,)


def _label_codes(labels, config):
    mapping = {label: index for index, label in enumerate(config.classes)}
    try:
        return np.asarray([mapping[label] for label in labels], dtype=np.int64)
    except KeyError as error:
        raise ValueError("unexpected P3 label {!r}".format(error.args[0])) from error


class _WindowDataset(Dataset):
    def __init__(self, sequences, config):
        self.config = config
        self.sequences = sequences
        self.starts = np.cumsum([0] + [len(labels) for _, _, labels in sequences])

    def __len__(self):
        return int(self.starts[-1])

    def __getitem__(self, index):
        sequence = int(np.searchsorted(self.starts, index, side="right") - 1)
        values, periods, labels = self.sequences[sequence]
        target = index - self.starts[sequence]
        window = p3_tcn.target_windows(
            values, periods, [target], self.config.past_frames, self.config.future_frames
        )[0]
        return torch.from_numpy(window), int(labels[target])


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _probabilities(model, values, periods, config, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(values), config.batch_size):
            targets = np.arange(start, min(start + config.batch_size, len(values)))
            windows = p3_tcn.target_windows(
                values, periods, targets, config.past_frames, config.future_frames
            )
            logits = p3_tcn.target_logits(model(torch.from_numpy(windows).to(device)))
            rows.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(rows)


def _mean_metrics(probabilities, data, config, penalty=None):
    rows = []
    for name, proba in probabilities.items():
        labels = np.asarray(config.classes)[proba.argmax(axis=1)]
        if penalty is not None:
            from src.decoder.viterbi import smooth
            labels = smooth(proba, data[name].table["period"], penalty)
        rows.append(sportec.score(data[name].labels["adjusted"], labels))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _train_schema(schema, config, split, data, device):
    train_sequences = []
    train_values = []
    for name in split["train"]:
        for match in (data[name], sportec.mirrored(data[name])):
            values, periods = p3_tcn.feature_matrix(match, config, schema)
            train_values.append(values)
            train_sequences.append((values, periods, _label_codes(match.labels["raw"], config)))
    scaler = p3_tcn.P3Scaler.fit(train_values)
    train_sequences = [(scaler.transform(values), periods, labels) for values, periods, labels in train_sequences]

    validation = {}
    validation_sequences = []
    for name in split["validate"]:
        values, periods = p3_tcn.feature_matrix(data[name], config, schema)
        validation[name] = (scaler.transform(values), periods)
        validation_sequences.append((validation[name][0], periods, _label_codes(data[name].labels["raw"], config)))

    train_loader = DataLoader(_WindowDataset(train_sequences, config), batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(_WindowDataset(validation_sequences, config), batch_size=config.batch_size)
    counts = np.bincount(np.concatenate([labels for _, _, labels in train_sequences]), minlength=len(config.classes))
    weights = counts.sum() / (len(config.classes) * counts)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    model = p3_tcn.P3Model(config, schema).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_loss = float("inf")
    best_state = None
    remaining = config.early_stopping_patience
    for epoch in range(config.max_epochs):
        model.train()
        for windows, labels in train_loader:
            optimizer.zero_grad()
            logits = p3_tcn.target_logits(model(windows.to(device)))
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for windows, labels in validation_loader:
                logits = p3_tcn.target_logits(model(windows.to(device)))
                losses.append(float(loss_fn(logits, labels.to(device)).cpu()))
        validation_loss = float(np.mean(losses))
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            remaining = config.early_stopping_patience
        else:
            remaining -= 1
            if remaining == 0:
                break
    model.load_state_dict(best_state)

    probabilities = {
        name: _probabilities(model, values, periods, config, device)
        for name, (values, periods) in validation.items()
    }
    penalty, curve = sportec.tune_penalty(probabilities, {name: data[name] for name in split["validate"]})
    raw = _mean_metrics(probabilities, data, config)
    smoothed = _mean_metrics(probabilities, data, config, penalty)
    directory = os.path.join(OUT, "schema_" + schema[:2])
    p3_tcn.save_artifact(directory, model.cpu(), schema, scaler, penalty)
    for name, proba in probabilities.items():
        np.save(os.path.join(directory, name + "_validation_probabilities.npy"), proba)
    result = {
        "schema": schema,
        "artifact_dir": directory,
        "validation_loss": best_loss,
        "switch_penalty": penalty,
        "penalty_curve": curve,
        "raw": raw,
        "smoothed": smoothed,
        "spellF1": smoothed["spellF1"],
        "macroF1": smoothed["macroF1"],
    }
    with open(os.path.join(directory, "metrics.json"), "w") as file:
        json.dump(result, file, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema")
    parser.add_argument("--config", default=p3_tcn.CONFIG_PATH)
    args = parser.parse_args()
    config = p3_tcn.load_config(args.config)
    torch.manual_seed(config.seed)
    split = sportec.split()
    data = {name: sportec.load(name) for name in split["train"] + split["validate"]}
    device = _device()
    results = [_train_schema(schema, config, split, data, device) for schema in schemas_to_run(config, args.schema)]
    selected = p3_tcn.select_run([result for result in results])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "selection.json"), "w") as file:
        json.dump(selected, file, indent=2)
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
