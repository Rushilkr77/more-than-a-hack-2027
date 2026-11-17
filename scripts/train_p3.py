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


def log_event(event, **details):
    """Emit one flushed JSON progress record for a long-running training job."""
    print(json.dumps({"event": event, **details}), flush=True)


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


class _ChunkDataset(Dataset):
    def __init__(self, sequences, config):
        self.config = config
        self.sequences = sequences
        self.descriptors = [
            (sequence, core_start)
            for sequence, (_, periods, _) in enumerate(sequences)
            for core_start in p3_tcn.period_core_starts(periods, config.core_frames)
        ]

    def __len__(self):
        return len(self.descriptors)

    def __getitem__(self, index):
        sequence, core_start = self.descriptors[index]
        values, periods, labels = self.sequences[sequence]
        window, valid_targets = p3_tcn.period_chunk(
            values, periods, core_start, self.config.past_frames,
            self.config.future_frames, self.config.core_frames,
        )
        target_labels = np.full(self.config.core_frames, -100, dtype=np.int64)
        target_labels[:valid_targets] = labels[core_start:core_start + valid_targets]
        return torch.from_numpy(window), torch.from_numpy(target_labels)


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _probabilities(model, values, periods, config, device):
    model.eval()
    rows = np.empty((len(values), len(config.classes)), dtype=np.float32)
    with torch.no_grad():
        for core_start in p3_tcn.period_core_starts(periods, config.core_frames):
            window, valid_targets = p3_tcn.period_chunk(
                values, periods, core_start, config.past_frames,
                config.future_frames, config.core_frames,
            )
            logits = p3_tcn.target_chunk_logits(
                model(torch.from_numpy(window[None, ...]).to(device)), config
            )
            rows[core_start:core_start + valid_targets] = torch.softmax(logits, dim=1)[0, :, :valid_targets].T.cpu().numpy()
    return rows


def _mean_metrics(probabilities, data, config, penalty=None):
    rows = []
    for name, proba in probabilities.items():
        labels = np.asarray(config.classes)[proba.argmax(axis=1)]
        if penalty is not None:
            from src.decoder.viterbi import smooth
            labels = smooth(proba, data[name].table["period"], penalty)
        rows.append(sportec.score(data[name].labels["adjusted"], labels))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def temporal_smoothing_loss(logits, labels, tau):
    """MS-TCN truncated log-probability MSE on adjacent valid frames."""
    if (logits.ndim != 3 or labels.ndim != 2
            or logits.shape[0] != labels.shape[0]
            or logits.shape[2] != labels.shape[1]):
        raise ValueError("temporal smoothing needs logits [batch, classes, frames] and labels [batch, frames]")
    if tau <= 0:
        raise ValueError("temporal smoothing tau must be positive")
    valid_pairs = (labels[:, 1:] != -100) & (labels[:, :-1] != -100)
    if not bool(valid_pairs.any()):
        return logits.sum() * 0.0
    log_probability = torch.nn.functional.log_softmax(logits, dim=1)
    difference = log_probability[:, :, 1:] - log_probability[:, :, :-1].detach()
    truncated = torch.clamp(difference.square(), max=float(tau) ** 2)
    return truncated.permute(0, 2, 1)[valid_pairs].mean()


def training_loss(logits, labels, loss_fn, config):
    classification = loss_fn(logits, labels)
    if config.temporal_smoothing_weight == 0:
        return classification
    smoothing = temporal_smoothing_loss(logits, labels, config.temporal_smoothing_tau)
    return classification + config.temporal_smoothing_weight * smoothing


def _train_schema(schema, config, split, data, device, output_dir):
    log_event("schema_preparation_started", schema=schema)
    train_sequences = []
    train_values = []
    for name in split["train"]:
        log_event("train_match_preparing", schema=schema, match=name)
        for match in (data[name], sportec.mirrored(data[name])):
            values, periods = p3_tcn.feature_matrix(match, config, schema)
            train_values.append(values)
            train_sequences.append((values, periods, _label_codes(match.labels["raw"], config)))
    scaler = p3_tcn.P3Scaler.fit(train_values)
    train_sequences = [(scaler.transform(values), periods, labels) for values, periods, labels in train_sequences]

    validation = {}
    validation_sequences = []
    for name in split["validate"]:
        log_event("validation_match_preparing", schema=schema, match=name)
        values, periods = p3_tcn.feature_matrix(data[name], config, schema)
        validation[name] = (scaler.transform(values), periods)
        validation_sequences.append((validation[name][0], periods, _label_codes(data[name].labels["raw"], config)))

    train_loader = DataLoader(_ChunkDataset(train_sequences, config), batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(_ChunkDataset(validation_sequences, config), batch_size=config.batch_size)
    counts = np.bincount(np.concatenate([labels for _, _, labels in train_sequences]), minlength=len(config.classes))
    weights = counts.sum() / (len(config.classes) * counts)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device), ignore_index=-100,
    )
    model = p3_tcn.P3Model(config, schema).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    log_event(
        "schema_training_started",
        schema=schema,
        device=str(device),
        train_targets=sum(len(labels) for _, _, labels in train_sequences),
        validation_targets=sum(len(labels) for _, _, labels in validation_sequences),
        train_chunks=len(train_loader.dataset),
        validation_chunks=len(validation_loader.dataset),
        epochs=config.max_epochs,
        batch_size=config.batch_size,
    )

    best_loss = float("inf")
    best_state = None
    remaining = config.early_stopping_patience
    for epoch in range(config.max_epochs):
        model.train()
        train_loss_total = 0.0
        train_rows = 0
        for windows, labels in train_loader:
            optimizer.zero_grad()
            logits = p3_tcn.target_chunk_logits(model(windows.to(device)), config)
            labels = labels.to(device)
            loss = training_loss(logits, labels, loss_fn, config)
            loss.backward()
            optimizer.step()
            valid_rows = int((labels != -100).sum())
            train_loss_total += float(loss.detach().cpu()) * valid_rows
            train_rows += valid_rows
        model.eval()
        validation_loss_total = 0.0
        validation_rows = 0
        with torch.no_grad():
            for windows, labels in validation_loader:
                logits = p3_tcn.target_chunk_logits(model(windows.to(device)), config)
                labels = labels.to(device)
                loss = training_loss(logits, labels, loss_fn, config)
                valid_rows = int((labels != -100).sum())
                validation_loss_total += float(loss.cpu()) * valid_rows
                validation_rows += valid_rows
        validation_loss = validation_loss_total / validation_rows
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            remaining = config.early_stopping_patience
            improved = True
        else:
            remaining -= 1
            improved = False
        log_event(
            "epoch_finished",
            schema=schema,
            epoch=epoch + 1,
            train_loss=train_loss_total / train_rows,
            validation_loss=validation_loss,
            best_validation_loss=best_loss,
            improved=improved,
            early_stop_remaining=remaining,
        )
        if not improved and remaining == 0:
            log_event("early_stopping", schema=schema, epoch=epoch + 1)
            break
    model.load_state_dict(best_state)

    probabilities = {
        name: _probabilities(model, values, periods, config, device)
        for name, (values, periods) in validation.items()
    }
    penalty, curve = sportec.tune_penalty(probabilities, {name: data[name] for name in split["validate"]})
    raw = _mean_metrics(probabilities, data, config)
    smoothed = _mean_metrics(probabilities, data, config, penalty)
    directory = os.path.join(output_dir, "schema_" + schema[:2])
    p3_tcn.save_artifact(directory, model.cpu(), schema, scaler, penalty)
    for name, proba in probabilities.items():
        np.save(os.path.join(directory, name + "_validation_probabilities.npy"), proba)
        np.save(os.path.join(directory, "proba_" + name + ".npy"), proba)
    result = {
        "schema": schema,
        "artifact_dir": directory,
        "seed": config.seed,
        "device": str(device),
        "torch": torch.__version__,
        "temporal_smoothing_weight": config.temporal_smoothing_weight,
        "temporal_smoothing_tau": config.temporal_smoothing_tau,
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
    log_event(
        "schema_finished",
        schema=schema,
        validation_loss=best_loss,
        spellF1=result["spellF1"],
        macroF1=result["macroF1"],
        switch_penalty=penalty,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema")
    parser.add_argument("--config", default=p3_tcn.CONFIG_PATH)
    args = parser.parse_args()
    config = p3_tcn.load_config(args.config)
    output_dir = os.path.join(ROOT, config.output_dir)
    torch.manual_seed(config.seed)
    split = sportec.split()
    schemas = schemas_to_run(config, args.schema)
    device = _device()
    log_event("training_started", schemas=list(schemas), device=str(device), config=args.config, output_dir=output_dir)
    log_event("sportec_loading_started", matches=split["train"] + split["validate"])
    data = {name: sportec.load(name) for name in split["train"] + split["validate"]}
    log_event("sportec_loading_finished", matches=list(data))
    results = [_train_schema(schema, config, split, data, device, output_dir) for schema in schemas]
    selected = p3_tcn.select_run([result for result in results])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "selection.json"), "w") as file:
        json.dump(selected, file, indent=2)
    log_event("training_finished", selected_schema=selected["schema"], spellF1=selected["spellF1"], macroF1=selected["macroF1"])
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == "__main__":
    main()
