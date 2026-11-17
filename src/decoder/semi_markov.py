"""Training-only explicit-duration decoder for H/A/D probability sequences."""

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


CLASSES: Tuple[str, ...] = ("H", "A", "D")


@dataclass(frozen=True)
class DurationModel:
    """Log-normal class durations plus training-estimated state transitions."""

    classes: Tuple[str, ...]
    token_frames: int
    duration_mu: np.ndarray
    duration_sigma: np.ndarray
    duration_log_probability: np.ndarray
    initial_log_probability: np.ndarray
    transition_log_probability: np.ndarray

    @property
    def max_duration_tokens(self) -> int:
        return int(self.duration_log_probability.shape[1])


def _period_runs(
    labels: np.ndarray, periods: np.ndarray, classes: Tuple[str, ...]
) -> List[List[Tuple[str, int]]]:
    """Return valid-label chains as (label, frame duration), breaking on unknowns."""

    chains: List[List[Tuple[str, int]]] = []
    edges = np.flatnonzero(periods[1:] != periods[:-1]) + 1
    for start, end in zip(np.r_[0, edges], np.r_[edges, len(labels)]):
        chain: List[Tuple[str, int]] = []
        index = int(start)
        while index < int(end):
            label = str(labels[index])
            run_end = index + 1
            while run_end < int(end) and labels[run_end] == labels[index]:
                run_end += 1
            if label in classes:
                chain.append((label, run_end - index))
            else:
                if chain:
                    chains.append(chain)
                    chain = []
            index = run_end
        if chain:
            chains.append(chain)
    return chains


def fit_duration_model(
    sequences: Iterable[Tuple[Sequence[str], np.ndarray]],
    *,
    token_frames: int = 5,
    classes: Tuple[str, ...] = CLASSES,
) -> DurationModel:
    """Fit all decoder probabilities from training labels and period boundaries."""

    if token_frames <= 0:
        raise ValueError("token_frames must be positive")
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("classes must contain at least two unique labels")
    class_index = {label: index for index, label in enumerate(classes)}
    durations: List[List[int]] = [[] for _ in classes]
    initial = np.ones(len(classes), dtype=np.float64)
    transitions = np.ones((len(classes), len(classes)), dtype=np.float64)
    np.fill_diagonal(transitions, 0.0)

    seen = False
    for raw_labels, raw_periods in sequences:
        labels = np.asarray(raw_labels)
        periods = np.asarray(raw_periods)
        if labels.ndim != 1 or periods.shape != labels.shape or len(labels) == 0:
            raise ValueError("each training sequence needs aligned non-empty labels and periods")
        seen = True
        for chain in _period_runs(labels, periods, classes):
            initial[class_index[chain[0][0]]] += 1
            for position, (label, frames) in enumerate(chain):
                durations[class_index[label]].append(
                    max(1, int(np.ceil(frames / token_frames)))
                )
                if position:
                    previous = class_index[chain[position - 1][0]]
                    current = class_index[label]
                    if previous != current:
                        transitions[previous, current] += 1
    if not seen or any(not values for values in durations):
        raise ValueError("training data must contain every decoder class")

    duration_arrays = [np.asarray(values, dtype=np.float64) for values in durations]
    maximum = int(max(values.max() for values in duration_arrays))
    support = np.arange(1, maximum + 1, dtype=np.float64)
    mus = np.asarray([np.log(values).mean() for values in duration_arrays])
    sigmas = []
    for values in duration_arrays:
        log_values = np.log(values)
        sigma = float(log_values.std(ddof=1)) if len(log_values) > 1 else 0.0
        sigmas.append(sigma if np.isfinite(sigma) and sigma >= 0.1 else 0.1)
    sigmas = np.asarray(sigmas)
    log_probability = np.empty((len(classes), maximum), dtype=np.float64)
    for index, (mu, sigma) in enumerate(zip(mus, sigmas)):
        log_density = (
            -np.log(support * sigma * np.sqrt(2 * np.pi))
            - np.square(np.log(support) - mu) / (2 * sigma * sigma)
        )
        log_probability[index] = log_density - np.logaddexp.reduce(log_density)

    initial /= initial.sum()
    transition_log_probability = np.full_like(transitions, -np.inf)
    for index, row in enumerate(transitions):
        allowed = np.arange(len(classes)) != index
        row_probability = row[allowed] / row[allowed].sum()
        transition_log_probability[index, allowed] = np.log(row_probability)
    return DurationModel(
        classes=classes,
        token_frames=token_frames,
        duration_mu=mus,
        duration_sigma=sigmas,
        duration_log_probability=log_probability,
        initial_log_probability=np.log(initial),
        transition_log_probability=transition_log_probability,
    )


def _decode_tokens(log_probability: np.ndarray, model: DurationModel) -> np.ndarray:
    tokens, class_count = log_probability.shape
    prefix = np.vstack(
        (np.zeros((1, class_count)), np.cumsum(log_probability, axis=0))
    )
    best = np.full((tokens + 1, class_count), -np.inf)
    back_duration = np.zeros((tokens + 1, class_count), dtype=np.int32)
    back_class = np.full((tokens + 1, class_count), -1, dtype=np.int16)

    for end in range(1, tokens + 1):
        duration_count = min(end, model.max_duration_tokens)
        durations = np.arange(1, duration_count + 1)
        starts = end - durations
        for current in range(class_count):
            previous_score = np.full(duration_count, -np.inf)
            previous_class = np.full(duration_count, -1, dtype=np.int16)
            at_start = starts == 0
            previous_score[at_start] = model.initial_log_probability[current]
            after_start = ~at_start
            if after_start.any():
                scores = (
                    best[starts[after_start]]
                    + model.transition_log_probability[:, current][None, :]
                )
                choices = scores.argmax(axis=1)
                previous_score[after_start] = scores[
                    np.arange(len(choices)), choices
                ]
                previous_class[after_start] = choices
            emission = prefix[end, current] - prefix[starts, current]
            candidate = (
                previous_score
                + emission
                + model.duration_log_probability[current, durations - 1]
            )
            choice = int(candidate.argmax())
            best[end, current] = candidate[choice]
            back_duration[end, current] = int(durations[choice])
            back_class[end, current] = int(previous_class[choice])

    labels = np.empty(tokens, dtype=np.int16)
    end = tokens
    current = int(best[tokens].argmax())
    while end > 0:
        duration = int(back_duration[end, current])
        if duration <= 0:
            raise RuntimeError("semi-Markov backtracking found no valid segment")
        labels[end - duration : end] = current
        previous = int(back_class[end, current])
        end -= duration
        if end:
            current = previous
    return labels


def decode(
    probabilities: np.ndarray, periods: np.ndarray, model: DurationModel
) -> np.ndarray:
    """Decode full-rate probabilities without crossing period boundaries."""

    values = np.asarray(probabilities, dtype=np.float64)
    period_values = np.asarray(periods)
    if values.ndim != 2 or values.shape[1] != len(model.classes):
        raise ValueError("probabilities must have shape [frames, classes]")
    if period_values.shape != (len(values),) or len(values) == 0:
        raise ValueError("periods must align with a non-empty probability matrix")
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-4)
    ):
        raise ValueError("probability rows must be finite, non-negative and sum to one")

    output = np.empty(len(values), dtype="U1")
    edges = np.flatnonzero(period_values[1:] != period_values[:-1]) + 1
    for period_start, period_end in zip(
        np.r_[0, edges], np.r_[edges, len(values)]
    ):
        spans = [
            (start, min(start + model.token_frames, int(period_end)))
            for start in range(int(period_start), int(period_end), model.token_frames)
        ]
        token_probability = np.stack(
            [values[start:end].mean(axis=0) for start, end in spans]
        )
        token_labels = _decode_tokens(
            np.log(np.clip(token_probability, 1e-8, 1.0)), model
        )
        for (start, end), label in zip(spans, token_labels):
            output[start:end] = model.classes[int(label)]
    return output
