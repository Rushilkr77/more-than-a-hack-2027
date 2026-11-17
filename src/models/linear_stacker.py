"""Small, serializable multinomial stacker over frozen member probabilities."""

import os
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression


CLASSES: Tuple[str, ...] = ("H", "A", "D")


def stack_features(probabilities: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate clipped member log-probabilities after strict validation."""

    if not probabilities:
        raise ValueError("stacking requires at least one probability member")
    arrays = [np.asarray(values, dtype=np.float64) for values in probabilities]
    shape = arrays[0].shape
    if len(shape) != 2 or shape[1] != len(CLASSES):
        raise ValueError("stacking members must have shape [frames, 3]")
    if any(values.shape != shape for values in arrays):
        raise ValueError("stacking members must have identical shapes")
    if any(
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-4)
        for values in arrays
    ):
        raise ValueError("stacking members must contain normalized finite probabilities")
    return np.concatenate(
        [np.log(np.clip(values, 1e-6, 1.0)) for values in arrays], axis=1
    )


@dataclass(frozen=True)
class LinearStacker:
    """Standardized L2 multinomial logistic probability combiner."""

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    classes: Tuple[str, ...] = CLASSES
    regularization_c: float = 1.0
    iterations: int = 0

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: Sequence[str],
        *,
        regularization_c: float = 1.0,
    ) -> "LinearStacker":
        values = np.asarray(features, dtype=np.float64)
        targets = np.asarray(labels)
        if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("stacker features must be a non-empty finite matrix")
        if targets.shape != (len(values),):
            raise ValueError("stacker labels must align with feature rows")
        if regularization_c <= 0:
            raise ValueError("regularization_c must be positive")
        mapping = {label: index for index, label in enumerate(CLASSES)}
        try:
            codes = np.asarray([mapping[str(label)] for label in targets], dtype=np.int64)
        except KeyError as error:
            raise ValueError("unexpected stacker label {!r}".format(error.args[0])) from error
        if set(codes) != set(range(len(CLASSES))):
            raise ValueError("stacker fitting data must contain every class")

        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[~np.isfinite(scale) | (scale == 0)] = 1.0
        standardized = (values - mean) / scale
        estimator = LogisticRegression(
            C=regularization_c,
            penalty="l2",
            solver="lbfgs",
            max_iter=200,
            random_state=0,
        ).fit(standardized, codes)
        if not np.array_equal(estimator.classes_, np.arange(len(CLASSES))):
            raise RuntimeError("stacker estimator returned an unexpected class order")
        return cls(
            mean=mean,
            scale=scale,
            coefficients=estimator.coef_,
            intercept=estimator.intercept_,
            regularization_c=regularization_c,
            iterations=int(estimator.n_iter_.max()),
        )

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1:] != self.mean.shape or not np.isfinite(values).all():
            raise ValueError("stacker inference features must match the fitted finite matrix width")
        logits = ((values - self.mean) / self.scale) @ self.coefficients.T + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        return probability / probability.sum(axis=1, keepdims=True)

    def save(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        np.savez(
            path,
            mean=self.mean,
            scale=self.scale,
            coefficients=self.coefficients,
            intercept=self.intercept,
            classes=np.asarray(self.classes),
            regularization_c=np.asarray(self.regularization_c),
            iterations=np.asarray(self.iterations),
        )

    @classmethod
    def load(cls, path: str) -> "LinearStacker":
        with np.load(path, allow_pickle=False) as values:
            classes = tuple(str(label) for label in values["classes"])
            if classes != CLASSES:
                raise ValueError("saved stacker class order does not match H/A/D")
            return cls(
                mean=values["mean"],
                scale=values["scale"],
                coefficients=values["coefficients"],
                intercept=values["intercept"],
                classes=classes,
                regularization_c=float(values["regularization_c"]),
                iterations=int(values["iterations"]),
            )
