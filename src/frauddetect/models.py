"""Model family: supervised baselines, gradient boosting, unsupervised challengers.

Four models are fitted, and they answer different questions:

* **Logistic regression** - the transparent baseline. In a regulated review the
  first question is always "what does the simple thing get?", because a linear
  model can be handed to a risk committee and defended line by line.
* **LightGBM** - the performance ceiling on tabular transaction data.
* **Isolation forest** - what an unsupervised detector recovers with no labels
  at all. Relevant because fraud labels arrive weeks late, so a system has to
  work on day one before any labels exist.
* **Autoencoder** - the same question asked with a reconstruction objective,
  trained only on legitimate traffic.

Reporting the unsupervised arms honestly matters: on payment fraud they are
several times worse than a supervised model, and a portfolio that quietly omits
that comparison tells a reviewer the author has not run one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from . import config as cfg


def scale_pos_weight(y: np.ndarray) -> float:
    """Ratio of negatives to positives, LightGBM's knob for class weighting."""
    y = np.asarray(y)
    positives = float((y == 1).sum())
    negatives = float((y == 0).sum())
    if positives == 0:
        raise ValueError("Cannot compute scale_pos_weight with no positive labels.")
    return negatives / positives


def fit_logistic(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray,
    class_weight: str | None = "balanced",
    max_iter: int = 1500,
    seed: int = cfg.SEED,
) -> LogisticRegression:
    """L2 logistic regression on the imputed, standardised design matrix."""
    model = LogisticRegression(
        C=1.0,
        class_weight=class_weight,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def fit_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: dict | None = None,
    class_weight_balanced: bool = False,
    seed: int = cfg.SEED,
) -> LGBMClassifier:
    """Gradient-boosted trees with native categorical handling.

    ``class_weight_balanced`` is exposed rather than baked in: reweighting the
    loss changes the ranking only marginally but changes the *calibration*
    substantially, and the cost-optimal threshold depends on the calibration.
    The resampling study measures that difference instead of assuming it.
    """
    merged = dict(cfg.LIGHTGBM_PARAMS)
    if params:
        merged.update(params)
    merged["random_state"] = seed
    if class_weight_balanced:
        merged["scale_pos_weight"] = scale_pos_weight(y_train)
    model = LGBMClassifier(**merged)
    model.fit(X_train, y_train)
    return model


@dataclass
class UnsupervisedDetector:
    """Anomaly score wrapper with the same ``predict_proba`` shape as a classifier.

    Both challengers rank transactions by how unlike legitimate traffic they
    look. The scores are not probabilities and are never fed to the cost model
    directly, so ``predict_proba`` returns them normalised into [0, 1] purely so
    that the shared ranking metrics can consume them.
    """

    name: str
    score_fn: callable

    def predict_proba(self, X) -> np.ndarray:
        raw = self.score_fn(X)
        low, high = float(np.min(raw)), float(np.max(raw))
        scaled = (raw - low) / (high - low) if high > low else np.zeros_like(raw)
        return np.column_stack([1.0 - scaled, scaled])


def fit_isolation_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    max_samples: int = 40_000,
    seed: int = cfg.SEED,
) -> UnsupervisedDetector:
    """Isolation forest fitted on legitimate training traffic only.

    Training it on all traffic would let a handful of frauds define what "normal"
    looks like, which is exactly backwards for a rare-event problem.
    """
    X = np.asarray(X_train, dtype=float)
    legitimate = X[np.asarray(y_train) == 0]
    if len(legitimate) > max_samples:
        rng = np.random.default_rng(seed)
        legitimate = legitimate[rng.choice(len(legitimate), max_samples, replace=False)]

    detector = IsolationForest(
        n_estimators=300,
        max_samples=min(len(legitimate), 8192),
        contamination=cfg.DEFAULT_FRAUD_RATE,
        random_state=seed,
        n_jobs=-1,
    )
    detector.fit(legitimate)

    def score(X_input) -> np.ndarray:
        return -detector.score_samples(np.asarray(X_input, dtype=float))

    return UnsupervisedDetector(name="isolation_forest", score_fn=score)


def fit_autoencoder(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    max_samples: int = 40_000,
    hidden: tuple[int, ...] = (32, 8, 32),
    seed: int = cfg.SEED,
) -> tuple[UnsupervisedDetector, StandardScaler]:
    """Reconstruction-error detector, trained to reproduce legitimate traffic.

    A multilayer perceptron asked to output its own input through a narrow
    bottleneck becomes a nonlinear PCA. Fraudulent transactions, absent from the
    training set, reconstruct worse and therefore score higher.
    """
    X = np.asarray(X_train, dtype=float)
    legitimate = X[np.asarray(y_train) == 0]
    if len(legitimate) > max_samples:
        rng = np.random.default_rng(seed)
        legitimate = legitimate[rng.choice(len(legitimate), max_samples, replace=False)]

    scaler = StandardScaler().fit(legitimate)
    scaled = scaler.transform(legitimate)

    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        batch_size=512,
        max_iter=150,
        early_stopping=True,
        n_iter_no_change=5,
        validation_fraction=0.1,
        random_state=seed,
    )
    model.fit(scaled, scaled)

    def score(X_input) -> np.ndarray:
        transformed = scaler.transform(np.asarray(X_input, dtype=float))
        reconstructed = model.predict(transformed)
        return np.mean((transformed - reconstructed) ** 2, axis=1)

    return UnsupervisedDetector(name="autoencoder", score_fn=score), scaler


def fit_models_for_resampling_study(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = cfg.SEED,
) -> LGBMClassifier:
    """A deliberately small booster, used only inside the sampling sweep.

    The sweep runs four policies across four folds, so the per-fit cost is
    multiplied by sixteen. The conclusion it produces is a *ranking* of the
    policies, and that ranking is stable across hyper-parameters, so a reduced
    configuration is the right trade rather than an oversight.
    """
    return fit_lightgbm(X_train, y_train, params=cfg.RESAMPLING_STUDY_PARAMS, seed=seed)
