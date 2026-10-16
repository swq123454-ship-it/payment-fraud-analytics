"""Imbalance handling, temporal cross-validation, and the ablation that shows why.

At a 0.35% base rate, how you resample matters less than *where* you resample.
Two mistakes are endemic in fraud write-ups:

* **Resampling before splitting.** SMOTE creates synthetic minority points by
  interpolating between neighbours. If that happens before the train/validation
  split, synthetic points derived from a validation row's neighbours land in
  training, and the validation score inflates.
* **Random cross-validation on time-ordered data.** Fraud behaviour drifts, so a
  random fold trains on next month and validates on last month.

This module measures both effects on the same data with the same model, so the
size of the self-deception is a number in a table rather than a warning in a
paragraph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config as cfg
from .models import fit_lightgbm


@dataclass(frozen=True)
class Fold:
    """One expanding-window split, expressed as positional indices."""

    index: int
    train_index: np.ndarray
    valid_index: np.ndarray


def expanding_window_folds(
    n_samples: int,
    n_folds: int = cfg.N_TIME_FOLDS,
    min_train_fraction: float = 0.5,
) -> list[Fold]:
    """Forward-chaining folds: train on the past, validate on the next block.

    The first half of the window is always in the training set, so every fold
    has enough positives to fit on. Each later fold trains on strictly more
    history than the one before, which mirrors how the model is retrained in
    production.
    """
    start = int(n_samples * min_train_fraction)
    remaining = n_samples - start
    block = max(remaining // n_folds, 1)
    folds: list[Fold] = []
    for k in range(n_folds):
        train_end = start + k * block
        valid_end = start + (k + 1) * block if k < n_folds - 1 else n_samples
        if valid_end <= train_end:
            continue
        folds.append(
            Fold(
                index=k,
                train_index=np.arange(0, train_end),
                valid_index=np.arange(train_end, valid_end),
            )
        )
    return folds


def build_sampler(name: str, ratio: float | None, y: np.ndarray, seed: int = cfg.SEED):
    """Instantiate the imbalanced-learn sampler for a policy name, if any.

    ``None`` means "do not resample", which covers the two baseline arms. The
    difference between those two arms is a model parameter, not a sampler, and is
    applied where the model is fitted.
    """
    if ratio is None:
        return None
    minority = int(np.asarray(y).sum())
    if minority < 3:
        raise ValueError(f"Only {minority} positives: not enough to resample.")
    if name.startswith("random_undersample"):
        return RandomUnderSampler(sampling_strategy=ratio, random_state=seed)
    if name.startswith("smote_enn"):
        return SMOTEENN(
            sampling_strategy=ratio,
            smote=SMOTE(
                sampling_strategy=ratio,
                k_neighbors=min(5, minority - 1),
                random_state=seed,
            ),
            random_state=seed,
        )
    if name.startswith("smote"):
        return SMOTE(
            sampling_strategy=ratio,
            k_neighbors=min(5, minority - 1),
            random_state=seed,
        )
    raise ValueError(f"Unknown resampling strategy: {name!r}")


def resample(
    X: np.ndarray,
    y: np.ndarray,
    name: str,
    ratio: float | None,
    seed: int = cfg.SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a resampling policy, or return the data untouched."""
    sampler = build_sampler(name, ratio, y, seed)
    if sampler is None:
        return X, y
    X_res, y_res = sampler.fit_resample(X, y)
    return np.asarray(X_res, dtype=float), np.asarray(y_res)


def _evaluate_fold(X_train, y_train, X_valid, y_valid, name, ratio, seed, params):
    X_res, y_res = resample(X_train, y_train, name, ratio, seed)
    model = fit_lightgbm(
        X_res,
        y_res,
        params=params,
        class_weight_balanced=(name == "class_weight"),
        seed=seed,
    )
    score = model.predict_proba(X_valid)[:, 1]
    y_valid = np.asarray(y_valid)
    return {
        "pr_auc": float(average_precision_score(y_valid, score)),
        "roc_auc": float(roc_auc_score(y_valid, score)),
        "train_rows": int(len(y_res)),
        "train_positives": int(y_res.sum()),
        "valid_positives": int(y_valid.sum()),
        "score_mean": float(score.mean()),
    }


def compare_sampling_strategies(
    X: np.ndarray,
    y: np.ndarray,
    strategies: dict[str, float | None] | None = None,
    n_folds: int = cfg.N_TIME_FOLDS,
    seed: int = cfg.SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Rank the resampling policies using honest, in-fold resampling.

    Every fold resamples only its own training window. The validation block is
    never resampled and never seen, so the reported figures are what the policy
    would actually deliver.
    """
    strategies = strategies or cfg.RESAMPLING_STRATEGIES
    folds = expanding_window_folds(len(y), n_folds=n_folds)
    rows = []

    for name, ratio in strategies.items():
        fold_metrics = []
        for fold in folds:
            metrics = _evaluate_fold(
                X[fold.train_index],
                y[fold.train_index],
                X[fold.valid_index],
                y[fold.valid_index],
                name,
                ratio,
                seed,
                cfg.RESAMPLING_STUDY_PARAMS,
            )
            fold_metrics.append(metrics)
        frame = pd.DataFrame(fold_metrics)
        rows.append(
            {
                "strategy": name,
                "target_minority_share": ratio,
                "mean_pr_auc": round(frame["pr_auc"].mean(), 4),
                "sd_pr_auc": round(frame["pr_auc"].std(ddof=0), 4),
                "mean_roc_auc": round(frame["roc_auc"].mean(), 4),
                "mean_train_rows": int(frame["train_rows"].mean()),
                "share_of_data_used": round(frame["train_rows"].mean() / len(y), 4),
                "mean_score": round(frame["score_mean"].mean(), 6),
            }
        )
        if verbose:
            print(f"  [sampling] {name:<26} PR-AUC {rows[-1]['mean_pr_auc']:.4f}  rows {rows[-1]['mean_train_rows']:,}")

    return pd.DataFrame(rows).sort_values("mean_pr_auc", ascending=False).reset_index(drop=True)


def _random_fold_scores(
    X: np.ndarray,
    y: np.ndarray,
    name: str,
    ratio: float | None,
    seed: int,
    position: str = "before_split",
    n_splits: int = 5,
) -> float:
    """Five random stratified folds, under a chosen resampling placement.

    ``position="before_split"`` resamples the whole training window once and then
    draws the folds. That is the mistake: synthetic points interpolated between a
    validation row's neighbours land in training. ``position="inside_fold"``
    resamples only the training part of each fold and scores the untouched
    validation rows, which is the correct placement.
    """
    if position == "before_split" and ratio is not None:
        X, y = resample(X, y, name, ratio, seed)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_index, valid_index in splitter.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        if position == "inside_fold":
            X_train, y_train = resample(X_train, y_train, name, ratio, seed)
        model = fit_lightgbm(
            X_train,
            y_train,
            params=cfg.RESAMPLING_STUDY_PARAMS,
            class_weight_balanced=(name == "class_weight"),
            seed=seed,
        )
        probability = model.predict_proba(X[valid_index])[:, 1]
        scores.append(float(average_precision_score(y[valid_index], probability)))
    return float(np.mean(scores))


def sampling_leakage_ablation(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "smote_5pct",
    ratio: float | None = 0.05,
    n_folds: int = cfg.N_TIME_FOLDS,
    seed: int = cfg.SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Measure the optimism of the protocol, then attribute it to one factor each.

    Four protocols are run on identical data with an identical model and an
    identical resampling policy. Only one thing changes between each adjacent
    pair, which is what makes the differences interpretable:

    ==== ============================= ========================
    #    resampling placement          fold protocol
    ==== ============================= ========================
    A    whole window before folding   random stratified 5-fold
    B    inside each fold              random stratified 5-fold
    C    inside each fold              forward-chaining folds
    D    none                          forward-chaining folds
    ==== ============================= ========================

    * ``A - B`` isolates where the resampling happened, with the folds held fixed.
    * ``B - C`` isolates the fold protocol, with the resampling placement fixed.
    * ``A - C`` is what a reader of a typical tutorial would believe.
    * ``C - D`` is what the resampling machinery is actually worth.
    """
    folds = expanding_window_folds(len(y), n_folds=n_folds)
    baseline = "unweighted_baseline"

    honest_resampled = [
        _evaluate_fold(
            X[f.train_index],
            y[f.train_index],
            X[f.valid_index],
            y[f.valid_index],
            strategy,
            ratio,
            seed,
            cfg.RESAMPLING_STUDY_PARAMS,
        )["pr_auc"]
        for f in folds
    ]
    honest_plain = [
        _evaluate_fold(
            X[f.train_index],
            y[f.train_index],
            X[f.valid_index],
            y[f.valid_index],
            baseline,
            None,
            seed,
            cfg.RESAMPLING_STUDY_PARAMS,
        )["pr_auc"]
        for f in folds
    ]

    values = {
        "A_resample_before_split_random_folds": _random_fold_scores(
            X, y, strategy, ratio, seed, position="before_split"
        ),
        "B_resample_inside_fold_random_folds": _random_fold_scores(X, y, strategy, ratio, seed, position="inside_fold"),
        "C_resample_inside_fold_time_folds": float(np.mean(honest_resampled)),
        "D_no_resampling_time_folds": float(np.mean(honest_plain)),
    }

    descriptions = {
        "A_resample_before_split_random_folds": ("whole window, before folding", "random stratified 5-fold"),
        "B_resample_inside_fold_random_folds": ("inside each fold", "random stratified 5-fold"),
        "C_resample_inside_fold_time_folds": ("inside each fold", f"forward-chaining, {n_folds} folds"),
        "D_no_resampling_time_folds": ("none", f"forward-chaining, {n_folds} folds"),
    }

    table = pd.DataFrame(
        [
            {
                "variant": name,
                "resampling_placement": descriptions[name][0],
                "fold_protocol": descriptions[name][1],
                "mean_pr_auc": round(value, 4),
            }
            for name, value in values.items()
        ]
    )

    a, b, c, d = (values[key] for key in values)
    for label, note, delta in (
        ("A_minus_B_resampling_placement_effect", "where the resampling happened, folds held fixed", a - b),
        ("B_minus_C_random_vs_time_fold_effect", "the fold protocol, resampling placement held fixed", b - c),
        ("A_minus_C_total_optimism", "what a typical tutorial would report", a - c),
        ("C_minus_D_value_of_resampling", "what the resampling machinery is worth", c - d),
    ):
        table.loc[len(table)] = {
            "variant": label,
            "resampling_placement": "-",
            "fold_protocol": note,
            "mean_pr_auc": round(delta, 4),
        }

    if verbose:
        print(table.to_string(index=False))
    return table
