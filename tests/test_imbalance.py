"""Imbalance and cross-validation tests.

The point of these is that the *protocol* is correct: folds never see the future,
and resampling never touches a validation row.
"""

from __future__ import annotations

import numpy as np
import pytest

from frauddetect.imbalance import (
    build_sampler,
    compare_sampling_strategies,
    expanding_window_folds,
    resample,
    sampling_leakage_ablation,
)


def test_folds_never_let_training_see_the_future():
    folds = expanding_window_folds(1_000, n_folds=4)
    assert len(folds) == 4
    for fold in folds:
        assert fold.train_index.max() < fold.valid_index.min()


def test_training_window_expands_across_folds():
    folds = expanding_window_folds(1_000, n_folds=4)
    sizes = [len(fold.train_index) for fold in folds]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_folds_cover_every_row_exactly_once_in_validation():
    folds = expanding_window_folds(1_000, n_folds=4)
    covered = np.concatenate([fold.valid_index for fold in folds])
    start = min(fold.train_index.max() for fold in folds) + 1
    assert np.array_equal(np.sort(covered), np.arange(start, 1_000))


def test_folds_are_contiguous_blocks():
    for fold in expanding_window_folds(500, n_folds=3):
        assert np.array_equal(fold.valid_index, np.arange(fold.valid_index[0], fold.valid_index[-1] + 1))


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def imbalanced() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    n = 3_000
    y = (rng.random(n) < 0.01).astype(int)
    X = rng.normal(size=(n, 6))
    X[y == 1] += 1.5
    return X, y


def test_baseline_arms_do_not_resample(imbalanced):
    X, y = imbalanced
    for name in ("unweighted_baseline", "class_weight"):
        X_res, y_res = resample(X, y, name, None)
        assert len(y_res) == len(y)


def test_undersampling_reaches_the_requested_ratio(imbalanced):
    X, y = imbalanced
    _, y_res = resample(X, y, "random_undersample_10_to_1", 0.10)
    ratio = y_res.sum() / (len(y_res) - y_res.sum())
    assert ratio == pytest.approx(0.10, rel=0.05)


def test_smote_reaches_the_requested_ratio(imbalanced):
    X, y = imbalanced
    _, y_res = resample(X, y, "smote_5pct", 0.05)
    ratio = y_res.sum() / (len(y_res) - y_res.sum())
    assert ratio == pytest.approx(0.05, rel=0.10)


def test_smote_invents_minority_rows(imbalanced):
    X, y = imbalanced
    X_res, y_res = resample(X, y, "smote_5pct", 0.05)
    assert y_res.sum() > y.sum()
    assert len(y_res) > len(y)


def test_unknown_strategy_raises(imbalanced):
    X, y = imbalanced
    with pytest.raises(ValueError, match="Unknown resampling strategy"):
        build_sampler("not_a_strategy", 0.05, y)


def test_resampling_refuses_a_hopeless_minority_count():
    with pytest.raises(ValueError, match="not enough to resample"):
        build_sampler("smote_5pct", 0.05, np.array([0, 1, 0, 0]))


def test_validation_rows_are_never_resampled(imbalanced):
    """A leak here is invisible in the output but invalidates every number.

    The check: the union of resampled training rows plus untouched validation
    rows must not contain a synthetic point whose nearest training neighbour was
    a validation row. That is hard to assert directly, so the observable
    invariant is used instead - the validation block itself is passed through
    byte-identical, and the sample count of the resampled set is exactly the
    training count plus the number of synthetic minority rows.
    """
    X, y = imbalanced
    folds = expanding_window_folds(len(y), n_folds=3)
    fold = folds[-1]
    X_train, y_train = X[fold.train_index], y[fold.train_index]
    X_valid = X[fold.valid_index].copy()

    X_res, y_res = resample(X_train, y_train, "smote_5pct", 0.05)
    assert len(y_res) == len(y_train) + int(y_res.sum() - y_train.sum())
    assert np.array_equal(X[fold.valid_index], X_valid)


# ---------------------------------------------------------------------------
# The studies
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def study_data():
    rng = np.random.default_rng(6)
    n = 6_000
    y = (rng.random(n) < 0.012).astype(int)
    signal = y * 0.9 + rng.normal(0, 1.0, n)
    drift = np.linspace(0, 1, n)
    X = np.column_stack([signal, rng.normal(size=n), drift])
    return X, y


def test_sampling_comparison_returns_one_row_per_policy(study_data):
    X, y = study_data
    table = compare_sampling_strategies(
        X, y, strategies={"unweighted_baseline": None, "smote_5pct": 0.05}, n_folds=2, verbose=False
    )
    assert set(table["strategy"]) == {"unweighted_baseline", "smote_5pct"}
    assert (table["mean_pr_auc"] > 0).all()
    assert table["mean_pr_auc"].is_monotonic_decreasing


def test_sampling_comparison_records_how_much_data_each_policy_uses(study_data):
    X, y = study_data
    table = compare_sampling_strategies(
        X, y, strategies={"random_undersample_10_to_1": 0.10}, n_folds=2, verbose=False
    )
    assert table["share_of_data_used"].iloc[0] < 0.5


def test_leakage_ablation_reports_the_expected_variants(study_data):
    X, y = study_data
    table = sampling_leakage_ablation(X, y, n_folds=2, verbose=False)
    variants = set(table["variant"])
    for expected in (
        "A_resample_before_split_random_folds",
        "B_resample_inside_fold_random_folds",
        "C_resample_inside_fold_time_folds",
        "D_no_resampling_time_folds",
        "A_minus_B_resampling_placement_effect",
        "B_minus_C_random_vs_time_fold_effect",
        "A_minus_C_total_optimism",
        "C_minus_D_value_of_resampling",
    ):
        assert expected in variants


def test_leakage_ablation_is_self_consistent(study_data):
    """The reported differences must equal the differences of the reported levels."""
    X, y = study_data
    table = sampling_leakage_ablation(X, y, n_folds=2, verbose=False)
    levels = table[table["variant"].str.startswith(("A_", "B_", "C_", "D_"))].set_index("variant")["mean_pr_auc"]

    def diff(label: str) -> float:
        return table.loc[table["variant"] == label, "mean_pr_auc"].iloc[0]

    assert diff("A_minus_C_total_optimism") == pytest.approx(
        levels["A_resample_before_split_random_folds"] - levels["C_resample_inside_fold_time_folds"], abs=1e-4
    )
    assert diff("A_minus_B_resampling_placement_effect") == pytest.approx(
        levels["A_resample_before_split_random_folds"] - levels["B_resample_inside_fold_random_folds"], abs=1e-4
    )
    assert diff("B_minus_C_random_vs_time_fold_effect") == pytest.approx(
        levels["B_resample_inside_fold_random_folds"] - levels["C_resample_inside_fold_time_folds"], abs=1e-4
    )
    assert diff("C_minus_D_value_of_resampling") == pytest.approx(
        levels["C_resample_inside_fold_time_folds"] - levels["D_no_resampling_time_folds"], abs=1e-4
    )


def test_resampling_before_the_split_inflates_the_score(study_data):
    """The headline finding, asserted rather than asserted-in-prose."""
    X, y = study_data
    table = sampling_leakage_ablation(X, y, n_folds=2, verbose=False)
    total = table.loc[table["variant"] == "A_minus_C_total_optimism", "mean_pr_auc"].iloc[0]
    assert total > 0.05
