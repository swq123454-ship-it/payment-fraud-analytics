"""Drift monitoring and explanation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frauddetect.drift import (
    feature_drift_table,
    monthly_stability,
    population_stability_index,
    psi_verdict,
    score_drift,
)
from frauddetect.explain import (
    ShapExplainer,
    alert_queue,
    case_attribution,
    case_narrative,
    global_importance,
    permutation_table,
)
from frauddetect.models import fit_lightgbm


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------
def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    values = rng.normal(size=5_000)
    assert population_stability_index(values, values) == pytest.approx(0.0, abs=1e-9)


def test_psi_is_small_for_the_same_distribution_resampled():
    rng = np.random.default_rng(1)
    reference = rng.normal(size=20_000)
    actual = rng.normal(size=20_000)
    assert population_stability_index(reference, actual) < 0.02


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=20_000)
    small = population_stability_index(reference, rng.normal(0.1, 1.0, 20_000))
    large = population_stability_index(reference, rng.normal(1.0, 1.0, 20_000))
    assert small < large


def test_psi_flags_a_clear_shift_for_retraining():
    rng = np.random.default_rng(3)
    reference = rng.normal(size=20_000)
    shifted = rng.normal(1.5, 1.0, 20_000)
    assert population_stability_index(reference, shifted) > 0.25
    assert psi_verdict(population_stability_index(reference, shifted)) == "retrain"


def test_psi_verdict_bands():
    assert psi_verdict(0.05) == "stable"
    assert psi_verdict(0.15) == "watch"
    assert psi_verdict(0.60) == "retrain"
    assert psi_verdict(float("nan")) == "unknown"


def test_psi_is_zero_when_a_distribution_has_no_spread():
    values = np.ones(1_000)
    assert population_stability_index(values, values) == 0.0


def test_score_drift_reports_both_psi_and_ks():
    rng = np.random.default_rng(4)
    reference = rng.random(5_000)
    actual = np.clip(rng.random(5_000) + 0.2, 0, 1)
    report = score_drift(reference, actual)
    assert set(report) >= {"psi", "verdict", "ks_statistic", "ks_pvalue", "reference_mean_score"}
    assert report["actual_mean_score"] > report["reference_mean_score"]


def test_feature_drift_table_ranks_the_shifted_feature_first():
    rng = np.random.default_rng(5)
    reference = pd.DataFrame({"stable": rng.normal(size=8_000), "moving": rng.normal(size=8_000), "constant": 1.0})
    actual = pd.DataFrame({"stable": rng.normal(size=8_000), "moving": rng.normal(1.2, 1.0, 8_000), "constant": 1.0})
    table = feature_drift_table(reference, actual)
    assert table["feature"].iloc[0] == "moving"
    # A constant column carries no information and must be skipped.
    assert "constant" not in set(table["feature"])


def test_monthly_stability_covers_every_month():
    rng = np.random.default_rng(6)
    frame = pd.DataFrame({"TransactionDT": np.sort(rng.integers(1_514_764_800, 1_530_000_000, 20_000))})
    score = rng.random(20_000)
    table = monthly_stability(frame, score)
    assert len(table) >= 4
    assert (table["psi_vs_first_month"] >= 0).all()
    assert set(table["verdict"]).issubset({"stable", "watch", "retrain", "unknown"})


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fitted():
    """A deliberately learnable problem.

    The base rate is set at 5% rather than the 0.35% the project operates at, and
    the trees are kept shallow. That is not to make the test easy: at a very low
    base rate a deep booster will happily spend most of its splits on pure noise,
    which is a real and interesting failure mode but a terrible thing to assert
    against. The production configuration is regularised for exactly that reason.
    """
    rng = np.random.default_rng(7)
    n = 20_000
    y = (rng.random(n) < 0.05).astype(int)
    frame = pd.DataFrame(
        {
            "signal": y * 3.0 + rng.normal(size=n),
            "noise": rng.normal(size=n),
            "category": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        }
    )
    model = fit_lightgbm(frame, y, params={"n_estimators": 80, "num_leaves": 15, "min_child_samples": 40})
    return model, frame, y


def test_shap_contributions_are_additive_and_reconstruct_the_margin(fitted):
    model, frame, _ = fitted
    explainer = ShapExplainer.from_model(model, list(frame.columns))
    contributions = explainer.contributions(frame.iloc[:200])
    margins = model.booster_.predict(frame.iloc[:200], raw_score=True)
    assert np.allclose(contributions.sum(axis=1).to_numpy() + explainer.base_value, margins, atol=1e-6)


def test_shap_library_and_lightgbm_pred_contrib_agree(fitted):
    """Two independent TreeSHAP implementations must produce the same numbers."""
    model, frame, _ = fitted
    explainer = ShapExplainer.from_model(model, list(frame.columns))
    ours = explainer.contributions(frame.iloc[:100]).to_numpy()
    theirs = explainer.shap_values_library(frame.iloc[:100])
    assert np.allclose(ours, theirs, atol=1e-5)


def test_global_importance_ranks_the_real_signal_first(fitted):
    model, frame, _ = fitted
    explainer = ShapExplainer.from_model(model, list(frame.columns))
    table = global_importance(explainer, frame)
    assert table["feature"].iloc[0] == "signal"
    assert table["mean_abs_shap"].is_monotonic_decreasing
    assert table["share_of_total"].sum() == pytest.approx(1.0, abs=1e-4)


def test_permutation_importance_ranks_the_real_signal_first(fitted):
    model, frame, y = fitted
    table = permutation_table(model, frame, y, n_repeats=2, max_rows=4_000)
    assert table["feature"].iloc[0] == "signal"


def test_permutation_importance_keeps_categorical_columns_intact(fitted):
    """Permuting a categorical column must not silently turn it into a number."""
    model, frame, y = fitted
    table = permutation_table(model, frame, y, n_repeats=2, max_rows=4_000)
    assert "category" in set(table["feature"])


def test_case_attribution_returns_the_requested_number_of_drivers(fitted):
    model, frame, _ = fitted
    explainer = ShapExplainer.from_model(model, list(frame.columns))
    table = case_attribution(explainer, frame.iloc[[0]], top_k=3)
    assert len(table) == 3
    assert table["abs_contribution"].is_monotonic_decreasing
    assert set(table["direction"]).issubset({"increases risk", "decreases risk"})


def test_case_attribution_rejects_multi_row_input(fitted):
    model, frame, _ = fitted
    explainer = ShapExplainer.from_model(model, list(frame.columns))
    with pytest.raises(ValueError, match="exactly one row"):
        case_attribution(explainer, frame.iloc[:3])


def test_case_narrative_mentions_the_margin_and_the_base(fitted):
    model, frame, _ = fitted
    explainer = ShapExplainer.from_model(model, list(frame.columns))
    explainer.contributions(frame.iloc[:1])
    narrative = case_narrative(explainer, frame.iloc[[0]], top_k=3)
    assert "log-odds" in narrative and "Largest drivers" in narrative


def test_alert_queue_returns_the_highest_scoring_rows(fitted):
    model, frame, y = fitted
    explainer = ShapExplainer.from_model(model, list(frame.columns))
    score = model.predict_proba(frame)[:, 1]
    amount = np.full(len(frame), 100.0)
    queue = alert_queue(explainer, frame, score, y, amount, top_n=5)
    assert len(queue) == 5
    assert queue["score"].is_monotonic_decreasing
    assert (queue["top_driver_1"] != "").all()
