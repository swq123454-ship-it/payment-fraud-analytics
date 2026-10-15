"""Production monitoring: population stability, score drift and retraining triggers.

A fraud model does not decay politely. Adversaries adapt, so the score
distribution moves first and the label rate follows weeks later - which is why
monitoring has to run on inputs and scores, not on accuracy.

The thresholds below are the conventional ones (PSI 0.10 for a watch, 0.25 for
action) and they are configured rather than hard-coded so that a reviewer can see
where the escalation policy lives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

#: Population Stability Index bands. These are the standard credit-risk
#: conventions, chosen here because a fraud score behaves like a credit score:
#: a monotone, bounded, decision-driving quantity.
PSI_WATCH: float = 0.10
PSI_ACTION: float = 0.25


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """PSI between a reference distribution and a later one.

    Bin edges come from the reference sample, then are frozen and applied to the
    new sample. Re-deriving the edges from the incoming data would quietly hide
    exactly the shift the metric exists to detect.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    quantiles[0], quantiles[-1] = -np.inf, np.inf

    expected_counts, _ = np.histogram(expected, bins=quantiles)
    actual_counts, _ = np.histogram(actual, bins=quantiles)

    expected_share = np.clip(expected_counts / max(expected_counts.sum(), 1), epsilon, None)
    actual_share = np.clip(actual_counts / max(actual_counts.sum(), 1), epsilon, None)

    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def psi_verdict(psi: float) -> str:
    if not np.isfinite(psi):
        return "unknown"
    if psi < PSI_WATCH:
        return "stable"
    if psi < PSI_ACTION:
        return "watch"
    return "retrain"


def score_drift(reference_score: np.ndarray, actual_score: np.ndarray) -> dict:
    """How the score distribution moved between two periods."""
    reference_score = np.asarray(reference_score, dtype=float)
    actual_score = np.asarray(actual_score, dtype=float)
    psi = population_stability_index(reference_score, actual_score)
    ks_stat, ks_pvalue = stats.ks_2samp(reference_score, actual_score)
    return {
        "psi": round(float(psi), 4),
        "verdict": psi_verdict(psi),
        "ks_statistic": round(float(ks_stat), 4),
        "ks_pvalue": round(float(ks_pvalue), 6),
        "reference_mean_score": round(float(reference_score.mean()), 6),
        "actual_mean_score": round(float(actual_score.mean()), 6),
        "reference_flag_rate": round(float((reference_score >= 0.5).mean()), 6),
        "actual_flag_rate": round(float((actual_score >= 0.5).mean()), 6),
    }


def _numeric_block(frame: pd.DataFrame) -> pd.DataFrame:
    block = frame.select_dtypes(include=[np.number]).copy()
    return block.loc[:, block.nunique(dropna=True) > 1]


def feature_drift_table(
    reference: pd.DataFrame,
    actual: pd.DataFrame,
    columns: list[str] | None = None,
    top_k: int | None = None,
) -> pd.DataFrame:
    """PSI and a KS test for each feature the model consumes.

    A single score-level PSI says the model is behaving differently but not why.
    The feature table says which input moved, which is the difference between a
    monitoring alert and a diagnosis.
    """
    numeric = _numeric_block(reference)
    if columns:
        numeric = numeric[[c for c in columns if c in numeric.columns]]

    rows = []
    for column in numeric.columns:
        if column not in actual.columns:
            continue
        reference_values = reference[column].to_numpy(dtype=float)
        actual_values = pd.to_numeric(actual[column], errors="coerce").to_numpy(dtype=float)
        reference_values = reference_values[np.isfinite(reference_values)]
        actual_values = actual_values[np.isfinite(actual_values)]
        if len(reference_values) < 10 or len(actual_values) < 10:
            continue
        psi = population_stability_index(reference_values, actual_values)
        ks_stat, ks_pvalue = stats.ks_2samp(reference_values, actual_values)
        rows.append(
            {
                "feature": column,
                "psi": round(float(psi), 4),
                "verdict": psi_verdict(psi),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pvalue), 6),
                "reference_mean": round(float(reference_values.mean()), 4),
                "actual_mean": round(float(actual_values.mean()), 4),
            }
        )

    table = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return table if top_k is None else table.head(top_k)


def drift_monitor(
    reference_features: pd.DataFrame,
    actual_features: pd.DataFrame,
    reference_score: np.ndarray,
    actual_score: np.ndarray,
    top_k: int = 15,
) -> dict:
    """The monitoring bundle a model owner would receive each month."""
    score = score_drift(reference_score, actual_score)
    features = feature_drift_table(reference_features, actual_features, top_k=top_k)
    return {
        "score_drift": score,
        "top_drifting_features": features.to_dict(orient="records"),
        "features_actionable": int((features["verdict"] == "retrain").sum()) if len(features) else 0,
        "retrain_recommended": bool(score["verdict"] == "retrain" or (len(features) and (features["verdict"] == "retrain").sum() >= 3)),
        "thresholds": {"psi_watch": PSI_WATCH, "psi_action": PSI_ACTION},
    }


def monthly_stability(
    frame: pd.DataFrame,
    score: np.ndarray,
    date_column: str = "TransactionDT",
) -> pd.DataFrame:
    """Score PSI against the first observed month, month by month.

    This is how drift actually shows up in production: not as a single jump but
    as a slow creep that a one-shot train/test comparison cannot see.
    """
    working = pd.DataFrame(
        {
            "month": pd.to_datetime(frame[date_column], unit="s").dt.to_period("M").astype(str),
            "score": np.asarray(score, dtype=float),
        }
    )
    reference = working.loc[working["month"] == working["month"].min(), "score"].to_numpy()
    rows = []
    for month, group in working.groupby("month", sort=True):
        current = group["score"].to_numpy()
        psi = population_stability_index(reference, current)
        rows.append(
            {
                "month": month,
                "transactions": int(len(current)),
                "mean_score": round(float(current.mean()), 6),
                "share_above_0_5": round(float((current >= 0.5).mean()), 6),
                "psi_vs_first_month": round(float(psi), 4),
                "verdict": psi_verdict(psi),
            }
        )
    return pd.DataFrame(rows)
