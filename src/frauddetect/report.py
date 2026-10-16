"""End-to-end orchestration: data, warehouse, models, policy, explanation, report.

Validation protocol, stated once so that every number in the output can be traced
to it:

==================  =========================================================
window              use
==================  =========================================================
train (70%)         fit the model, fit the encoder vocabularies
test (15%)          calibration, threshold selection - never used to fit weights
out-of-time (15%)   evaluation only, touched by nothing that fits anything
==================  =========================================================

The out-of-time window is strictly later in time than the training window, so it
also carries the drift the model will actually meet.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import config as cfg
from . import plots
from .data import load_extract
from .drift import drift_monitor, feature_drift_table, monthly_stability, score_drift
from .evaluate import (
    Calibrator,
    Policy,
    approve_all_cost,
    expected_calibration_error,
    optimise_policy,
    policy_outcome,
    policy_table,
    pr_curve_points,
    ranking_metrics,
    reliability_table,
    review_band_sensitivity,
    roc_curve_points,
)
from .explain import (
    ShapExplainer,
    alert_queue,
    case_attribution,
    global_importance,
    permutation_table,
)
from .features import FeaturePrep, assert_no_forbidden_features, split_frames
from .imbalance import compare_sampling_strategies, sampling_leakage_ablation
from .models import (
    fit_autoencoder,
    fit_isolation_forest,
    fit_lightgbm,
    fit_logistic,
)
from .warehouse import build_warehouse, read_table, table_counts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_table(frame: pd.DataFrame, name: str) -> None:
    cfg.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cfg.TABLE_DIR / name, index=False)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _policy_row(label: str, outcome: dict) -> dict:
    return {"policy": label, **outcome}


def _render_missing(value):
    """Show an absent entity-history feature as text, not as ``nan``."""
    if isinstance(value, float) and np.isnan(value):
        return "missing"
    if isinstance(value, float):
        return f"{value:g}"
    return value


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    data_mode: str = "synthetic",
    n_transactions: int = cfg.DEFAULT_N_TRANSACTIONS,
    seed: int = cfg.SEED,
    verbose: bool = True,
) -> dict:
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Raw extract and warehouse
    # ------------------------------------------------------------------
    if verbose:
        print("[1/9] generating extract and building the SQL warehouse")
    extract = load_extract(data_mode=data_mode, n_transactions=n_transactions, seed=seed)
    build_warehouse(extract, verbose=verbose)

    mart = read_table("mart_features")
    mart_vendor = read_table("mart_features_vendor")
    quality = read_table("data_quality")
    split_summary = read_table("mart_split_summary")
    governance = read_table("feature_governance")

    mart["split"] = mart["split"].astype(str)
    mart_vendor["split"] = mart_vendor["split"].astype(str)

    _write_table(quality, "01_data_quality.csv")
    _write_table(split_summary, "02_split_summary.csv")
    _write_table(governance, "03_feature_governance.csv")

    frames = split_frames(mart)
    train, test, oot = frames["train"], frames["test"], frames["oot"]

    if verbose:
        print(
            f"  [warehouse] {len(mart):,} transactions | train {len(train):,} "
            f"({train[cfg.TARGET_COLUMN].sum():.0f} frauds) | test {len(test):,} "
            f"({test[cfg.TARGET_COLUMN].sum():.0f}) | oot {len(oot):,} "
            f"({oot[cfg.TARGET_COLUMN].sum():.0f})"
        )

    y_train = train[cfg.TARGET_COLUMN].to_numpy()
    y_test = test[cfg.TARGET_COLUMN].to_numpy()
    y_oot = oot[cfg.TARGET_COLUMN].to_numpy()
    amount_test = test["TransactionAmt"].to_numpy(dtype=float)
    amount_oot = oot["TransactionAmt"].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # 2. Encoder, design matrices, leakage guard
    # ------------------------------------------------------------------
    if verbose:
        print("[2/9] fitting the encoder and materialising design matrices")
    prep = FeaturePrep().fit(train)
    assert_no_forbidden_features(prep.feature_names)

    X_train = prep.transform(train)
    X_test = prep.transform(test)
    X_oot = prep.transform(oot)
    feature_names = prep.feature_names

    # One-hot matrix, used wherever a sampler needs dense, NaN-free numbers.
    L_train = prep.transform_linear(train).to_numpy(dtype=float)
    L_test = prep.transform_linear(test).to_numpy(dtype=float)
    L_oot = prep.transform_linear(oot).to_numpy(dtype=float)

    _write_table(prep.audit(), "04_encoder_audit.csv")
    if verbose:
        print(f"  [features] {len(feature_names)} features ({len(prep.numeric_columns_)} numeric)")

    # ------------------------------------------------------------------
    # 3. Imbalance study: resampling policy and the leakage ablation
    # ------------------------------------------------------------------
    if verbose:
        print("[3/9] comparing resampling policies and quantifying the optimism")
    study_rows = min(cfg.SAMPLING_STUDY_MAX_ROWS, len(L_train))
    L_study = L_train[-study_rows:]
    y_study = y_train[-study_rows:]

    sampling_table = compare_sampling_strategies(L_study, y_study, verbose=verbose)
    _write_table(sampling_table, "05_resampling_policies.csv")

    ablation = sampling_leakage_ablation(L_study, y_study, verbose=verbose)
    _write_table(ablation, "06_leakage_ablation.csv")

    # ------------------------------------------------------------------
    # 4. Final models
    # ------------------------------------------------------------------
    if verbose:
        print("[4/9] fitting the model family on the native-categorical matrix")
    production = fit_lightgbm(X_train, y_train, seed=seed)
    logistic = fit_logistic(L_train, y_train, seed=seed)

    # The unsupervised challengers need dense, NaN-free numbers, so they consume
    # the same numeric block the linear model uses and share its imputation.
    def numeric_block(matrix: pd.DataFrame) -> pd.DataFrame:
        return matrix[prep.numeric_columns_].fillna(0.0)

    N_train = numeric_block(X_train)
    N_test = numeric_block(X_test)
    N_oot = numeric_block(X_oot)

    isolation = fit_isolation_forest(N_train, y_train, seed=seed)
    autoencoder, _ = fit_autoencoder(N_train, y_train, seed=seed)

    model_scores_test: dict[str, np.ndarray] = {
        "lightgbm": production.predict_proba(X_test)[:, 1],
        "logistic_regression": logistic.predict_proba(L_test)[:, 1],
        "isolation_forest": isolation.predict_proba(N_test)[:, 1],
        "autoencoder": autoencoder.predict_proba(N_test)[:, 1],
    }
    model_scores_oot: dict[str, np.ndarray] = {
        "lightgbm": production.predict_proba(X_oot)[:, 1],
        "logistic_regression": logistic.predict_proba(L_oot)[:, 1],
        "isolation_forest": isolation.predict_proba(N_oot)[:, 1],
        "autoencoder": autoencoder.predict_proba(N_oot)[:, 1],
    }

    # ------------------------------------------------------------------
    # 5. Calibration and thresholds on the validation window
    # ------------------------------------------------------------------
    if verbose:
        print("[5/9] calibrating on the validation window and choosing thresholds")
    calibrator = Calibrator().fit(y_test, model_scores_test["lightgbm"])
    score_test = calibrator.apply(model_scores_test["lightgbm"])
    score_oot = calibrator.apply(model_scores_oot["lightgbm"])

    policy = optimise_policy(y_test, amount_test, score_test, allow_review=True)
    single_policy = optimise_policy(y_test, amount_test, score_test, allow_review=False)

    outcome_oot = policy_outcome(y_oot, amount_oot, score_oot, policy.review_threshold, policy.decline_threshold)
    outcome_oot_single = policy_outcome(
        y_oot, amount_oot, score_oot, single_policy.decline_threshold, single_policy.decline_threshold
    )
    outcome_oot_naive = policy_outcome(y_oot, amount_oot, score_oot, 0.5, 0.5)
    baseline = approve_all_cost(y_oot, amount_oot)

    comparison = pd.DataFrame(
        [
            _policy_row(
                "approve everything (baseline)",
                baseline | {"review_rate": 0.0, "decline_rate": 0.0, "fraud_capture_rate": 0.0},
            ),
            _policy_row("fixed 0.5 threshold", outcome_oot_naive),
            _policy_row("single threshold fitted on validation", outcome_oot_single),
            _policy_row("two-threshold policy fitted on validation", outcome_oot),
        ]
    )
    _write_table(comparison, "07_policy_comparison_oot.csv")

    policy_records = pd.DataFrame(
        [
            {"window": "validation (test)", **asdict(policy)},
            {"window": "validation (test, no review band)", **asdict(single_policy)},
        ]
    )
    _write_table(policy_records, "08_policy_fitted_on_validation.csv")
    _write_table(pd.DataFrame([{"window": "out-of-time", **outcome_oot}]), "09_policy_outcome_oot.csv")

    # Cost curves for the figures.
    thresholds = np.quantile(score_test, np.linspace(0.5, 0.9995, 60))
    cost_curve = policy_table(y_test, amount_test, score_test, thresholds)
    _write_table(cost_curve, "10_cost_curve_single_threshold.csv")

    surface = _cost_surface(y_test, amount_test, score_test)
    _write_table(surface, "11_cost_surface_two_thresholds.csv")

    # When is the manual queue worth paying for? Swept on the validation window,
    # because it is a statement about the cost model rather than about the
    # out-of-time draw.
    sensitivity = review_band_sensitivity(y_test, amount_test, score_test)
    _write_table(sensitivity, "23_review_band_sensitivity.csv")

    # ------------------------------------------------------------------
    # 6. Model comparison on both windows
    # ------------------------------------------------------------------
    if verbose:
        print("[6/9] scoring the model family on the validation and out-of-time windows")
    rows = []
    for name in model_scores_test:
        metrics_test = ranking_metrics(y_test, model_scores_test[name])
        metrics_oot = ranking_metrics(y_oot, model_scores_oot[name], with_interval=(name == "lightgbm"))
        rows.append(
            {
                "model": name,
                "test_pr_auc": metrics_test["pr_auc"],
                "test_roc_auc": metrics_test["roc_auc"],
                "oot_pr_auc": metrics_oot["pr_auc"],
                "oot_roc_auc": metrics_oot["roc_auc"],
                "oot_ks": metrics_oot["ks"],
                "oot_pr_auc_ci_low": (metrics_oot.get("pr_auc_bootstrap") or {}).get("ci_low"),
                "oot_pr_auc_ci_high": (metrics_oot.get("pr_auc_bootstrap") or {}).get("ci_high"),
                "recall_at_capacity": metrics_oot["recall_at_capacity"],
                "precision_at_capacity": metrics_oot["precision_at_capacity"],
                "lift_at_capacity": metrics_oot["lift_at_capacity"],
            }
        )
    performance = pd.DataFrame(rows)
    _write_table(
        performance.drop(columns=["recall_at_capacity", "precision_at_capacity", "lift_at_capacity"]),
        "12_model_performance.csv",
    )

    recall_table = performance[["model", "recall_at_capacity", "precision_at_capacity", "lift_at_capacity"]].copy()
    for capacity, suffix in (("1.000%", "1pct"), ("5.000%", "5pct")):
        recall_table[f"recall_at_{suffix}"] = recall_table["recall_at_capacity"].apply(lambda d, c=capacity: d[c])
        recall_table[f"precision_at_{suffix}"] = recall_table["precision_at_capacity"].apply(lambda d, c=capacity: d[c])
        recall_table[f"lift_at_{suffix}"] = recall_table["lift_at_capacity"].apply(lambda d, c=capacity: d[c])
    _write_table(
        recall_table[
            [
                "model",
                "recall_at_1pct",
                "precision_at_1pct",
                "lift_at_1pct",
                "recall_at_5pct",
                "precision_at_5pct",
                "lift_at_5pct",
            ]
        ],
        "13_recall_at_capacity.csv",
    )

    curves_pr = {
        "lightgbm": pr_curve_points(y_oot, model_scores_oot["lightgbm"]),
        "logistic_regression": pr_curve_points(y_oot, model_scores_oot["logistic_regression"]),
        "isolation_forest": pr_curve_points(y_oot, model_scores_oot["isolation_forest"]),
        "autoencoder": pr_curve_points(y_oot, model_scores_oot["autoencoder"]),
    }
    curves_roc = {
        "lightgbm": roc_curve_points(y_oot, model_scores_oot["lightgbm"]),
        "logistic_regression": roc_curve_points(y_oot, model_scores_oot["logistic_regression"]),
        "isolation_forest": roc_curve_points(y_oot, model_scores_oot["isolation_forest"]),
        "autoencoder": roc_curve_points(y_oot, model_scores_oot["autoencoder"]),
    }
    summary_for_curves = performance[["model", "oot_pr_auc", "oot_roc_auc"]].rename(
        columns={"oot_pr_auc": "pr_auc", "oot_roc_auc": "roc_auc"}
    )

    # ------------------------------------------------------------------
    # 7. Vendor aggregate leakage study
    # ------------------------------------------------------------------
    if verbose:
        print("[7/9] measuring the optimism in the vendor-supplied counters")
    vendor_gap = _vendor_leakage_study(mart_vendor, verbose=verbose)
    _write_table(vendor_gap, "14_vendor_leakage_study.csv")

    # ------------------------------------------------------------------
    # 8. Explanation and drift
    # ------------------------------------------------------------------
    if verbose:
        print("[8/9] explaining the model and monitoring its drift")
    explainer = ShapExplainer.from_model(production, feature_names)
    X_shap = prep.transform(oot).iloc[: min(len(oot), cfg.SHAP_SAMPLE_ROWS)]
    shap_table = global_importance(explainer, X_shap)
    _write_table(shap_table, "15_shap_global.csv")

    permutation = permutation_table(production, X_oot, y_oot, n_repeats=cfg.PERMUTATION_REPEATS)
    _write_table(permutation, "16_permutation_importance.csv")

    queue = alert_queue(explainer, X_oot, score_oot, y_oot, amount_oot, top_n=10)
    _write_table(queue, "17_alert_queue.csv")

    case_rows = []
    order = np.argsort(-score_oot)
    for label, position in (
        ("highest_scoring_fraud", _first_index(order, y_oot == 1)),
        ("highest_scoring_false_positive", _first_index(order, y_oot == 0)),
    ):
        if position is None:
            continue
        frame = X_oot.iloc[[position]]
        attribution = case_attribution(explainer, frame, top_k=6)
        attribution.insert(0, "case", label)
        attribution.insert(1, "score", round(float(score_oot[position]), 6))
        attribution.insert(2, "actual_fraud", int(y_oot[position]))
        attribution.insert(3, "amount", round(float(amount_oot[position]), 2))
        case_rows.append(attribution)
    cases = pd.concat(case_rows, ignore_index=True) if case_rows else pd.DataFrame()
    if not cases.empty:
        _write_table(cases, "18_case_attributions.csv")

    reliability_before = reliability_table(y_oot, model_scores_oot["lightgbm"])
    reliability_after = reliability_table(y_oot, score_oot)
    calibration_table = reliability_before.merge(reliability_after, on="bucket", suffixes=("_raw", "_calibrated"))
    _write_table(calibration_table, "19_calibration.csv")

    score_drift_table = pd.DataFrame([score_drift(model_scores_test["lightgbm"], score_oot)])
    _write_table(score_drift_table, "20_score_drift.csv")
    monthly = monthly_stability(oot, score_oot)
    _write_table(monthly, "21_monthly_stability.csv")

    # ------------------------------------------------------------------
    # 9. Figures and report
    # ------------------------------------------------------------------
    if verbose:
        print("[9/9] rendering figures and writing the report")
    plots.plot_pr_curves(curves_pr, summary_for_curves, float(y_oot.mean()))
    plots.plot_roc_curves(curves_roc, summary_for_curves)
    plots.plot_score_distribution(
        y_oot, score_oot, "Score distribution by class, out-of-time window", "03_score_distribution.png"
    )
    plots.plot_calibration(reliability_before, reliability_after)
    plots.plot_cost_curve(cost_curve, single_policy.decline_threshold)
    plots.plot_cost_surface(surface, policy.review_threshold, policy.decline_threshold)
    plots.plot_recall_at_capacity(performance, cfg.REVIEW_CAPACITY_GRID)
    plots.plot_sampling_comparison(sampling_table)
    plots.plot_leakage_ablation(ablation)
    plots.plot_shap_global(shap_table)
    if not cases.empty:
        for label, colour_name in (
            ("highest_scoring_fraud", "09b_shap_case_fraud.png"),
            ("highest_scoring_false_positive", "09c_shap_case_false_positive.png"),
        ):
            subset = cases[cases["case"] == label]
            if not subset.empty:
                plots.plot_shap_case(
                    subset[["feature", "value", "shap_contribution"]],
                    float(subset["score"].iloc[0]),
                    f"Why this transaction scored what it did - {label.replace('_', ' ')}",
                    colour_name,
                )
    plots.plot_psi_heatmap(feature_drift_table(test, oot, columns=prep.numeric_columns_, top_k=15))
    plots.plot_monthly_stability(monthly)
    plots.plot_amount_vs_score(amount_oot, score_oot, y_oot, policy.review_threshold)
    segments = _entity_risk_buckets(train, oot)
    _write_table(segments, "22_entity_risk_segments.csv")
    plots.plot_entity_risk(segments)
    plots.plot_review_band_sensitivity(sensitivity)

    # ------------------------------------------------------------------
    # Metrics bundle and report
    # ------------------------------------------------------------------
    manifest = extract.manifest
    (cfg.OUTPUT_DIR / "generator_manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")

    drift_bundle = drift_monitor(test, oot, model_scores_test["lightgbm"], score_oot, top_k=15)

    headline_oot = ranking_metrics(y_oot, score_oot, with_interval=True)
    results = {
        "data": {
            "mode": data_mode,
            "transactions": int(len(mart)),
            "fraud_rate": round(float(mart[cfg.TARGET_COLUMN].mean()), 6),
            "train": int(len(train)),
            "test": int(len(test)),
            "oot": int(len(oot)),
            "oot_frauds": int(y_oot.sum()),
            "identity_coverage": round(float(1 - mart["identity_missing"].mean()), 4),
            "generator_manifest": manifest,
        },
        "headline_oot": headline_oot,
        "models": performance.drop(columns=["recall_at_capacity", "precision_at_capacity", "lift_at_capacity"]).to_dict(
            orient="records"
        ),
        "calibration": {
            "ece_raw": round(expected_calibration_error(y_oot, model_scores_oot["lightgbm"]), 6),
            "ece_calibrated": round(expected_calibration_error(y_oot, score_oot), 6),
        },
        "sampling": sampling_table.to_dict(orient="records"),
        "leakage_ablation": ablation.to_dict(orient="records"),
        "vendor_leakage_study": vendor_gap.to_dict(orient="records"),
        "review_band_sensitivity": sensitivity.to_dict(orient="records"),
        "policy": {
            "fitted_on": "validation (test) window",
            "cost_model": asdict(cfg.COST_MODEL),
            "two_threshold": asdict(policy),
            "single_threshold": asdict(single_policy),
            "out_of_time_outcome": outcome_oot,
            "approve_all_baseline": baseline,
            "comparison": comparison.to_dict(orient="records"),
        },
        "drift": drift_bundle,
        "top_features_shap": shap_table.head(12).to_dict(orient="records"),
        "warehouse": table_counts(),
    }
    cfg.METRICS_PATH.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")

    _write_results_markdown(results, quality, split_summary, comparison, cases, segments)
    _save_bundle(prep, production, calibrator, policy, performance, results, seed=seed, n_requested=n_transactions)

    if verbose:
        print("\nDone.")
        print(f"  out-of-time PR-AUC   : {headline_oot['pr_auc']}")
        print(f"  out-of-time ROC-AUC  : {headline_oot['roc_auc']}")
        print(f"  policy thresholds    : review {policy.review_threshold:.4f} / decline {policy.decline_threshold:.4f}")
        print(
            f"  policy on validation : review {policy.n_reviewed:,} / "
            f"decline {policy.n_declined:,} / approve {policy.n_approved:,} "
            f"(two-threshold {policy.expected_cost:,.0f} vs single "
            f"{single_policy.expected_cost:,.0f})"
        )
        print(f"  cost vs approve-all  : {outcome_oot['expected_cost']:,.0f} vs {baseline['expected_cost']:,.0f}")
    return results


# ---------------------------------------------------------------------------
# Support routines
# ---------------------------------------------------------------------------
def _first_index(order: np.ndarray, mask: np.ndarray) -> int | None:
    for position in order:
        if mask[position]:
            return int(position)
    return None


def _cost_surface(
    y: np.ndarray,
    amount: np.ndarray,
    score: np.ndarray,
    n_points: int = 22,
) -> pd.DataFrame:
    """Grid of total cost over review and decline thresholds, fitted on validation."""
    review_grid = np.quantile(score, np.linspace(0.0, 0.995, n_points))
    decline_grid = np.quantile(score, np.linspace(0.5, 0.9999, n_points))
    rows = []
    for review_threshold in np.unique(review_grid):
        for decline_threshold in np.unique(decline_grid):
            if review_threshold > decline_threshold:
                continue
            outcome = policy_outcome(y, amount, score, review_threshold, decline_threshold)
            rows.append(
                {
                    "review_threshold": round(float(review_threshold), 6),
                    "decline_threshold": round(float(decline_threshold), 6),
                    "total_cost": outcome["expected_cost"],
                }
            )
    return pd.DataFrame(rows)


def _vendor_leakage_study(mart_vendor: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Fit the same model twice: with and without the vendor's all-time counters.

    ``C5`` is a forward-looking seven-day count and ``C6`` is a deviation from an
    all-time mean, so both see transactions that had not happened yet. The gap
    between the two rows is the price of a leaked feature, measured rather than
    asserted.
    """
    frames = split_frames(mart_vendor)
    train, test, oot = frames["train"], frames["test"], frames["oot"]
    y_train = train[cfg.TARGET_COLUMN].to_numpy()
    y_test = test[cfg.TARGET_COLUMN].to_numpy()
    y_oot = oot[cfg.TARGET_COLUMN].to_numpy()

    rows = []
    for label, include_vendor in (("causal features only", False), ("plus vendor counters", True)):
        prep = FeaturePrep(include_vendor_aggregates=include_vendor).fit(train)
        model = fit_lightgbm(prep.transform(train), y_train, seed=cfg.SEED)
        score_test = model.predict_proba(prep.transform(test))[:, 1]
        score_oot = model.predict_proba(prep.transform(oot))[:, 1]
        rows.append(
            {
                "feature_set": label,
                "n_features": len(prep.feature_names),
                "test_pr_auc": round(float(ranking_metrics(y_test, score_test)["pr_auc"]), 4),
                "oot_pr_auc": round(float(ranking_metrics(y_oot, score_oot)["pr_auc"]), 4),
                "oot_roc_auc": round(float(ranking_metrics(y_oot, score_oot)["roc_auc"]), 4),
            }
        )

    table = pd.DataFrame(rows)
    table["pr_auc_gap"] = (table["oot_pr_auc"] - table["oot_pr_auc"].iloc[0]).round(4)
    if verbose:
        print(table.to_string(index=False))
    return table


def _entity_risk_buckets(train: pd.DataFrame, oot: pd.DataFrame) -> pd.DataFrame:
    """Fraud rate by the resolution-lagged card history feature."""
    edges = [-np.inf, 0.0, 0.01, 0.05, 0.15, 0.5, np.inf]
    labels = ["no resolved history", "0%", "0-1%", "1-5%", "5-15%", ">15%"]
    frame = oot.copy()
    frame["bucket"] = pd.cut(frame["card_hist_fraud_rate"], bins=edges, labels=labels)
    grouped = (
        frame.groupby("bucket", observed=True)
        .agg(fraud_rate=(cfg.TARGET_COLUMN, "mean"), n=(cfg.TARGET_COLUMN, "size"))
        .reset_index()
    )
    grouped["bucket"] = grouped["bucket"].astype(str)
    return grouped


def _save_bundle(prep, model, calibrator, policy: Policy, performance, results, seed: int, n_requested: int) -> None:
    from .api import ScoringBundle, save_bundle

    bundle = ScoringBundle(
        prep=prep,
        model=model,
        calibrator=calibrator,
        feature_names=prep.feature_names,
        review_threshold=policy.review_threshold,
        decline_threshold=policy.decline_threshold,
        metadata={
            # Keyed to the requested size, not the realised row count. Episodes
            # add transactions on top of the base stream, so the realised count
            # is 300,001 for a --transactions 300000 run; a version string that
            # disagreed with the command that produced it would be worse than
            # useless in an incident review.
            "version": f"lightgbm-seed{seed}-{n_requested}rows",
            "transactions_realised": results["data"]["transactions"],
            "trained_on": f"{results['data']['train']} transactions",
            "metrics": {
                "oot_pr_auc": results["headline_oot"]["pr_auc"],
                "oot_roc_auc": results["headline_oot"]["roc_auc"],
                "oot_recall_at_1pct": results["headline_oot"]["recall_at_capacity"]["1.000%"],
            },
            "cost_model": asdict(cfg.COST_MODEL),
        },
    )
    save_bundle(bundle)


def _ablation_reading(ablation) -> str:
    """Explain the ablation from its own numbers, not from a hard-coded story.

    The four protocols differ by exactly one thing at a time, so each adjacent
    difference isolates a single cause. Quoting those differences instead of
    restating the design keeps the paragraph honest if the numbers move.
    """
    table = pd.DataFrame(ablation).set_index("variant")["mean_pr_auc"]
    total = table.get("A_minus_C_total_optimism")
    placement = table.get("A_minus_B_resampling_placement_effect")
    folds = table.get("B_minus_C_random_vs_time_fold_effect")
    value = table.get("C_minus_D_value_of_resampling")
    if any(pd.isna(x) for x in (total, placement, folds, value)):
        return "The ablation did not produce all four contrasts; see the CSV."

    return (
        f"A protocol that resamples before splitting reports PR-AUC "
        f"**{total:+.4f}** higher than the honest one. Almost all of that - "
        f"{placement:+.4f} of it - is the *placement* of the resampling, not the "
        f"sampler: synthesising points across the split boundary lets a validation "
        f"row's nearest neighbours include the row itself. Holding the placement "
        f"fixed and swapping random folds for forward-chaining folds moves the "
        f"number by {folds:+.4f}, and dropping resampling altogether - the honest "
        f"baseline - costs {value:+.4f}. So the sampler is worth about "
        f"{abs(value):.4f} here, and the protocol is worth roughly "
        f"{abs(placement):.4f}. Reporting the first number as if it were the second "
        f"is the single most common way a fraud benchmark is overstated."
    )


def _review_band_reading(sensitivity, cost_model) -> str:
    """State the break-even price of an analyst alert, from the sweep itself.

    The useful output of the sweep is not "the queue is or is not worth it" but
    the price at which the answer flips, at the catch rate the model is actually
    configured with. That is a number an operations lead can act on, so it is
    read out of the grid rather than asserted.
    """
    catch = cost_model["review_catch_rate"]
    price = cost_model["review_cost"]
    at_configured = [row for row in sensitivity if abs(row["review_catch_rate"] - catch) < 1e-9]
    if not at_configured:
        return (
            "The sweep did not include the configured catch rate, so no break-even can be quoted; see the table below."
        )

    open_rows = [row for row in at_configured if row["review_band_open"]]
    if not open_rows:
        return (
            f"**At a {catch:.0%} catch rate the review band never pays for itself, at any alert "
            "price in the grid.** A three-way rule is available and its optimum is global, but "
            "that optimum collapses to approve-or-decline: the model's ranking is decisive enough "
            "that the alerts worth reviewing are already worth refusing, and the alerts that are "
            "not worth refusing are not worth an analyst's time either.\n"
            "\n"
            f"This is a finding about the cost model and the score distribution rather than a "
            f"defect, and it is why the shipped policy at {price:.0f} per alert barely improves on "
            "the single-threshold rule. It also frames the build-versus-buy decision correctly: "
            "the question is not whether a queue would help but what an alert would have to cost."
        )

    break_even = max(row["review_cost"] for row in open_rows)
    dearer = [
        row["review_cost"] for row in at_configured if not row["review_band_open"] and row["review_cost"] > break_even
    ]
    at_break_even = next(row for row in open_rows if row["review_cost"] == break_even)
    best = max(sensitivity, key=lambda row: row["review_band_saving"])

    if dearer:
        tail = (
            f"At {min(dearer):.0f} per alert the band closes again, so the break-even sits between "
            f"{break_even:.0f} and {min(dearer):.0f}."
        )
    else:
        tail = f"The band is still open at the dearest price tested ({break_even:.0f})."

    return (
        f"At the configured {catch:.0%} catch rate the review band pays for itself only while an "
        f"alert costs **{break_even:.0f} or less**, where it is worth "
        f"**{at_break_even['review_band_saving']:,.0f}** on the validation window over "
        f"approve-or-decline. The shipped price is {price:.0f}, which is on the other side of that "
        f"line. {tail}\n"
        "\n"
        f"Across the whole grid the most the band is ever worth is "
        f"**{best['review_band_saving']:,.0f}**, at {best['review_cost']:.0f} per alert and a "
        f"{best['review_catch_rate']:.0%} catch rate. Note which lever matters more: halving the "
        "analyst cost buys far less than making the review actually stop the transaction, because "
        "a review that only catches part of what it finds has already lost most of the money by "
        "the time anyone looks."
    )


def _write_results_markdown(results, quality, split_summary, comparison, cases, segments) -> None:
    lines: list[str] = []
    add = lines.append

    add("# Results\n")
    add(
        "Every number below is produced by `scripts/run_pipeline.py` on the synthetic "
        "extract described in the README. Nothing is hand-entered.\n"
    )

    add("## Data\n")
    add(
        f"- Transactions: **{results['data']['transactions']:,}**  "
        f"(train {results['data']['train']:,} / test {results['data']['test']:,} / out-of-time {results['data']['oot']:,})"
    )
    add(
        f"- Fraud rate: **{results['data']['fraud_rate']:.3%}** ({results['data']['oot_frauds']} frauds in the out-of-time window)"
    )
    add(f"- Identity-table coverage: **{results['data']['identity_coverage']:.1%}**\n")
    add(split_summary.to_markdown(index=False))
    add("")

    add("## Headline performance (out-of-time window)\n")
    headline = results["headline_oot"]
    interval = headline["pr_auc_bootstrap"]
    add(
        f"- **PR-AUC {headline['pr_auc']:.4f}** (95% bootstrap interval "
        f"{interval['ci_low']:.4f} - {interval['ci_high']:.4f}) against a random baseline of {headline['base_rate']:.4%}"
    )
    add(f"- ROC-AUC {headline['roc_auc']:.4f}, KS {headline['ks']:.4f}")
    add("Capacity-constrained performance on the same window:\n")
    capacity = pd.DataFrame(
        [
            {
                "capacity": label,
                "recall": f"{headline['recall_at_capacity'][label]:.1%}",
                "precision": f"{headline['precision_at_capacity'][label]:.1%}",
                "lift": f"{headline['lift_at_capacity'][label]:.1f}x",
            }
            for label in headline["recall_at_capacity"]
        ]
    )
    add(capacity.to_markdown(index=False))
    add("")
    add(
        "- Recall is the share of fraud inside the top slice; precision is how much "
        "of that slice is fraud; lift is precision divided by the base rate."
    )
    add(
        "- Accuracy is not reported: an all-negative classifier is "
        f"{1 - headline['base_rate']:.2%} accurate and catches nothing.\n"
    )

    add("## Model comparison\n")
    add(pd.DataFrame(results["models"]).to_markdown(index=False))
    add("")

    add("## Calibration\n")
    add(
        f"- Expected calibration error, raw: **{results['calibration']['ece_raw']:.5f}**; "
        f"after isotonic calibration: **{results['calibration']['ece_calibrated']:.5f}**"
    )
    add(
        "- The cost-optimal threshold is a function of the predicted probability, so an "
        "uncalibrated model produces a threshold that is wrong by the same factor.\n"
    )

    add("## Where the risk sits\n")
    add(
        "Fraud rate by the card's *lagged* resolved fraud history. The feature is\n"
        "deliberately excluded from its own bucket boundary - the lag means the\n"
        "current transaction cannot contribute to the number it is bucketed by.\n"
    )
    add(segments.to_markdown(index=False))
    add("")

    add("## Imbalance handling\n")
    add(pd.DataFrame(results["sampling"]).to_markdown(index=False))
    add("")
    add("**Where the optimism comes from** (same data, same model, four protocols):\n")
    add(pd.DataFrame(results["leakage_ablation"]).to_markdown(index=False))
    add("")
    add(_ablation_reading(results["leakage_ablation"]))
    add("")

    add("## Vendor counter leakage study\n")
    add(pd.DataFrame(results["vendor_leakage_study"]).to_markdown(index=False))
    add("")

    add("## Cost-sensitive decisioning\n")
    add("Cost model: " + json.dumps(results["policy"]["cost_model"]))
    add("")
    two = results["policy"]["two_threshold"]
    one = results["policy"]["single_threshold"]
    add("Fitted on the validation window (thresholds are chosen here and never re-tuned):\n")
    add(
        f"- **Two-threshold policy**: review at {two['review_threshold']:.4f}, decline at "
        f"{two['decline_threshold']:.4f} - expected cost **{two['expected_cost']:,.0f}**. "
        f"On this window it puts {two['review_rate']:.3%} of transactions through the analyst "
        f"queue and refuses {two['decline_rate']:.3%} outright."
    )
    add(
        f"- Single-threshold policy: approve or decline at {one['decline_threshold']:.4f} "
        f"(expected cost {one['expected_cost']:,.0f})"
    )
    add(
        f"- The review band is what the third action buys: "
        f"**{one['expected_cost'] - two['expected_cost']:,.0f}** cheaper on the window it was fitted on. "
        "It is deliberately described that way, because the thresholds are then frozen - an "
        "advantage measured on the window that chose them is not yet evidence.\n"
    )
    add("Applied unchanged to the out-of-time window:\n")
    add(comparison.to_markdown(index=False))
    add("")
    add(
        f"Reading the table: every row is the same model on the same {results['data']['oot']:,} "
        "unseen transactions, and only the decision rule differs. `approve everything` is the "
        "do-nothing benchmark; `fixed 0.5` is the default anyone reaches for, and it leaves most "
        "of the available saving on the table because a raw score of 0.5 is not a calibrated "
        "probability and has no reason to be the right cut.\n"
    )

    add("### Is the manual queue worth funding?\n")
    sensitivity = results.get("review_band_sensitivity") or []
    if sensitivity:
        add(_review_band_reading(sensitivity, results["policy"]["cost_model"]))
        add("")
        add(pd.DataFrame(sensitivity).to_markdown(index=False))
        add("")

    add("## Drift monitoring\n")
    add("| Metric | Value |")
    add("|---|---|")
    for key, value in results["drift"]["score_drift"].items():
        add(f"| {key} | {value} |")
    add("")
    add(f"Retraining recommended by the monitor: **{results['drift']['retrain_recommended']}**\n")

    add("## Top drivers (mean absolute SHAP)\n")
    add(pd.DataFrame(results["top_features_shap"]).to_markdown(index=False))
    add("")

    if not cases.empty:
        add("## Worked examples\n")
        for label, group in cases.groupby("case", sort=False):
            add(
                f"**{label}** - score {group['score'].iloc[0]:.6f}, amount {group['amount'].iloc[0]:.2f}, actual fraud {group['actual_fraud'].iloc[0]}\n"
            )
            shown = group[["feature", "value", "shap_contribution", "direction"]].copy()
            # A third of this design matrix is entity history, and the strongest
            # driver is often a feature that is absent. Printing ``nan`` in the
            # one table a reader studies line by line hides that.
            shown["value"] = shown["value"].map(_render_missing)
            add(shown.to_markdown(index=False))
            add("")

    add("## Data quality checks\n")
    add(quality.to_markdown(index=False))
    add("")

    cfg.RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
