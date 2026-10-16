"""Figures.

Every chart carries a title, labelled axes with units, and a source note. A
figure without a stated sample and seed is not evidence, it is decoration.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as cfg  # noqa: E402

FG = "#1f2d3d"
MUTED = "#5b6b7c"
GRID = "#d8dee6"
ACCENT = "#c0392b"
ACCENT_2 = "#1f6feb"
ACCENT_3 = "#7d3c98"
ACCENT_4 = "#1e8449"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": FG,
        "axes.titlecolor": FG,
        "axes.titlesize": 11.5,
        "axes.labelsize": 9.5,
        "axes.titleweight": "bold",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "font.size": 9.5,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "figure.autolayout": False,
    }
)


def _source(fig, text: str) -> None:
    fig.text(0.01, 0.008, text, fontsize=7.5, color=MUTED, ha="left", va="bottom")


def _save(fig, name: str, source: str) -> Path:
    _source(fig, source)
    cfg.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.FIGURE_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


SOURCE = "Simulated transaction extract (seed 42, 182 days). Strictly causal features."

#: One colour per series, in reporting order. Add a colour before adding a model.
MODEL_COLOURS = (ACCENT_2, ACCENT, ACCENT_3, ACCENT_4, MUTED)


def _model_colours(n: int) -> tuple[str, ...]:
    """The first ``n`` model colours, refusing to reuse one.

    A repeated colour on a comparison chart is worse than a crash: two series
    become indistinguishable exactly where the reader is trying to tell them
    apart. Running out of colours is therefore an error, not a wrap-around.
    """
    if n > len(MODEL_COLOURS):
        raise ValueError(f"{n} series but only {len(MODEL_COLOURS)} colours defined")
    return MODEL_COLOURS[:n]


# ---------------------------------------------------------------------------
def plot_pr_curves(curves: dict[str, pd.DataFrame], summary: pd.DataFrame, base_rate: float) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for (name, frame), colour in zip(curves.items(), _model_colours(len(curves)), strict=True):
        pr_auc = summary.loc[summary["model"] == name, "pr_auc"]
        label = f"{name} (PR-AUC {float(pr_auc.iloc[0]):.3f})" if len(pr_auc) else name
        ax.plot(frame["recall"], frame["precision"], color=colour, lw=1.8, label=label)
    ax.axhline(base_rate, color=MUTED, ls="--", lw=1.1, label=f"random baseline ({base_rate:.3%})")
    ax.set_xlabel("Recall - share of fraud caught")
    ax.set_ylabel("Precision - share of alerts that are fraud")
    ax.set_title(
        "Precision-recall curves, out-of-time window\nAt a 0.35% base rate, ROC-AUC is flattering and PR-AUC is not"
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    return _save(fig, "01_pr_curves.png", SOURCE)


def plot_roc_curves(curves: dict[str, pd.DataFrame], summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for (name, frame), colour in zip(curves.items(), _model_colours(len(curves)), strict=True):
        roc_auc = summary.loc[summary["model"] == name, "roc_auc"]
        label = f"{name} (AUC {float(roc_auc.iloc[0]):.3f})" if len(roc_auc) else name
        ax.plot(frame["false_positive_rate"], frame["true_positive_rate"], color=colour, lw=1.8, label=label)
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1.1, label="no skill")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves, out-of-time window")
    ax.legend(loc="lower right")
    return _save(fig, "02_roc_curves.png", SOURCE)


def plot_score_distribution(y: np.ndarray, score: np.ndarray, title: str, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    y = np.asarray(y)
    score = np.asarray(score)
    bins = np.linspace(0, 1, 60)
    ax.hist(score[y == 0], bins=bins, color=ACCENT_2, alpha=0.75, label="legitimate", log=True)
    ax.hist(score[y == 1], bins=bins, color=ACCENT, alpha=0.85, label="fraud", log=True)
    ax.set_xlabel("Predicted fraud probability")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title(title)
    ax.legend(loc="upper center")
    return _save(fig, name, SOURCE)


def plot_calibration(table_before: pd.DataFrame, table_after: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1.1, label="perfect calibration")
    for table, colour, label in (
        (table_before, ACCENT_2, "raw model output"),
        (table_after, ACCENT_4, "after isotonic calibration"),
    ):
        ax.plot(table["mean_predicted"], table["observed_rate"], "o-", color=colour, lw=1.6, ms=4.5, label=label)
    ax.set_xlabel("Mean predicted fraud probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title(
        "Calibration on the out-of-time window\nThresholds chosen on miscalibrated scores are wrong by the same factor"
    )
    ax.legend(loc="upper left")
    return _save(fig, "03_calibration.png", SOURCE)


def plot_cost_curve(table: pd.DataFrame, optimal_threshold: float) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(table["threshold"], table["approved_fraud_loss"], color=ACCENT, lw=1.7, label="loss on approved fraud")
    ax.plot(
        table["threshold"],
        table["decline_friction"],
        color=ACCENT_2,
        lw=1.7,
        label="friction from declined legitimate traffic",
    )
    ax.plot(table["threshold"], table["total_cost"], color=FG, lw=2.2, label="total expected cost")
    if np.isfinite(optimal_threshold):
        ax.axvline(optimal_threshold, color=ACCENT_3, ls=":", lw=1.5)
        ax.annotate(
            f"review threshold {optimal_threshold:.3f}",
            xy=(optimal_threshold, ax.get_ylim()[1] * 0.55),
            xytext=(optimal_threshold + 0.05, ax.get_ylim()[1] * 0.72),
            fontsize=8,
            color=ACCENT_3,
            arrowprops=dict(arrowstyle="->", color=ACCENT_3, lw=0.9),
        )
    ax.set_xlabel("Score threshold (approve below, decline at or above)")
    ax.set_ylabel("Expected cost across the window (currency units)")
    ax.set_title("Single-threshold cost curve, fitted on the validation window")
    ax.legend(loc="upper center")
    return _save(fig, "04_cost_curve_single_threshold.png", SOURCE)


def plot_cost_surface(grid: pd.DataFrame, review_threshold: float, decline_threshold: float) -> Path:
    """Heat map of the two-threshold cost surface, with the optimum marked."""
    pivot = grid.pivot(index="review_threshold", columns="decline_threshold", values="total_cost")
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    mesh = ax.pcolormesh(
        pivot.columns.to_numpy(),
        pivot.index.to_numpy(),
        pivot.to_numpy(),
        shading="auto",
        cmap="viridis_r",
    )
    fig.colorbar(mesh, ax=ax, label="Expected cost")
    if np.isfinite(review_threshold) and np.isfinite(decline_threshold):
        ax.plot(decline_threshold, review_threshold, marker="*", ms=14, color=ACCENT, mec="white", mew=1.0)
    ax.set_xlabel("Decline threshold")
    ax.set_ylabel("Review threshold")
    ax.set_title(
        "Two-threshold policy cost surface\nAuto-declining customers costs more than reviewing them, so a middle band exists"
    )
    return _save(fig, "05_cost_surface_two_thresholds.png", SOURCE)


def plot_recall_at_capacity(summary: pd.DataFrame, capacities: tuple[float, ...]) -> Path:
    columns = [f"{c:.3%}" for c in capacities]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    width = 0.8 / len(summary)
    positions = np.arange(len(columns))
    palette = [ACCENT_2, ACCENT, ACCENT_3, ACCENT_4]
    for i, (_, row) in enumerate(summary.iterrows()):
        values = [row["recall_at_capacity"][c] for c in columns]
        bars = ax.bar(positions + i * width, values, width, label=row["model"], color=palette[i % len(palette)])
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.0%}", ha="center", fontsize=7)
    ax.set_xticks(positions + width * (len(summary) - 1) / 2)
    ax.set_xticklabels([f"top {c}" for c in columns])
    ax.set_xlabel("Share of transactions reviewed (fixed analyst capacity)")
    ax.set_ylabel("Share of fraud caught")
    ax.set_ylim(0, 1.12)
    ax.set_title(
        "Recall at fixed review capacity, out-of-time window\nThe metric an operations team plans headcount against"
    )
    ax.legend(loc="upper left", ncols=2)
    return _save(fig, "06_recall_at_capacity.png", SOURCE)


def plot_sampling_comparison(table: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ordered = table.sort_values("mean_pr_auc")
    bars = ax.barh(ordered["strategy"], ordered["mean_pr_auc"], color=ACCENT_2)
    best = ordered["mean_pr_auc"].max()
    for bar, (_, row) in zip(bars, ordered.iterrows(), strict=True):
        colour = ACCENT if row["mean_pr_auc"] == best else ACCENT_2
        bar.set_color(colour)
        ax.text(
            bar.get_width() + 0.006,
            bar.get_y() + bar.get_height() / 2,
            f"{row['mean_pr_auc']:.3f}  (uses {row['share_of_data_used']:.0%} of rows)",
            va="center",
            fontsize=8,
        )
    ax.axvline(ordered["mean_pr_auc"].iloc[0] * 0 + best, color=GRID, lw=0.8)
    ax.set_xlabel("Mean PR-AUC across forward-chaining folds")
    ax.set_title(
        "Resampling policies compared, with resampling applied inside each fold\nUndersampling throws away 97% of the data and the loss is visible"
    )
    ax.set_xlim(0, max(ordered["mean_pr_auc"]) * 1.42)
    return _save(fig, "07_sampling_comparison.png", SOURCE)


def plot_leakage_ablation(table: pd.DataFrame) -> Path:
    variants = table[table["variant"].str.startswith(("A_", "B_", "C_", "D_"))].copy()
    variants["label"] = variants["variant"].str.replace("_", " ", regex=False).str.slice(2)
    # A and B are the leaky protocols, C and D the honest ones.
    colours = [ACCENT, ACCENT, ACCENT_2, ACCENT_4][: len(variants)]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.barh(variants["label"], variants["mean_pr_auc"], color=colours)
    for bar, value in zip(bars, variants["mean_pr_auc"], strict=True):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8.5)
    total = table[table["variant"] == "A_minus_C_total_optimism"]["mean_pr_auc"]
    if len(total):
        ax.set_title(
            "Four validation protocols, one model, one dataset\n"
            f"PR-AUC moves by {float(total.iloc[0]):+.3f} purely because of how the split was drawn"
        )
    ax.invert_yaxis()
    ax.set_xlabel("Mean PR-AUC")
    ax.set_xlim(0, max(variants["mean_pr_auc"]) * 1.22)
    return _save(fig, "08_leakage_ablation.png", SOURCE)


def plot_shap_global(table: pd.DataFrame, top_k: int = 15) -> Path:
    top = table.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    colours = [ACCENT if value > 0 else ACCENT_2 for value in top["mean_shap"]]
    ax.barh(top["feature"], top["mean_abs_shap"], color=colours, alpha=0.9)
    ax.set_xlabel("Mean |SHAP contribution| to the model's log-odds")
    ax.set_title(
        "Global feature attribution (exact TreeSHAP)\nRed = pushes scores up on average, blue = pushes them down"
    )
    return _save(fig, "09_shap_global.png", SOURCE)


def plot_shap_case(attribution: pd.DataFrame, score: float, title: str, name: str) -> Path:
    ordered = attribution.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    colours = [ACCENT if value > 0 else ACCENT_2 for value in ordered["shap_contribution"]]
    labels = [
        f"{row['feature']} = {row['value']:.3g}"
        if isinstance(row["value"], (int, float, np.floating))
        else f"{row['feature']} = {row['value']}"
        for _, row in ordered.iterrows()
    ]
    ax.barh(labels, ordered["shap_contribution"], color=colours)
    ax.axvline(0, color=FG, lw=0.9)
    ax.set_xlabel("Contribution to log-odds")
    ax.set_title(f"{title}\nModel output {score:.4f} - contributions are additive and sum to the log-odds")
    return _save(fig, name, SOURCE)


def plot_psi_heatmap(table: pd.DataFrame, top_k: int = 15) -> Path:
    top = table.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    colours = []
    for verdict in top["verdict"]:
        colours.append({"stable": ACCENT_4, "watch": "#d68910", "retrain": ACCENT}.get(verdict, MUTED))
    ax.barh(top["feature"], top["psi"], color=colours)
    ax.axvline(0.10, color="#d68910", ls="--", lw=1.1)
    ax.axvline(0.25, color=ACCENT, ls="--", lw=1.1)
    ax.text(0.101, len(top) - 0.4, "watch 0.10", fontsize=7.5, color="#d68910")
    ax.text(0.251, len(top) - 0.4, "retrain 0.25", fontsize=7.5, color=ACCENT)
    ax.set_xlabel("Population Stability Index, training window versus out-of-time window")
    ax.set_title("Feature drift between the training window and the out-of-time window")
    return _save(fig, "10_feature_drift_psi.png", SOURCE)


def plot_monthly_stability(table: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.plot(
        table["month"],
        table["psi_vs_first_month"],
        "o-",
        color=ACCENT_2,
        lw=1.7,
        ms=4.5,
        label="score PSI vs first month",
    )
    ax.axhline(0.10, color="#d68910", ls="--", lw=1.1, label="watch 0.10")
    ax.axhline(0.25, color=ACCENT, ls="--", lw=1.1, label="retrain 0.25")
    ax2 = ax.twinx()
    ax2.bar(table["month"], table["transactions"], color=GRID, alpha=0.5, zorder=0)
    ax2.set_ylabel("Transactions", color=MUTED)
    ax2.grid(False)
    ax.set_xlabel("Month")
    ax.set_ylabel("Score PSI")
    ax.set_title("Monthly score stability\nDrift arrives as a creep, not a jump, which is why it is monitored monthly")
    ax.legend(loc="upper left")
    return _save(fig, "11_monthly_score_stability.png", SOURCE)


def plot_amount_vs_score(
    amount: np.ndarray,
    score: np.ndarray,
    y: np.ndarray,
    review_threshold: float,
) -> Path:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    y = np.asarray(y)
    amount = np.asarray(amount, dtype=float)
    score = np.asarray(score, dtype=float)
    ax.scatter(amount[y == 0], score[y == 0], s=4, color=ACCENT_2, alpha=0.22, label="legitimate")
    ax.scatter(amount[y == 1], score[y == 1], s=22, color=ACCENT, alpha=0.9, marker="x", label="fraud")
    ax.axhline(review_threshold, color=ACCENT_3, ls=":", lw=1.4, label="review threshold")
    ax.set_xscale("log")
    ax.set_xlabel("Transaction amount (currency units, log scale)")
    ax.set_ylabel("Predicted fraud probability")
    ax.set_title(
        "Score against amount\nThe sub-3-unit cluster high on the axis is card testing, not noise to be trimmed"
    )
    ax.legend(loc="upper right")
    return _save(fig, "12_score_vs_amount.png", SOURCE)


def plot_entity_risk(historical: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    groups = historical.dropna(subset=["bucket"])
    ax.bar(groups["bucket"], groups["fraud_rate"], color=ACCENT_2)
    for index, (_, row) in enumerate(groups.iterrows()):
        ax.text(
            index,
            row["fraud_rate"] + 0.0001,
            f"{row['fraud_rate']:.3%}\nn={int(row['n']):,}",
            ha="center",
            fontsize=7.5,
        )
    ax.set_xlabel("Resolution-lagged card fraud rate bucket (60-day lag)")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title(
        "A card's own resolved history is the strongest single signal\n"
        "Built only from chargebacks that would already have settled at scoring time"
    )
    ax.set_ylim(0, groups["fraud_rate"].max() * 1.35)
    return _save(fig, "13_historical_entity_risk.png", SOURCE)


def plot_review_band_sensitivity(table: pd.DataFrame) -> Path:
    """Does the manual queue pay for itself, and at what analyst cost does it start to?"""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for catch, colour in zip(
        sorted(table["review_catch_rate"].unique(), reverse=True),
        _model_colours(table["review_catch_rate"].nunique()),
        strict=True,
    ):
        subset = table[table["review_catch_rate"] == catch].sort_values("review_cost")
        ax.plot(
            subset["review_cost"],
            subset["review_band_saving"],
            marker="o",
            lw=1.8,
            color=colour,
            label=f"review stops {catch:.0%} of the fraud it finds",
        )
    ax.axhline(0.0, color=MUTED, ls="--", lw=1.2)
    ax.set_xscale("log")
    ax.set_xticks(sorted(table["review_cost"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Cost of reviewing one alert (currency units, log scale)")
    ax.set_ylabel("Saving from the review band vs approve/decline")
    ax.set_title(
        "What a manual review queue is worth, as its price changes\n"
        "Above the dashed line a three-way rule beats approve-or-decline; below it the queue is not worth funding"
    )
    ax.legend(loc="lower left")
    return _save(fig, "14_review_band_sensitivity.png", SOURCE)
