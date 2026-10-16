"""Model explanation: global importance and per-case attribution.

"Feature importance" is the weakest form of explanation, and in an interview it
is the easiest to puncture - a gain ranking tells you what the model used, not
why *this* transaction was declined. Two things are therefore produced:

* a global view, with SHAP magnitude compared against permutation importance so
  that a disagreement between the two is visible rather than hidden;
* a per-case attribution using exact TreeSHAP contributions, which turns an
  alert into a sentence an analyst can act on.

Contributions are computed on the raw margin, so they add up to the model's
log-odds. That additivity is the property that makes the attribution defensible:
the analyst can check the arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import config as cfg


@dataclass
class ShapExplainer:
    """Exact TreeSHAP contributions via LightGBM's own implementation.

    LightGBM's ``pred_contrib=True`` is the reference implementation of the same
    algorithm the ``shap`` package wraps, and it consumes the booster's native
    categorical encoding directly. ``shap.TreeExplainer`` is used in the test
    suite as an independent cross-check that the two agree.
    """

    model: object
    feature_names: list[str]
    base_value: float = 0.0

    @classmethod
    def from_model(cls, model, feature_names: list[str]) -> ShapExplainer:
        return cls(model=model, feature_names=list(feature_names))

    def contributions(self, X: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
        """Per-row, per-feature contribution to the raw margin."""
        frame = X if max_rows is None or len(X) <= max_rows else X.iloc[:max_rows]
        raw = self.model.booster_.predict(frame, pred_contrib=True)
        raw = np.asarray(raw)
        contributions = raw[:, :-1]
        self.base_value = float(raw[0, -1])
        return pd.DataFrame(contributions, columns=self.feature_names, index=frame.index)

    def shap_values_library(self, X: pd.DataFrame) -> np.ndarray:
        """Same quantity computed through the ``shap`` package.

        Kept as an explicit, callable cross-check rather than the primary path:
        two independent implementations producing the same numbers is the
        strongest available evidence that the attribution is right.
        """
        import shap

        explainer = shap.TreeExplainer(self.model)
        values = explainer.shap_values(X)
        if isinstance(values, list):
            values = values[-1]
        return np.asarray(values)


def global_importance(
    explainer: ShapExplainer,
    X: pd.DataFrame,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Mean absolute SHAP contribution per feature, signed mean included.

    The signed mean separates a feature that pushes scores up on average from one
    that is merely noisy in both directions - a distinction a magnitude-only
    ranking hides.
    """
    contributions = explainer.contributions(X)
    table = pd.DataFrame(
        {
            "feature": explainer.feature_names,
            "mean_abs_shap": contributions.abs().mean().to_numpy(dtype=float),
            "mean_shap": contributions.mean().to_numpy(dtype=float),
            "max_abs_shap": contributions.abs().max().to_numpy(dtype=float),
        }
    )
    table = table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    table["share_of_total"] = (table["mean_abs_shap"] / table["mean_abs_shap"].sum()).round(6)
    return table if top_k is None else table.head(top_k)


def permutation_table(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int = 5,
    max_rows: int = 20_000,
    seed: int = cfg.SEED,
) -> pd.DataFrame:
    """Permutation importance on PR-AUC, computed column by column.

    Two deliberate choices. First, the score is average precision rather than
    accuracy: at a 0.35% base rate, shuffling an important feature barely moves
    accuracy while it visibly collapses the precision-recall curve.

    Second, the shuffle is done directly on the DataFrame so that a categorical
    column is permuted *within* its own category set. Converting the design
    matrix to a NumPy array first would silently turn LightGBM's native
    categorical splits into numeric comparisons, and the attributions would then
    describe a different model from the one being shipped.
    """
    if len(X) > max_rows:
        rng = np.random.default_rng(seed)
        index = rng.choice(len(X), max_rows, replace=False)
        X = X.iloc[np.sort(index)].reset_index(drop=True)
        y = np.asarray(y)[np.sort(index)]

    baseline = float(average_precision_score(y, model.predict_proba(X)[:, 1]))
    rng = np.random.default_rng(seed)
    rows = []

    for column in X.columns:
        series = X[column]
        is_categorical = isinstance(series.dtype, pd.CategoricalDtype)
        categories = list(series.cat.categories) if is_categorical else None
        deltas = []
        for _ in range(n_repeats):
            values = series.to_numpy(copy=True)
            rng.shuffle(values)
            shuffled = X.copy()
            shuffled[column] = pd.Categorical(values, categories=categories) if is_categorical else values
            score = float(average_precision_score(y, model.predict_proba(shuffled)[:, 1]))
            deltas.append(baseline - score)
        rows.append(
            {
                "feature": column,
                "permutation_importance": float(np.mean(deltas)),
                "permutation_sd": float(np.std(deltas)),
                "baseline_pr_auc": round(baseline, 4),
            }
        )

    return pd.DataFrame(rows).sort_values("permutation_importance", ascending=False).reset_index(drop=True)


def case_attribution(
    explainer: ShapExplainer,
    frame: pd.DataFrame,
    top_k: int = 6,
) -> pd.DataFrame:
    """Why this one transaction scored what it did.

    ``frame`` is a one-row slice of the design matrix rather than a Series:
    transposing a Series through ``to_frame().T`` would coerce the mixed dtypes
    to ``object`` and LightGBM would reject the frame as a mismatch with the
    categorical encoding it was trained on.

    Returns the largest contributions in absolute value, with the feature's value
    on this row and the direction of the push, which is the form a fraud analyst
    can act on: "the card had 14 transactions in the last hour, and this device
    has never been seen before".
    """
    if len(frame) != 1:
        raise ValueError("case_attribution expects exactly one row.")
    row = frame.iloc[0]
    contribution_row = explainer.contributions(frame).iloc[0]
    table = pd.DataFrame(
        {
            "feature": explainer.feature_names,
            "value": [row[name] for name in explainer.feature_names],
            "shap_contribution": contribution_row.to_numpy(dtype=float),
        }
    )
    table["abs_contribution"] = table["shap_contribution"].abs()
    table["direction"] = np.where(table["shap_contribution"] > 0, "increases risk", "decreases risk")
    return table.sort_values("abs_contribution", ascending=False).head(top_k).reset_index(drop=True)


def _render_value(value) -> str:
    """Render a feature value the way a person would say it out loud.

    A missing entity-history feature is the single most common thing in this
    design matrix, and it is also the strongest signal in the model. Printing it
    as ``nan`` buries that; printing it as ``missing`` tells the analyst exactly
    what the model reacted to.
    """
    if value is None:
        return "missing"
    if isinstance(value, float) and np.isnan(value):
        return "missing"
    if isinstance(value, np.floating) and np.isnan(value):
        return "missing"
    if isinstance(value, (int, float, np.number)):
        return f"{float(value):.3g}"
    return str(value)


def case_narrative(explainer: ShapExplainer, frame: pd.DataFrame, top_k: int = 4) -> str:
    """A one-paragraph explanation suitable for an alert queue."""
    attribution = case_attribution(explainer, frame, top_k=top_k)
    margin = float(attribution["shap_contribution"].sum()) + explainer.base_value
    parts = []
    for _, entry in attribution.iterrows():
        parts.append(
            f"{entry['feature']}={_render_value(entry['value'])} "
            f"({entry['direction']}, {entry['shap_contribution']:+.3f})"
        )
    return (
        f"Score log-odds {margin:+.3f} against a base of {explainer.base_value:+.3f}. "
        f"Largest drivers: " + "; ".join(parts) + "."
    )


def alert_queue(
    explainer: ShapExplainer,
    X: pd.DataFrame,
    score: np.ndarray,
    y: np.ndarray,
    amount: np.ndarray,
    top_n: int = 5,
) -> pd.DataFrame:
    """The analyst-facing view: highest-risk cases with reasons attached."""
    order = np.argsort(-np.asarray(score), kind="stable")[:top_n]
    rows = []
    for rank, position in enumerate(order, start=1):
        frame = X.iloc[[int(position)]]
        rows.append(
            {
                "rank": rank,
                "score": round(float(score[position]), 4),
                "amount": round(float(amount[position]), 2),
                "actual_fraud": int(y[position]),
                "top_driver_1": _driver(explainer, frame, 0),
                "top_driver_2": _driver(explainer, frame, 1),
                "top_driver_3": _driver(explainer, frame, 2),
            }
        )
    return pd.DataFrame(rows)


def _driver(explainer: ShapExplainer, frame: pd.DataFrame, position: int) -> str:
    attribution = case_attribution(explainer, frame, top_k=position + 1)
    if position >= len(attribution):
        return ""
    entry = attribution.iloc[position]
    return f"{entry['feature']}={_render_value(entry['value'])} ({entry['shap_contribution']:+.2f})"
