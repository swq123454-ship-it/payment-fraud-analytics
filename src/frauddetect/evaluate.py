"""Ranking metrics, probability calibration and cost-sensitive decisioning.

Three things in this module exist because the obvious alternative is wrong on
imbalanced data:

1. **Ranking is measured with PR-AUC.** At a 0.35% base rate a model can reach
   0.95 ROC-AUC while doing nothing useful, because the negative class dominates
   the false-positive rate. Precision-recall curves are computed against the
   minority class and cannot be inflated that way.

2. **The decision rule is chosen by cost, not by a 0.5 probability.** The
   thresholds here minimise expected money lost plus money spent on analysts.

3. **Policy optimisation is O(n), not O(n^2).** Solving for two thresholds by
   brute force over a grid is quadratic; the prefix-sum / suffix-minimum
   formulation below finds the global optimum in a single sorted pass, which is
   what makes it practical to re-solve whenever costs change.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from . import config as cfg


def _cost_model(cost_model):
    """Resolve the cost model, falling back to the configured one.

    It is threaded through as an argument rather than read from the module
    global so that the sensitivity study can price a different queue without
    monkey-patching process state.
    """
    return cfg.COST_MODEL if cost_model is None else cost_model


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------
def ks_statistic(y_true: np.ndarray, score: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation between the two score distributions."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def recall_at_capacity(y_true: np.ndarray, score: np.ndarray, capacity: float) -> float:
    """Share of fraud caught if the top ``capacity`` share of alerts is worked.

    This is the metric a fraud operations team actually plans against: headcount
    is fixed, so the question is never "what is your AUC" but "how much of the
    loss do you stop with the analysts you have".
    """
    y_true = np.asarray(y_true)
    n_alerts = max(int(round(len(y_true) * capacity)), 1)
    order = np.argsort(-np.asarray(score), kind="stable")[:n_alerts]
    positives = y_true.sum()
    if positives == 0:
        return float("nan")
    return float(y_true[order].sum() / positives)


def precision_at_capacity(y_true: np.ndarray, score: np.ndarray, capacity: float) -> float:
    y_true = np.asarray(y_true)
    n_alerts = max(int(round(len(y_true) * capacity)), 1)
    order = np.argsort(-np.asarray(score), kind="stable")[:n_alerts]
    return float(y_true[order].mean())


def lift_at_capacity(y_true: np.ndarray, score: np.ndarray, capacity: float) -> float:
    base = float(np.asarray(y_true).mean())
    if base == 0:
        return float("nan")
    return precision_at_capacity(y_true, score, capacity) / base


def bootstrap_pr_auc(
    y_true: np.ndarray,
    score: np.ndarray,
    n_resamples: int = 200,
    seed: int = cfg.SEED,
) -> tuple[float, float, float]:
    """Percentile interval for average precision.

    A point estimate of PR-AUC on 60 frauds is not a result, it is a coin flip.
    The interval is reported alongside every headline number for that reason.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    n = len(y_true)
    values = []
    for _ in range(n_resamples):
        index = rng.integers(0, n, n)
        if y_true[index].sum() == 0:
            continue
        values.append(average_precision_score(y_true[index], score[index]))
    if not values:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(values)),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def ranking_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    capacities: tuple[float, ...] = cfg.REVIEW_CAPACITY_GRID,
    with_interval: bool = False,
    seed: int = cfg.SEED,
) -> dict:
    """Everything that matters on a rare-event problem, in one dictionary.

    ``accuracy`` is deliberately absent. On this data an all-negative classifier
    is 99.65% accurate and catches nothing.
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score)

    metrics: dict[str, float | dict] = {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "base_rate": round(float(y_true.mean()), 6),
        "pr_auc": round(float(average_precision_score(y_true, score)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, score)), 4),
        "ks": round(ks_statistic(y_true, score), 4),
        "brier": round(float(brier_score_loss(y_true, np.clip(score, 0, 1))), 6),
        "pr_auc_baseline": round(float(y_true.mean()), 6),
        "recall_at_capacity": {f"{c:.3%}": round(recall_at_capacity(y_true, score, c), 4) for c in capacities},
        "precision_at_capacity": {f"{c:.3%}": round(precision_at_capacity(y_true, score, c), 4) for c in capacities},
        # Lift is the number a fraud manager repeats back: "the queue is N times
        # denser in fraud than a random sample of the same size". Reported so
        # that it is read off the pipeline rather than divided by hand.
        "lift_at_capacity": {f"{c:.3%}": round(lift_at_capacity(y_true, score, c), 2) for c in capacities},
    }
    if with_interval:
        mean, low, high = bootstrap_pr_auc(y_true, score, seed=seed)
        metrics["pr_auc_bootstrap"] = {
            "mean": round(mean, 4),
            "ci_low": round(low, 4),
            "ci_high": round(high, 4),
        }
    return metrics


def pr_curve_points(y_true: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    return pd.DataFrame(
        {
            "threshold": np.append(thresholds, np.nan),
            "precision": precision,
            "recall": recall,
        }
    )


def roc_curve_points(y_true: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    fpr, tpr, thresholds = roc_curve(y_true, score)
    return pd.DataFrame({"threshold": thresholds, "false_positive_rate": fpr, "true_positive_rate": tpr})


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
@dataclass
class Calibrator:
    """Isotonic probability calibration, fitted on data the model never saw.

    Calibration is not cosmetic here. The cost-optimal threshold is a function
    of the predicted probabilities, so a model that ranks well but is
    miscalibrated by a factor of five produces a threshold that is wrong by a
    factor of five - and every downstream money figure inherits that error.
    """

    model: IsotonicRegression | None = None
    fitted_on: str = "held-out"

    def fit(self, y_true: np.ndarray, score: np.ndarray) -> Calibrator:
        self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.model.fit(score, y_true)
        return self

    def apply(self, score: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.asarray(score, dtype=float)
        return np.asarray(self.model.predict(score), dtype=float)


def reliability_table(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 12) -> pd.DataFrame:
    """Observed versus predicted fraud rate per probability bucket."""
    frame = pd.DataFrame({"y": y_true, "p": prob})
    frame["bucket"] = pd.qcut(frame["p"].rank(method="first"), n_bins, labels=False)
    grouped = frame.groupby("bucket").agg(
        n=("y", "size"),
        mean_predicted=("p", "mean"),
        observed_rate=("y", "mean"),
        min_score=("p", "min"),
        max_score=("p", "max"),
    )
    grouped["calibration_gap"] = grouped["observed_rate"] - grouped["mean_predicted"]
    return grouped.reset_index()


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 12) -> float:
    table = reliability_table(y_true, prob, n_bins)
    weights = table["n"] / table["n"].sum()
    return float((weights * table["calibration_gap"].abs()).sum())


# ---------------------------------------------------------------------------
# Cost-sensitive decisioning
# ---------------------------------------------------------------------------
@dataclass
class Policy:
    """A decision rule: two thresholds and the money it is expected to lose."""

    review_threshold: float
    decline_threshold: float
    expected_cost: float
    cost_per_10k: float
    loss_approved_fraud: float
    review_operating_cost: float
    decline_friction_cost: float
    n_approved: int
    n_reviewed: int
    n_declined: int
    fraud_approved: int
    fraud_reviewed: int
    fraud_declined: int
    legitimate_reviewed: int
    legitimate_declined: int
    review_rate: float = 0.0
    decline_rate: float = 0.0
    review_capacity_binding: bool = False


def _score_groups(y_true: np.ndarray, amount: np.ndarray, score: np.ndarray, cost_model=None) -> dict:
    """Collapse the book into its distinct score values, with prefix sums.

    Cuts are taken **between distinct scores**, never between two rows that share
    a score. That is not an optimisation, it is what makes the answer honest: a
    threshold rule can only express ``score >= t``, so two rows with the same
    score can never be sent to different actions. Optimising over per-row
    positions would let the search "decide" to split a tied block, report the
    gain from doing so, and then be unable to implement it - which is exactly how
    a two-threshold policy ends up costing more than the single-threshold policy
    it is supposed to dominate. Grouping first means the index cost and the
    realised cost are the same number by construction.

    It is also what makes calibration usable at all. Isotonic calibration maps
    many raw scores onto one probability, so the flagged tail is a handful of
    wide tiers; on a 45,000-row book the calibrated score can take fewer than
    forty distinct values. Cutting at row positions inside a tier is meaningless,
    cutting at tier boundaries is not.
    """
    y_true = np.asarray(y_true).astype(float)
    amount = np.asarray(amount, dtype=float)
    score = np.asarray(score, dtype=float)
    order = np.argsort(score, kind="stable")

    y = y_true[order]
    amt = amount[order]
    scr = score[order]

    cost_model = _cost_model(cost_model)
    fraud_loss = np.where(y == 1, amt * cost_model.fraud_loss_ratio, 0.0)
    reviewable = (amt >= cost_model.min_review_amount).astype(float)

    # Cost of placing a row in the review band. An analyst costs the same whether
    # the alert is a fraud or a false positive, but a fraudulent alert is only
    # *stopped* with probability ``review_catch_rate``: the rest has already
    # authorised by the time anyone looks. A flagged amount too small to be worth
    # reviewing is approved instead, and carries its full loss.
    band_cost = (
        reviewable * (cost_model.review_cost + fraud_loss * (1.0 - cost_model.review_catch_rate))
        + (1.0 - reviewable) * fraud_loss
    )
    legitimate = 1.0 - y

    unique_scores, inverse = np.unique(scr, return_inverse=True)
    n_groups = len(unique_scores)

    group_loss = np.bincount(inverse, weights=fraud_loss, minlength=n_groups)
    group_band = np.bincount(inverse, weights=band_cost, minlength=n_groups)
    group_legit = np.bincount(inverse, weights=legitimate, minlength=n_groups)
    group_rows = np.bincount(inverse, minlength=n_groups)

    zeros = np.zeros(1)
    return {
        "y": y,
        "amount": amt,
        "score": scr,
        "unique_scores": unique_scores,
        "n_groups": n_groups,
        # Prefix arrays are indexed by group boundary: entry g covers groups [0, g).
        "prefix_loss": np.concatenate([zeros, np.cumsum(group_loss)]),
        "prefix_band": np.concatenate([zeros, np.cumsum(group_band)]),
        "prefix_legit": np.concatenate([zeros, np.cumsum(group_legit)]),
        "prefix_rows": np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(group_rows, dtype=np.int64)]),
        "n": len(y),
    }


def optimise_policy(
    y_true: np.ndarray,
    amount: np.ndarray,
    score: np.ndarray,
    allow_review: bool = True,
    max_review_rate: float | None = None,
    cost_model=None,
) -> Policy:
    """Find the cost-minimising (review, decline) threshold pair.

    With actions ordered approve < review < decline and costs that are additive
    over the three bands, the total cost of any pair of cut points decomposes into
    a term depending on the first cut plus a term depending on the second, so a
    suffix minimum yields the global optimum in one pass.

    Three actions exist rather than two because reviewing and declining are not
    substitutes. A review costs an analyst's time and only stops the fraud with
    probability ``review_catch_rate``; a decline stops it outright but burns
    goodwill when the customer was innocent. Where the score is decisive, decline
    wins; just below that, review wins; below that, approving a legitimate
    transaction costs nothing at all.
    """
    cost_model = _cost_model(cost_model)
    max_review_rate = cost_model.max_review_rate if max_review_rate is None else max_review_rate
    data = _score_groups(y_true, amount, score, cost_model)
    n = data["n"]
    k = data["n_groups"]
    prefix_loss, prefix_band, prefix_legit = (
        data["prefix_loss"],
        data["prefix_band"],
        data["prefix_legit"],
    )
    total_legit = prefix_legit[-1]
    bounds = slice(0, k + 1)

    # --- single threshold: approve below, decline at or above ----------------
    single_cost = prefix_loss[bounds] + (total_legit - prefix_legit[bounds]) * cost_model.decline_friction_cost
    best_single = int(np.argmin(single_cost))
    best = (best_single, best_single, float(single_cost[best_single]), False)

    # --- two thresholds: approve / review / decline --------------------------
    if allow_review and k > 1:
        suffix_term = prefix_band - prefix_legit * cost_model.decline_friction_cost

        # Best second cut for every possible first cut, in one backward scan.
        suffix_best = np.empty(k + 1, dtype=np.int64)
        best_index = k
        best_value = suffix_term[k]
        suffix_best[k] = k
        for i in range(k - 1, -1, -1):
            if suffix_term[i] <= best_value:
                best_value = suffix_term[i]
                best_index = i
            suffix_best[i] = best_index

        unconstrained = (
            prefix_loss[bounds]
            - prefix_band[bounds]
            + suffix_term[suffix_best]
            + total_legit * cost_model.decline_friction_cost
        )
        best_two = int(np.argmin(unconstrained))
        best_b = int(suffix_best[best_two])
        window = max(int(round(max_review_rate * n)), 1)
        binding = int(data["prefix_rows"][best_b] - data["prefix_rows"][best_two]) > window

        if not binding:
            if unconstrained[best_two] < best[2]:
                best = (best_two, best_b, float(unconstrained[best_two]), False)
        else:
            best_two, best_b, constrained_cost = _constrained_two_threshold(data, window, cost_model)
            if constrained_cost < best[2]:
                best = (best_two, best_b, constrained_cost, True)

    return _describe_policy(data, best[0], best[1], allow_review, best[3], cost_model)


def _constrained_two_threshold(data: dict, window: int, cost_model=None) -> tuple[int, int, float]:
    """Global optimum of the review-band-capped problem, in O(k).

    The cap is on the number of *transactions* the queue can work, but the cuts
    live on score tiers, and tiers hold different numbers of rows. The largest
    legal second cut for a given first cut is therefore a monotone function of
    the first cut, which lets a deque maintain the running minimum of
    ``prefix_band - prefix_legit * friction`` over a window whose right edge only
    advances and whose left edge is dropped as it falls behind.
    """
    prefix_loss = data["prefix_loss"]
    prefix_band = data["prefix_band"]
    prefix_legit = data["prefix_legit"]
    prefix_rows = data["prefix_rows"]
    k = data["n_groups"]
    friction = _cost_model(cost_model).decline_friction_cost
    total_legit = prefix_legit[-1]
    suffix_term = prefix_band - prefix_legit * friction

    best_cost = np.inf
    best_pair = (k, k)
    queue: deque[int] = deque()
    right = -1

    for first in range(k + 1):
        # Largest second cut whose band still fits inside the queue's capacity.
        limit = int(np.searchsorted(prefix_rows, prefix_rows[first] + window, side="right")) - 1
        limit = min(limit, k)
        while right < limit:
            right += 1
            while queue and suffix_term[queue[-1]] >= suffix_term[right]:
                queue.pop()
            queue.append(right)
        while queue and queue[0] < first:
            queue.popleft()
        if not queue:
            continue
        second = queue[0]
        candidate = prefix_loss[first] - prefix_band[first] + suffix_term[second] + total_legit * friction
        if candidate < best_cost:
            best_cost = float(candidate)
            best_pair = (first, second)

    return best_pair[0], best_pair[1], best_cost


def _cut_to_threshold(unique_scores: np.ndarray, cut: int) -> float:
    """Turn a cut between score tiers into an applicable threshold.

    A row is *reviewed* when ``score >= threshold``, so a cut at tier ``i`` needs
    a threshold strictly above ``unique_scores[i-1]`` and at most
    ``unique_scores[i]``; the midpoint is that, and the two ends are sentinels.
    Because the tiers are distinct values, ``score >= threshold`` reproduces the
    intended partition exactly - which is the property the whole optimisation
    rests on.
    """
    k = len(unique_scores)
    if cut <= 0:
        return 0.0
    if cut >= k:
        return float("inf")
    return float(0.5 * (unique_scores[cut - 1] + unique_scores[cut]))


def apply_policy(
    score: np.ndarray,
    review_threshold: float,
    decline_threshold: float,
    amount: np.ndarray | None = None,
    cost_model=None,
) -> dict[str, np.ndarray]:
    """The single place where a policy is turned into three masks."""
    score = np.asarray(score, dtype=float)
    flagged = score >= review_threshold
    declined = score >= decline_threshold
    if amount is None:
        reviewable = np.ones(len(score), dtype=bool)
    else:
        reviewable = np.asarray(amount, dtype=float) >= _cost_model(cost_model).min_review_amount
    reviewed = flagged & ~declined & reviewable
    approved = ~flagged | (flagged & ~declined & ~reviewable)
    return {"approved": approved, "reviewed": reviewed, "declined": declined}


def _describe_policy(
    data: dict,
    review_cut: int,
    decline_cut: int,
    allow_review: bool,
    capacity_binding: bool = False,
    cost_model=None,
) -> Policy:
    """Turn two cut indices back into thresholds plus a full money breakdown."""
    cost_model = _cost_model(cost_model)
    y, amount, score = data["y"], data["amount"], data["score"]
    n = data["n"]

    review_threshold = _cut_to_threshold(data["unique_scores"], review_cut)
    decline_threshold = _cut_to_threshold(data["unique_scores"], decline_cut)
    if not allow_review or review_cut == decline_cut:
        review_threshold = decline_threshold

    masks = apply_policy(score, review_threshold, decline_threshold, amount, cost_model)
    approved, reviewed, declined = masks["approved"], masks["reviewed"], masks["declined"]

    loss_approved = float((amount * y * approved * cost_model.fraud_loss_ratio).sum())
    review_cost = float(
        reviewed.sum() * cost_model.review_cost
        + (amount * y * reviewed * cost_model.fraud_loss_ratio * (1.0 - cost_model.review_catch_rate)).sum()
    )
    friction = float(((1.0 - y) * declined).sum() * cost_model.decline_friction_cost)

    total = loss_approved + review_cost + friction

    return Policy(
        review_threshold=review_threshold,
        decline_threshold=decline_threshold,
        expected_cost=round(total, 2),
        cost_per_10k=round(total / max(n, 1) * 10_000, 2),
        loss_approved_fraud=round(loss_approved, 2),
        review_operating_cost=round(review_cost, 2),
        decline_friction_cost=round(friction, 2),
        n_approved=int(approved.sum()),
        n_reviewed=int(reviewed.sum()),
        n_declined=int(declined.sum()),
        fraud_approved=int(y[approved].sum()),
        fraud_reviewed=int(y[reviewed].sum()),
        fraud_declined=int(y[declined].sum()),
        legitimate_reviewed=int((1 - y)[reviewed].sum()),
        legitimate_declined=int((1 - y)[declined].sum()),
        review_rate=round(float(reviewed.mean()), 6),
        decline_rate=round(float(declined.mean()), 6),
        review_capacity_binding=bool(capacity_binding),
    )


def policy_table(
    y_true: np.ndarray,
    amount: np.ndarray,
    score: np.ndarray,
    thresholds: np.ndarray,
    cost_model=None,
) -> pd.DataFrame:
    """Cost of a single-threshold approve/decline rule across a threshold grid."""
    cost_model = _cost_model(cost_model)
    data = _score_groups(y_true, amount, score, cost_model)
    prefix_loss, prefix_legit = data["prefix_loss"], data["prefix_legit"]
    sorted_scores = data["score"]
    unique_scores = data["unique_scores"]
    n = data["n"]
    friction_cost = cost_model.decline_friction_cost

    rows = []
    for threshold in thresholds:
        # Two indices are needed: the prefix arrays are indexed by score tier,
        # while the reported counts are in transactions.
        cut = int(np.searchsorted(unique_scores, threshold, side="left"))
        approved = int(np.searchsorted(sorted_scores, threshold, side="left"))
        loss = float(prefix_loss[cut])
        friction = float((prefix_legit[-1] - prefix_legit[cut]) * friction_cost)
        rows.append(
            {
                "threshold": float(threshold),
                "approved": approved,
                "declined": n - approved,
                "approved_fraud_loss": round(loss, 2),
                "decline_friction": round(friction, 2),
                "total_cost": round(loss + friction, 2),
                "declined_share": round((n - approved) / n, 6),
            }
        )
    return pd.DataFrame(rows)


def policy_outcome(
    y_true: np.ndarray,
    amount: np.ndarray,
    score: np.ndarray,
    review_threshold: float,
    decline_threshold: float,
    cost_model=None,
) -> dict:
    """Apply a *fixed* policy and report what actually happened.

    Thresholds are always fitted on a validation window and then applied
    unchanged to the out-of-time window. Re-tuning them on the holdout would
    turn an honest estimate into a fitted number.
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    amount = np.asarray(amount, dtype=float)
    cost_model = _cost_model(cost_model)

    masks = apply_policy(score, review_threshold, decline_threshold, amount, cost_model)
    approved_mask, reviewed_mask, declined_mask = (
        masks["approved"],
        masks["reviewed"],
        masks["declined"],
    )

    loss = float((amount * y_true * approved_mask * cost_model.fraud_loss_ratio).sum())
    # A reviewed fraud is only stopped with probability ``review_catch_rate``; the
    # remainder settles before the analyst reaches it and is still a loss.
    review = float(
        reviewed_mask.sum() * cost_model.review_cost
        + (amount * y_true * reviewed_mask * cost_model.fraud_loss_ratio * (1.0 - cost_model.review_catch_rate)).sum()
    )
    friction = float(((1 - y_true) * declined_mask).sum() * cost_model.decline_friction_cost)
    total = loss + review + friction

    total_fraud = int(y_true.sum())
    caught = int(y_true[reviewed_mask].sum() * cost_model.review_catch_rate + y_true[declined_mask].sum())

    return {
        "expected_cost": round(total, 2),
        "cost_per_10k": round(total / max(len(y_true), 1) * 10_000, 2),
        "cost_as_pct_of_processed_amount": round(100.0 * total / max(amount.sum(), 1.0), 4),
        "loss_approved_fraud": round(loss, 2),
        "review_operating_cost": round(review, 2),
        "decline_friction_cost": round(friction, 2),
        "n_approved": int(approved_mask.sum()),
        "n_reviewed": int(reviewed_mask.sum()),
        "n_declined": int(declined_mask.sum()),
        "review_rate": round(float(reviewed_mask.mean()), 6),
        "decline_rate": round(float(declined_mask.mean()), 6),
        "fraud_caught": caught,
        "fraud_total": total_fraud,
        "fraud_capture_rate": round(caught / max(total_fraud, 1), 4),
        "false_positive_rate": round(float(declined_mask[~y_true.astype(bool)].mean()), 6),
        "value_saved_vs_approve_all": round(float((amount * y_true).sum()) - loss, 2),
    }


def review_band_sensitivity(
    y_true: np.ndarray,
    amount: np.ndarray,
    score: np.ndarray,
    review_costs: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 25.0, 50.0),
    catch_rates: tuple[float, ...] = (0.30, 0.50, 0.70, 0.90),
) -> pd.DataFrame:
    """When is a manual review queue worth paying for?

    A three-way policy is only worth having if the review band earns its keep,
    and whether it does depends on two numbers a fraud team can actually
    negotiate: what an analyst hour costs, and how much of a reviewed fraud is
    stopped before the money leaves. Rather than assert that a queue is
    valuable, this sweeps both and reports what the optimum does.

    The output is the answer to a question that comes up in every build-versus-buy
    conversation: "at our analyst cost, does the queue pay for itself, and if not,
    how cheap would it have to get?" Each row also carries the saving of the
    two-threshold optimum over the single-threshold optimum, so the point at
    which the band opens is visible as a sign change rather than inferred.
    """
    rows = []
    for review_cost in review_costs:
        for catch_rate in catch_rates:
            model = dataclasses.replace(
                cfg.COST_MODEL, review_cost=float(review_cost), review_catch_rate=float(catch_rate)
            )
            two = optimise_policy(y_true, amount, score, allow_review=True, cost_model=model)
            one = optimise_policy(y_true, amount, score, allow_review=False, cost_model=model)
            rows.append(
                {
                    "review_cost": float(review_cost),
                    "review_catch_rate": float(catch_rate),
                    "review_band_open": bool(two.n_reviewed > 0),
                    "n_reviewed": two.n_reviewed,
                    "n_declined": two.n_declined,
                    "review_rate": round(float(two.review_rate), 6),
                    "two_threshold_cost": two.expected_cost,
                    "single_threshold_cost": one.expected_cost,
                    "review_band_saving": round(one.expected_cost - two.expected_cost, 2),
                }
            )
    return pd.DataFrame(rows)


def approve_all_cost(y_true: np.ndarray, amount: np.ndarray) -> dict:
    """The do-nothing benchmark every fraud project should be measured against."""
    y_true = np.asarray(y_true)
    amount = np.asarray(amount, dtype=float)
    loss = float((amount * y_true).sum())
    return {
        "expected_cost": round(loss, 2),
        "cost_per_10k": round(loss / max(len(y_true), 1) * 10_000, 2),
        "cost_as_pct_of_processed_amount": round(100.0 * loss / max(amount.sum(), 1.0), 4),
    }
