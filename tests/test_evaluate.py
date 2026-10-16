"""Metric, calibration and decisioning tests.

The most important test in the repository is
``test_two_threshold_optimum_matches_brute_force``: it verifies that the O(n)
prefix-sum optimiser agrees with an exhaustive search over every cut pair. A
faster algorithm that silently returns a worse policy would be worse than the
slow one it replaces.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pandas as pd
import pytest

from frauddetect import config as cfg
from frauddetect.evaluate import (
    Calibrator,
    _score_groups,
    apply_policy,
    approve_all_cost,
    bootstrap_pr_auc,
    expected_calibration_error,
    ks_statistic,
    lift_at_capacity,
    optimise_policy,
    policy_outcome,
    policy_table,
    precision_at_capacity,
    ranking_metrics,
    recall_at_capacity,
    reliability_table,
    review_band_sensitivity,
)


@pytest.fixture(scope="module")
def toy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A deliberately *overlapping* book.

    The two score distributions overlap, so no single cut is optimal: the very
    top of the range is fraudulent enough to decline, the band just below it is
    worth an analyst's time, and everything under that is approved. A perfectly
    separable toy - which is what a naive ``score = a + b * y + tiny_noise``
    gives - makes the review band empty, and then the flagship test passes
    without ever evaluating a three-way split.
    """
    rng = np.random.default_rng(4)
    n = 4_000
    y = (rng.random(n) < 0.02).astype(int)
    amount = np.round(np.exp(rng.normal(3.6, 1.0, n)), 2)
    # Score correlated with the label so the optimiser has something to find.
    score = np.clip(0.35 + 0.30 * y + rng.normal(0, 0.20, n), 0.0, 1.0)
    return y, amount, score


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------
def test_recall_at_capacity_is_one_when_everything_is_reviewed(toy):
    y, _, score = toy
    assert recall_at_capacity(y, score, 1.0) == pytest.approx(1.0)


def test_recall_at_capacity_is_zero_for_a_perfectly_inverted_score():
    # Built locally and tightly separated: the shared ``toy`` fixture overlaps
    # on purpose, and an inverted score over overlapping classes would still
    # pick up a fraud or two at the very top.
    rng = np.random.default_rng(2)
    n = 2_000
    y = (rng.random(n) < 0.02).astype(int)
    score = 0.5 + 0.40 * y + rng.normal(0, 0.02, n)
    assert recall_at_capacity(y, 1.0 - score, 0.01) == pytest.approx(0.0)


def test_precision_and_lift_at_capacity_move_together(toy):
    y, _, score = toy
    precision = precision_at_capacity(y, score, 0.05)
    assert lift_at_capacity(y, score, 0.05) == pytest.approx(precision / y.mean())


def test_ranking_metrics_reports_pr_auc_above_the_base_rate(toy):
    y, _, score = toy
    metrics = ranking_metrics(y, score)
    assert metrics["pr_auc"] > metrics["base_rate"]
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["ks"] <= 1.0
    assert "accuracy" not in metrics


def test_ranking_metrics_carries_capacity_tables(toy):
    y, _, score = toy
    metrics = ranking_metrics(y, score, capacities=(0.01, 0.05))
    assert set(metrics["recall_at_capacity"]) == {"1.000%", "5.000%"}
    assert set(metrics["precision_at_capacity"]) == {"1.000%", "5.000%"}
    assert set(metrics["lift_at_capacity"]) == {"1.000%", "5.000%"}
    # Lift is precision over the base rate, so the two must agree exactly.
    for label in metrics["lift_at_capacity"]:
        assert metrics["lift_at_capacity"][label] == pytest.approx(
            metrics["precision_at_capacity"][label] / metrics["base_rate"], rel=1e-3
        )


def test_bootstrap_interval_brackets_the_point_estimate(toy):
    y, _, score = toy
    metrics = ranking_metrics(y, score, with_interval=True)
    interval = metrics["pr_auc_bootstrap"]
    assert interval["ci_low"] <= interval["ci_high"]
    assert interval["ci_low"] - 0.01 <= metrics["pr_auc"] <= interval["ci_high"] + 0.01


def test_bootstrap_returns_nan_when_no_resample_has_a_positive(toy):
    y = np.zeros(50, dtype=int)
    mean, low, high = bootstrap_pr_auc(y, np.random.default_rng(0).random(50), n_resamples=5)
    assert np.isnan(mean) and np.isnan(low) and np.isnan(high)


def test_ks_is_zero_for_an_uninformative_score():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4_000)
    score = rng.random(4_000)
    assert ks_statistic(y, score) < 0.1


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def test_calibrator_improves_a_deliberately_distorted_score(toy):
    y, _, score = toy
    distorted = np.clip(score**0.35, 0, 1)  # ranks preserved, probabilities wrong
    before = expected_calibration_error(y, distorted)
    after = expected_calibration_error(y, Calibrator().fit(y, distorted).apply(distorted))
    assert after <= before


def test_calibrator_preserves_the_ordering(toy):
    y, _, score = toy
    calibrated = Calibrator().fit(y, score).apply(score)
    assert np.all(np.diff(calibrated[np.argsort(score, kind="stable")]) >= -1e-12)


def test_identity_calibrator_returns_the_input_unchanged():
    score = np.array([0.1, 0.4, 0.9])
    assert np.allclose(Calibrator().apply(score), score)


def test_reliability_table_covers_the_probability_range(toy):
    y, _, score = toy
    table = reliability_table(y, score, n_bins=6)
    assert len(table) > 0
    assert table["n"].sum() == len(y)
    assert table["min_score"].le(table["max_score"]).all()


# ---------------------------------------------------------------------------
# Decisioning
# ---------------------------------------------------------------------------
def test_apply_policy_partitions_every_row(toy):
    y, amount, score = toy
    masks = apply_policy(score, 0.1, 0.5, amount)
    total = masks["approved"].sum() + masks["reviewed"].sum() + masks["declined"].sum()
    assert total == len(score)


def test_tiny_flagged_amounts_are_approved_not_reviewed():
    score = np.array([0.9, 0.9])
    amount = np.array([cfg.COST_MODEL.min_review_amount - 1.0, cfg.COST_MODEL.min_review_amount + 1.0])
    masks = apply_policy(score, 0.1, 0.99, amount)
    assert masks["approved"][0]
    assert masks["reviewed"][1]


def test_the_toy_book_actually_needs_a_review_band(toy):
    """Guard the fixture.

    If the unconstrained optimum ever becomes a pure single cut, the flagship
    brute-force test below would still pass while testing nothing about the
    three-way split. This assertion fails first, and loudly.
    """
    y, amount, score = toy
    policy = optimise_policy(y, amount, score, allow_review=True, max_review_rate=1.0)
    assert policy.n_reviewed > 0
    assert policy.n_declined > 0
    assert policy.review_threshold < policy.decline_threshold


def test_two_threshold_optimum_matches_brute_force(toy):
    """The prefix-sum optimiser must equal an exhaustive search over all cuts.

    Every pair of score tiers is enumerated directly, using the same cost
    definitions, and the best cost is compared with the one the optimiser
    returns. The capacity cap is switched off so that the unconstrained problem
    is the one under test. If the fast path ever drifts from the specification,
    this fails.
    """
    y, amount, score = toy
    data = _score_groups(y, amount, score)
    k = data["n_groups"]
    loss = data["prefix_loss"]
    band = data["prefix_band"]
    legit = data["prefix_legit"]
    friction = cfg.COST_MODEL.decline_friction_cost

    best = np.inf
    for a, b in itertools.combinations_with_replacement(range(k + 1), 2):
        cost = loss[a] + (band[b] - band[a]) + (legit[k] - legit[b]) * friction
        best = min(best, cost)

    policy = optimise_policy(y, amount, score, allow_review=True, max_review_rate=1.0)
    # ``expected_cost`` is money, reported to the cent, so the equality is only
    # meaningful to the cent.
    assert policy.expected_cost == pytest.approx(best, abs=0.01)


def test_capacity_capped_optimum_matches_the_constrained_brute_force(toy):
    """The band-width cap must be honoured, at the best price.

    Two things are checked at once: that the cap actually binds on this book
    (otherwise the test is vacuous), and that the cost the optimiser reports is
    the true minimum over every pair of cuts whose band holds no more rows than
    the queue can work. The cap is on rows, not on tiers, which is the whole
    reason this needs a deque rather than a fixed-width window.
    """
    y, amount, score = toy
    data = _score_groups(y, amount, score)
    k = data["n_groups"]
    loss = data["prefix_loss"]
    band = data["prefix_band"]
    legit = data["prefix_legit"]
    rows = data["prefix_rows"]
    friction = cfg.COST_MODEL.decline_friction_cost

    max_review_rate = 0.02
    window = max(int(round(max_review_rate * data["n"])), 1)

    best = np.inf
    for a in range(k + 1):
        for b in range(a, k + 1):
            if rows[b] - rows[a] > window:
                continue
            cost = loss[a] + (band[b] - band[a]) + (legit[k] - legit[b]) * friction
            best = min(best, cost)

    policy = optimise_policy(y, amount, score, allow_review=True, max_review_rate=max_review_rate)
    assert policy.review_capacity_binding, "fixture no longer exercises the cap"
    assert policy.review_rate <= max_review_rate + 1e-9
    assert policy.expected_cost == pytest.approx(best, abs=0.01)


def test_two_threshold_policy_never_costs_more_than_single_threshold(toy):
    y, amount, score = toy
    two = optimise_policy(y, amount, score, allow_review=True)
    one = optimise_policy(y, amount, score, allow_review=False)
    assert two.expected_cost <= one.expected_cost
    assert two.review_threshold <= two.decline_threshold
    assert one.review_threshold == one.decline_threshold


def test_optimal_policy_beats_declining_everything(toy):
    y, amount, score = toy
    policy = optimise_policy(y, amount, score)
    decline_all = float((1 - y).sum() * cfg.COST_MODEL.decline_friction_cost)
    assert policy.expected_cost < decline_all


def test_optimal_policy_beats_approving_everything_when_signal_exists(toy):
    y, amount, score = toy
    policy = optimise_policy(y, amount, score)
    assert policy.expected_cost < approve_all_cost(y, amount)["expected_cost"]


def test_a_review_that_can_never_save_anything_is_never_ordered(toy, monkeypatch):
    """The catch rate is what makes the queue worth paying for.

    At a catch rate of zero a review saves no money at all, so for a legitimate
    row it costs the $8 fee against approving for free, and for a fraudulent row
    it costs the $8 on top of the loss that approving would have booked anyway.
    Approving then dominates reviewing pointwise, and the optimiser must drop
    the review band entirely rather than merely shrink it.
    """
    y, amount, score = toy
    useless = dataclasses.replace(cfg.COST_MODEL, review_catch_rate=0.0)
    monkeypatch.setattr(cfg, "COST_MODEL", useless)
    policy = optimise_policy(y, amount, score)
    assert policy.n_reviewed == 0
    assert policy.review_threshold == policy.decline_threshold
    assert policy.review_operating_cost == pytest.approx(0.0, abs=0.01)


def test_a_higher_catch_rate_can_only_help(toy, monkeypatch):
    """Optimal cost is non-increasing in the catch rate.

    For any fixed policy the fraud-loss term scales with ``1 - catch``, so the
    best achievable cost at a higher catch rate is bounded by the cost the
    lower-rate optimum would have incurred - which is exactly what ``<=`` below
    checks. A violation would mean the optimiser is failing to exploit a
    strictly better queue.
    """
    y, amount, score = toy
    rates = (0.0, 0.3, 0.7, 1.0)
    costs = []
    for catch in rates:
        monkeypatch.setattr(cfg, "COST_MODEL", dataclasses.replace(cfg.COST_MODEL, review_catch_rate=catch))
        costs.append(optimise_policy(y, amount, score).expected_cost)
    assert costs == sorted(costs, reverse=True)


def test_decline_is_reserved_for_the_alerts_that_are_almost_certainly_fraud(toy):
    """Declining carries a fixed goodwill cost, so it must stay a last resort.

    The shipped model declines only where the expected friction of refusing an
    honest customer is smaller than the analyst fee plus the fraud still likely
    to slip through - a small, extreme slice of the book. If the decline band
    ever grew comparable to the review band, the friction term would be mispriced.
    """
    y, amount, score = toy
    policy = optimise_policy(y, amount, score)
    assert policy.n_declined > 0
    assert policy.review_threshold < policy.decline_threshold
    assert policy.decline_rate < policy.review_rate
    # Every declined row must be a fraud or a near-certain one: on this book the
    # decline band should be mostly fraudulent, unlike the review band.
    assert policy.fraud_declined / policy.n_declined > policy.fraud_reviewed / policy.n_reviewed


def test_review_band_is_capped_by_the_book_owners_capacity(toy):
    """A tighter queue must not be bought with a wider review band."""
    y, amount, score = toy
    tight = optimise_policy(y, amount, score, max_review_rate=0.005)
    loose = optimise_policy(y, amount, score, max_review_rate=1.0)
    assert tight.review_rate <= 0.005 + 1e-9
    assert tight.review_rate <= loose.review_rate + 1e-9
    # Capacity is a constraint, never a subsidy: it can only cost more.
    assert tight.expected_cost >= loose.expected_cost - 0.01


def test_a_discretised_score_never_claims_a_gain_it_cannot_implement():
    """The regression test for the failure isotonic calibration induces.

    Isotonic calibration maps an interval of raw scores onto a single
    probability, so the flagged tail becomes a handful of wide tiers and rows
    inside a tier are indistinguishable. A search that cut *inside* a tier would
    report the saving from splitting it and then be unable to express the split
    as a threshold, and the reported cost - which is recomputed by applying the
    thresholds - would come out higher than the single-threshold policy it
    claims to beat. With ten distinct scores this reproduces that setup exactly.
    """
    rng = np.random.default_rng(9)
    n = 3_000
    y = (rng.random(n) < 0.05).astype(int)
    amount = np.round(np.exp(rng.normal(3.6, 1.0, n)), 2)
    score = np.round(np.clip(0.05 + 0.5 * y + rng.normal(0, 0.2, n), 0.0, 1.0), 1)
    assert len(np.unique(score)) <= 11

    two = optimise_policy(y, amount, score, allow_review=True, max_review_rate=1.0)
    one = optimise_policy(y, amount, score, allow_review=False)

    # A three-way rule can only ever match or beat the two-way rule it contains.
    assert two.expected_cost <= one.expected_cost + 1e-9

    # No two rows sharing a score may be sent to different actions - apart from
    # the one non-threshold split the design allows: a flagged amount too small
    # to be worth an analyst's time is approved rather than reviewed. Those rows
    # are excluded here and asserted separately.
    masks = apply_policy(score, two.review_threshold, two.decline_threshold, amount)
    action = np.where(masks["declined"], 2, np.where(masks["reviewed"], 1, 0))
    reviewable = amount >= cfg.COST_MODEL.min_review_amount
    grouped = (
        pd.DataFrame({"score": score[reviewable], "action": action[reviewable]}).groupby("score")["action"].nunique()
    )
    assert grouped.max() == 1
    assert not masks["reviewed"][~reviewable].any()

    # And the cost the optimiser reports is the cost of applying it.
    realised = policy_outcome(y, amount, score, two.review_threshold, two.decline_threshold)
    assert realised["expected_cost"] == pytest.approx(two.expected_cost, abs=0.01)
    assert realised["n_reviewed"] == two.n_reviewed
    assert realised["n_declined"] == two.n_declined
    assert realised["n_approved"] == two.n_approved


def test_a_cheaper_queue_can_only_ever_help(toy):
    """The sensitivity sweep must be monotone in the analyst cost.

    A review band found at a dearer price must still be available when the price
    falls: the costs only shift, the achievable policies do not. If this ever
    fails, the sweep is comparing incomparable objectives.
    """
    y, amount, score = toy
    prices = (1.0, 4.0, 8.0, 20.0)
    table = review_band_sensitivity(y, amount, score, review_costs=prices, catch_rates=(0.7,))
    assert set(table["review_cost"]) == set(prices)

    ordered = table.sort_values("review_cost", ascending=False)
    savings = ordered["review_band_saving"].to_numpy()
    assert np.all(np.diff(savings) >= -1e-9), "cheaper reviews cannot be worth less"

    # Every row must reconcile with the two policies it summarises.
    for _, row in table.iterrows():
        assert row["review_band_saving"] == pytest.approx(
            row["single_threshold_cost"] - row["two_threshold_cost"], abs=0.01
        )
        assert row["review_band_saving"] >= -0.01, "the review band can never be a cost"
        assert row["review_band_open"] == (row["n_reviewed"] > 0)


def test_the_review_band_opens_when_reviews_get_cheap_enough(toy):
    """The economics the sweep is meant to demonstrate, on a book where it must.

    With a high catch rate and a token analyst cost, reviewing dominates both
    approving and declining over a wide band, so the sweep has to find it. This
    is the positive control for the sweep: without it, a sweep that always
    returns an empty band would look like a finding rather than a bug.
    """
    y, amount, score = toy
    table = review_band_sensitivity(y, amount, score, review_costs=(0.5,), catch_rates=(0.95,))
    assert table["review_band_open"].all()
    assert table["n_reviewed"].iloc[0] > 0
    assert table["review_band_saving"].iloc[0] > 0


def test_policy_counts_reconcile_with_the_book(toy):
    y, amount, score = toy
    policy = optimise_policy(y, amount, score)
    assert policy.n_approved + policy.n_reviewed + policy.n_declined == len(y)
    assert policy.fraud_approved + policy.fraud_reviewed + policy.fraud_declined == int(y.sum())


def test_policy_cost_equals_the_sum_of_its_components(toy):
    y, amount, score = toy
    policy = optimise_policy(y, amount, score)
    total = policy.loss_approved_fraud + policy.review_operating_cost + policy.decline_friction_cost
    assert policy.expected_cost == pytest.approx(total, rel=1e-6)


def test_policy_outcome_reproduces_the_optimiser_when_given_its_thresholds(toy):
    y, amount, score = toy
    policy = optimise_policy(y, amount, score)
    outcome = policy_outcome(y, amount, score, policy.review_threshold, policy.decline_threshold)
    assert outcome["expected_cost"] == pytest.approx(policy.expected_cost, rel=1e-6)
    assert outcome["n_approved"] == policy.n_approved
    assert outcome["n_reviewed"] == policy.n_reviewed
    assert outcome["n_declined"] == policy.n_declined


def test_policy_outcome_reports_rates_and_capture(toy):
    y, amount, score = toy
    outcome = policy_outcome(y, amount, score, 0.2, 0.6)
    assert outcome["fraud_caught"] <= outcome["fraud_total"]
    assert 0.0 <= outcome["fraud_capture_rate"] <= 1.0
    assert outcome["review_rate"] + outcome["decline_rate"] <= 1.0 + 1e-9


def test_approve_all_cost_is_the_full_fraud_amount(toy):
    y, amount, _ = toy
    baseline = approve_all_cost(y, amount)
    assert baseline["expected_cost"] == pytest.approx(float((amount * y).sum()), rel=1e-9)


def test_cost_curve_is_minimised_at_the_single_threshold_optimum(toy):
    y, amount, score = toy
    thresholds = np.quantile(score, np.linspace(0.5, 0.999, 40))
    curve = policy_table(y, amount, score, thresholds)
    best_on_grid = curve.loc[curve["total_cost"].idxmin(), "total_cost"]
    policy = optimise_policy(y, amount, score, allow_review=False)
    # The grid can only be as good as, or worse than, the exact optimum.
    assert policy.expected_cost <= best_on_grid + 1e-6
