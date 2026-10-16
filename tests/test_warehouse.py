"""Warehouse tests: the SQL layer is the contract every model depends on.

These build a small warehouse in a temporary database so they never disturb the
one the pipeline produced in ``data/processed``.
"""

from __future__ import annotations

import numpy as np
import pytest

from frauddetect import config as cfg
from frauddetect.data import generate_synthetic_extract
from frauddetect.warehouse import build_warehouse, list_tables, read_table, table_counts


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    """A book large enough for the lagged-history signal to be measurable.

    At six thousand transactions barely forty rows clear the sixty-day label
    resolution lag with a non-zero card history, and every conclusion drawn from
    them is coin-flip noise. Twenty thousand puts about four hundred rows in
    that bucket for roughly a second of extra build time, which is the point at
    which the signal stops being a coin flip.
    """
    db_path = tmp_path_factory.mktemp("warehouse") / "test.db"
    extract = generate_synthetic_extract(n_transactions=20_000, seed=17)
    build_warehouse(extract, verbose=False, db_path=db_path)
    return db_path


def test_every_script_produces_its_tables(warehouse):
    tables = set(list_tables(warehouse))
    for expected in (
        "stg_transactions",
        "stg_identity",
        "data_quality",
        "fct_transaction_features",
        "mart_features",
        "mart_features_vendor",
        "mart_split_summary",
        "feature_governance",
    ):
        assert expected in tables, expected


def test_staging_preserves_every_row(warehouse):
    raw = read_table("raw_transactions", warehouse)
    staged = read_table("stg_transactions", warehouse)
    assert len(staged) == len(raw)


def test_feature_table_preserves_every_transaction(warehouse):
    staged = read_table("stg_transactions", warehouse)
    features = read_table("fct_transaction_features", warehouse)
    assert len(features) == len(staged)
    assert features["TransactionID"].is_unique


def test_data_quality_checks_are_recorded_and_pass(warehouse):
    quality = read_table("data_quality", warehouse)
    assert len(quality) >= 15
    assert set(quality["status"]).issubset({"PASS", "WARN", "FAIL", "INFO"})
    failures = quality[quality["status"] == "FAIL"]
    assert failures.empty, failures[["check_name", "value"]].to_string()


def test_data_quality_records_the_majority_class_share(warehouse):
    quality = read_table("data_quality", warehouse).set_index("check_name")
    assert quality.loc["majority_class_share", "value"] > 0.99


def test_split_summary_covers_three_windows(warehouse):
    summary = read_table("mart_split_summary", warehouse)
    assert set(summary["split"]) == {"train", "test", "oot"}
    assert summary["transactions"].sum() == table_counts(warehouse)["mart_features"]


def test_split_boundaries_are_chronological(warehouse):
    summary = read_table("mart_split_summary", warehouse).set_index("split")
    assert summary.loc["train", "last_dt"] <= summary.loc["test", "first_dt"]
    assert summary.loc["test", "last_dt"] <= summary.loc["oot", "first_dt"]


# ---------------------------------------------------------------------------
# The causal invariants
# ---------------------------------------------------------------------------
def test_velocity_features_only_look_backwards(warehouse):
    """``card_txn_seq`` must count earlier transactions on the card, never later.

    The correspondence is checked with an independent pandas computation, so a
    change to the window frame that quietly includes the current row fails here.
    """
    staged = read_table("stg_transactions", warehouse).sort_values(["TransactionDT", "TransactionID"])
    features = read_table("fct_transaction_features", warehouse).set_index("TransactionID")

    expected = staged.groupby("card1").cumcount() + 1
    actual = features.loc[staged["TransactionID"].to_numpy(), "card_txn_seq"]
    assert np.array_equal(actual.to_numpy(), expected.to_numpy())

    expected_past = staged.groupby("card1").cumcount()
    actual_past = features.loc[staged["TransactionID"].to_numpy(), "card_txn_count_past"]
    assert np.array_equal(actual_past.to_numpy(), expected_past.to_numpy())


def test_first_transaction_on_a_card_has_no_history(warehouse):
    features = read_table("fct_transaction_features", warehouse)
    first = features[features["card_txn_seq"] == 1]
    assert first["card_hours_since_prev"].isna().all()
    assert (first["card_txn_count_past"] == 0).all()
    # The first transaction of the whole book also has an empty 24h window.
    assert (first["card_amt_sum_24h"] == 0).all()


def test_entity_counts_are_null_when_the_entity_is_unknown(warehouse):
    features = read_table("fct_transaction_features", warehouse)
    assert features.loc[features["addr1"].isna(), "addr_txn_count_past"].isna().all()
    assert features.loc[features["P_emaildomain"].isna(), "email_txn_count_past"].isna().all()
    assert features.loc[features["identity_missing"] == 1, "device_txn_count_past"].isna().all()


def test_device_fanout_is_at_least_one(warehouse):
    features = read_table("fct_transaction_features", warehouse).dropna(subset=["device_distinct_card_past"])
    assert (features["device_distinct_card_past"] >= 1).all()


def test_label_based_features_exclude_the_unresolved_period(warehouse):
    """No transaction may use a label from inside the resolution window.

    The generator resolves chargebacks after sixty days, so a transaction's own
    contribution to its card's historical fraud rate would be look-ahead bias.
    The invariant: the running fraud count is computed over a frame that strictly
    precedes the current row by at least the resolution lag.
    """
    features = read_table("fct_transaction_features", warehouse).sort_values("TransactionDT")
    lag_seconds = cfg.LABEL_RESOLUTION_LAG_DAYS * 86_400
    times = features["TransactionDT"].to_numpy()
    cards = features["card1"].to_numpy()
    labels = features["is_fraud"].to_numpy()
    counts = features["card_hist_txn_count"].to_numpy()

    history: dict[int, list[tuple[int, int]]] = {}
    checked = 0
    for index in range(len(features)):
        eligible = [lab for stamp, lab in history.setdefault(cards[index], []) if times[index] - stamp >= lag_seconds]
        expected = len(eligible)
        if abs(expected - counts[index]) > 1:
            raise AssertionError(f"row {index}: lagged count {counts[index]} but only {expected} eligible prior rows")
        checked += 1
        history[cards[index]].append((int(times[index]), int(labels[index])))
    assert checked == len(features)


def test_historical_fraud_rate_matches_its_own_components(warehouse):
    features = read_table("fct_transaction_features", warehouse)
    known = features.dropna(subset=["card_hist_fraud_rate"])
    assert len(known) > 0
    assert (known["card_hist_fraud_rate"] >= 0).all()
    assert (known["card_hist_fraud_rate"] <= 1).all()
    assert (known.loc[known["card_hist_fraud_rate"].notna(), "card_hist_txn_count"] > 0).all()


def test_observed_fraud_rate_rises_with_the_lagged_history(warehouse):
    """The feature has to be worth its place, not merely present.

    A card that has already been defrauded once is markedly more likely to be
    hit again than a card with a clean resolved history. The gap is an order of
    magnitude, so asserting a bare ``>`` would pass on noise; the ratio is
    pinned instead.
    """
    features = read_table("fct_transaction_features", warehouse)
    known = features.dropna(subset=["card_hist_fraud_rate"])
    clean = known.loc[known["card_hist_fraud_rate"] == 0, "is_fraud"]
    burnt = known.loc[known["card_hist_fraud_rate"] > 0, "is_fraud"]
    assert len(burnt) > 100
    assert clean.mean() > 0.0
    assert burnt.mean() > 3.0 * clean.mean()


# ---------------------------------------------------------------------------
# Governance and leak separation
# ---------------------------------------------------------------------------
def test_governance_view_classifies_the_sensitive_columns(warehouse):
    governance = read_table("feature_governance", warehouse).set_index("column_name")
    for column in cfg.POST_HOC_COLUMNS:
        assert governance.loc[column, "role"] == "post_hoc"
    for column in cfg.VENDOR_AGGREGATE_COLUMNS:
        assert governance.loc[column, "role"] == "vendor_aggregate"


def test_production_mart_has_no_vendor_counters(warehouse):
    mart = read_table("mart_features", warehouse)
    for column in cfg.VENDOR_AGGREGATE_COLUMNS:
        assert column not in mart.columns


def test_vendor_mart_carries_the_counters(warehouse):
    vendor = read_table("mart_features_vendor", warehouse)
    for column in cfg.VENDOR_AGGREGATE_COLUMNS:
        assert column in vendor.columns


def test_frequency_encoding_is_a_lookup_fitted_on_the_training_window(warehouse):
    """Coding on the full file would let the model see how often a card appears later.

    The code is the card's share of the *training* window, so two properties
    must hold: it is a pure function of the card (a test-period row for a card
    carries the value learned from train, not one recomputed from the full
    file), and its value really is the training-window share. Summing the column
    is not a usable check - the sum is ``sum(count^2) / N``, not one.
    """
    mart = read_table("mart_features", warehouse)
    train = mart[mart["split"] == "train"]
    train_n = len(train)
    train_counts = train["card1"].value_counts()

    # One value per card, everywhere in the file.
    assert mart.groupby("card1")["card1_freq_code"].nunique().max() == 1

    # And that value is the training-window share, not the full-file share.
    for card in train_counts.index[:10]:
        expected = train_counts[card] / train_n
        actual = mart.loc[mart["card1"] == card, "card1_freq_code"].iloc[0]
        assert actual == pytest.approx(expected, rel=1e-9)

    assert (mart["card1_freq_code"] >= 0.0).all()
    assert (mart["card1_freq_code"] <= 1.0).all()
    # Cards that never appear in train fall back to zero rather than to the
    # full-file share, which is the same leak in a different costume.
    unseen = set(mart["card1"].dropna()) - set(train_counts.index)
    if unseen:
        assert (mart.loc[mart["card1"].isin(unseen), "card1_freq_code"] == 0.0).all()


def test_mart_keeps_the_post_hoc_columns_available_for_audit(warehouse):
    """They are present in the mart and removed by the encoder, which the feature tests cover."""
    mart = read_table("mart_features", warehouse)
    for column in cfg.POST_HOC_COLUMNS:
        assert column in mart.columns
