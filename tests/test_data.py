"""Generator and warehouse tests.

These lock the properties the rest of the project assumes: reproducibility, a
fraud process that is genuinely clustered in time, and the presence of the
post-chargeback fields that the modelling layer must refuse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frauddetect import config as cfg
from frauddetect.data import (
    assign_time_split,
    generate_synthetic_extract,
)


@pytest.fixture(scope="module")
def extract():
    return generate_synthetic_extract(n_transactions=8_000, seed=7)


def test_generation_is_deterministic():
    first = generate_synthetic_extract(n_transactions=2_000, seed=11)
    second = generate_synthetic_extract(n_transactions=2_000, seed=11)
    pd.testing.assert_frame_equal(first.transactions, second.transactions)
    pd.testing.assert_frame_equal(first.identity, second.identity)


def test_different_seeds_produce_different_data():
    first = generate_synthetic_extract(n_transactions=2_000, seed=11)
    second = generate_synthetic_extract(n_transactions=2_000, seed=12)
    assert not first.transactions["TransactionAmt"].equals(second.transactions["TransactionAmt"])


def test_fraud_rate_lands_near_the_target(extract):
    assert extract.fraud_rate == pytest.approx(cfg.DEFAULT_FRAUD_RATE, rel=0.45)


def test_expected_columns_are_present(extract):
    for column in (
        "TransactionID", "TransactionDT", "TransactionAmt", "ProductCD",
        "card1", "card4", "card6", "addr1", "P_emaildomain",
        "C1", "C5", "C6", "M1", "M6", "is_fraud",
        "case_closed", "chargeback_amount", "chargeback_days",
    ):
        assert column in extract.transactions.columns, column


def test_post_chargeback_fields_exist_so_they_can_be_dropped(extract):
    """The trap only works as a teaching device if the fields are actually there."""
    closed = extract.transactions["case_closed"] == 1
    assert closed.any()
    charged = extract.transactions["chargeback_amount"] > 0
    # Chargebacks only exist for closed cases, and only for a majority of frauds.
    assert not extract.transactions.loc[~closed, "chargeback_amount"].gt(0).any()
    assert charged.sum() > 0


def test_transactions_are_ordered_by_time(extract):
    timestamps = extract.transactions["TransactionDT"].to_numpy()
    assert np.all(np.diff(timestamps) >= 0)


def test_identifiers_are_unique_and_contiguous(extract):
    ids = extract.transactions["TransactionID"].to_numpy()
    assert np.array_equal(ids, np.arange(1, len(extract) + 1))


def test_amounts_are_positive(extract):
    assert (extract.transactions["TransactionAmt"] > 0).all()


def test_identity_table_covers_a_minority_but_is_not_empty(extract):
    coverage = len(extract.identity) / len(extract.transactions)
    assert 0.05 < coverage < 0.60


def test_identity_joins_cleanly(extract):
    txn_ids = set(extract.transactions["TransactionID"])
    assert set(extract.identity["TransactionID"]).issubset(txn_ids)
    assert extract.identity["TransactionID"].is_unique


def test_fraud_is_clustered_in_time_within_a_card():
    """The whole point of the generator: velocity must be informative.

    If fraud were sprinkled independently, ``card_txn_1h`` would be noise and the
    project's SQL layer would have nothing to detect. The assertion is a ratio,
    not a target number: a fraudulent transaction must be far more likely than a
    legitimate one to follow another transaction on the same card within an hour.
    """
    frame = generate_synthetic_extract(n_transactions=60_000, seed=3).transactions
    ordered = frame.sort_values("TransactionDT", kind="stable")
    times = ordered["TransactionDT"].to_numpy()
    cards = ordered["card1"].to_numpy()
    labels = ordered["is_fraud"].to_numpy()

    history: dict[int, list[int]] = {}
    recent_fraud = recent_legit = total_fraud = total_legit = 0
    for index in range(len(ordered)):
        card = cards[index]
        previous = history.setdefault(card, [])
        nearby = sum(1 for stamp in previous if times[index] - stamp <= 3_600)
        if labels[index] == 1:
            total_fraud += 1
            recent_fraud += int(nearby > 0)
        else:
            total_legit += 1
            recent_legit += int(nearby > 0)
        previous.append(int(times[index]))

    assert total_fraud > 30
    assert recent_fraud / total_fraud > 3 * (recent_legit / total_legit)


def test_drift_is_present_across_the_period():
    """Fraud pressure rises through the calendar, so time and random splits differ."""
    frame = generate_synthetic_extract(n_transactions=120_000, seed=5).transactions
    day = (frame["TransactionDT"] - cfg.SIMULATION_START_EPOCH) // 86_400
    first_half = frame.loc[day < cfg.SIMULATION_DAYS / 2, "is_fraud"].mean()
    second_half = frame.loc[day >= cfg.SIMULATION_DAYS / 2, "is_fraud"].mean()
    assert second_half > first_half


def test_time_split_is_forward_in_time():
    frame = generate_synthetic_extract(n_transactions=3_000, seed=9).transactions
    split = assign_time_split(frame)
    assert set(split.unique()) == {"train", "test", "oot"}
    assert frame.loc[split == "train", "TransactionDT"].max() <= frame.loc[split == "test", "TransactionDT"].min()
    assert frame.loc[split == "test", "TransactionDT"].max() <= frame.loc[split == "oot", "TransactionDT"].min()
