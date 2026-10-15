"""Shared fixtures.

``frame`` is a small but *complete* engineered feature matrix: every column the
warehouse emits, with the right dtypes. Tests that only need "the shape the
encoder and the model server see" share it rather than each rebuilding a partial
version, because a hand-rolled subset is exactly how a test drifts away from the
schema it is supposed to be protecting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 2_000
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.sort(rng.integers(1_500_000_000, 1_510_000_000, n)),
            "TransactionAmt": np.round(np.exp(rng.normal(3.7, 1.1, n)), 2),
            "ProductCD": rng.choice(["W", "C", "R"], n),
            "card1": rng.integers(1_000, 1_500, n),
            "card2": rng.choice([np.nan, 200.0, 300.0], n),
            "card3": 150.0,
            "card4": rng.choice(["visa", "mastercard"], n),
            "card5": rng.choice([102.0, np.nan], n),
            "card6": rng.choice(["debit", "credit"], n),
            "addr1": rng.choice([np.nan, 120.0, 130.0], n),
            "addr2": 60.0,
            "dist1": np.where(rng.random(n) < 0.4, np.nan, 50.0),
            "P_emaildomain": rng.choice(["gmail.com", "anonymous.com", None], n),
            "R_emaildomain": None,
            "DeviceType": rng.choice(["desktop", "mobile", "unknown"], n),
            "DeviceInfo": rng.choice(["Windows", "MacOS", "unknown"], n),
            "id_31_browser": rng.choice(["chrome", "safari", "unknown"], n),
            "id_33_screen": rng.choice(["1280x720", "unknown"], n),
            "M1": rng.choice(["T", "F"], n),
            "M2": rng.choice(["T", "F"], n),
            "M3": rng.choice(["T", "F"], n),
            "M4": rng.choice(["T", "F"], n),
            "M5": rng.choice(["T", "F"], n),
            "M6": rng.choice(["T", "F"], n),
            "hour_of_day": rng.integers(0, 24, n),
            "day_of_week": rng.integers(0, 7, n),
            "TransactionAmt_log": np.log(np.exp(rng.normal(3.7, 1.1, n))),
            "dist1_log": 3.9,
            "is_night": rng.integers(0, 2, n),
            "card2_missing": rng.integers(0, 2, n),
            "card5_missing": rng.integers(0, 2, n),
            "addr1_missing": rng.integers(0, 2, n),
            "dist1_missing": rng.integers(0, 2, n),
            "identity_missing": rng.integers(0, 2, n),
            "mismatch_count": rng.integers(0, 7, n),
            "card_txn_seq": rng.integers(1, 40, n),
            "card_txn_count_past": rng.integers(0, 39, n),
            "card_hours_since_prev": rng.uniform(0, 200, n),
            "card_txn_1h": rng.integers(0, 5, n),
            "card_txn_24h": rng.integers(0, 12, n),
            "card_amt_sum_24h": np.round(rng.uniform(0, 900, n), 2),
            "card_amt_zscore_past": rng.normal(0, 1, n),
            "card_distinct_addr_past": rng.integers(1, 6, n),
            "addr_txn_count_past": rng.integers(0, 100, n),
            "email_txn_count_past": rng.integers(0, 400, n),
            "device_txn_count_past": rng.integers(0, 60, n),
            "device_distinct_card_past": rng.integers(1, 20, n),
            "device_hist_fraud_rate": rng.uniform(0, 0.4, n),
            "card_amt_vs_1h_mean": rng.normal(0, 0.5, n),
            "card_hist_fraud_rate": rng.uniform(0, 0.5, n),
            "card_hist_txn_count": rng.integers(0, 30, n),
            "addr_hist_fraud_rate": rng.uniform(0, 0.5, n),
            "card1_freq_code": rng.uniform(0, 0.01, n),
            "device_risk_code": rng.uniform(0, 0.01, n),
            "is_fraud": rng.integers(0, 2, n),
            "amount_bucket": rng.integers(0, 3, n),
            "chargeback_amount": np.zeros(n),
            "chargeback_days": np.zeros(n),
            "case_closed": np.zeros(n),
            "C1": rng.integers(1, 50, n).astype(float),
            "C2": rng.integers(1, 30, n).astype(float),
            "C3": rng.integers(1, 400, n).astype(float),
            "C4": rng.integers(1, 12, n).astype(float),
            "C5": rng.integers(1, 9, n).astype(float),
            "C6": np.round(rng.normal(0, 40, n), 2),
            "split": np.tile(["train", "test", "oot"], n // 3 + 1)[:n],
        }
    )
