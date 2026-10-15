"""Central configuration for the payment fraud analytics pipeline.

Every tunable that a reviewer might question lives here rather than being
scattered through the modules, so that the assumptions behind the reported
numbers can be inspected in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SQL_DIR: Path = PROJECT_ROOT / "sql"
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR: Path = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
FIGURE_DIR: Path = OUTPUT_DIR / "figures"
TABLE_DIR: Path = OUTPUT_DIR / "tables"
DB_PATH: Path = DATA_PROCESSED_DIR / "fraud.db"
RESULTS_PATH: Path = OUTPUT_DIR / "RESULTS.md"
METRICS_PATH: Path = OUTPUT_DIR / "metrics.json"

#: Single fixed seed. Every stochastic step derives from it, so two runs on the
#: same input produce byte-identical metrics.
SEED: int = 42

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
DEFAULT_N_TRANSACTIONS: int = 300_000
#: Real card-not-present fraud rates sit between roughly 0.1% and 0.5% of
#: transactions. The public IEEE-CIS training file is an *enriched* sample at
#: ~3.5%, which flatters accuracy-style metrics; we simulate the operational
#: rate instead.
DEFAULT_FRAUD_RATE: float = 0.0035
#: Share of transaction volume produced by account-takeover episodes. Fraud is
#: concentrated, not uniformly sprinkled: a small share of traffic carries most
#: of the loss, which is why the effective detection rate per alert matters.
EPISODE_VOLUME_SHARE: float = 0.006
#: Share of legitimate traffic that is a shopping spree (several transactions on
#: one card in an afternoon). Without it, velocity would be a perfect detector.
LEGITIMATE_SPREE_SHARE: float = 0.030
SIMULATION_DAYS: int = 182

#: Size of the drift applied over the simulated period. Fraud is not stationary:
#: adversaries learn, tooling rotates, and seasonal shopping changes what
#: "normal" looks like. Without drift a temporal split has nothing to bite on and
#: every monitoring figure comes out flat - which would make the whole drift
#: section of this project vacuous.
FRAUD_PRESSURE_DRIFT: float = 0.90
AMOUNT_INFLATION_DRIFT: float = 0.10
SPREE_GROWTH_DRIFT: float = 1.10
#: 2018-01-01T00:00:00Z. Anchoring the simulation to a real date keeps the
#: month-by-month quality checks readable and stops `strftime` from reporting a
#: fake 1970 calendar.
SIMULATION_START_EPOCH: int = 1_514_764_800

# ---------------------------------------------------------------------------
# Target and split
# ---------------------------------------------------------------------------
TARGET_COLUMN: str = "is_fraud"
TRAIN_FRACTION: float = 0.70
TEST_FRACTION: float = 0.15
#: The remaining 15% is the out-of-time holdout.
SPLIT_ORDER: tuple[str, ...] = ("train", "test", "oot")

#: Columns that are only knowable *after* a chargeback is confirmed. They are
#: present in the raw extract because a real data warehouse stores them, and
#: they are dropped before modelling. A unit test locks this invariant.
POST_HOC_COLUMNS: tuple[str, ...] = (
    "chargeback_amount",
    "chargeback_days",
    "case_closed",
)

#: Vendor-supplied `C*` counters in the extract. These are all-time aggregates
#: that include transactions *after* the one being scored, so they are mildly
#: leaky. Kept in the warehouse for the leakage-gap study, excluded from the
#: production feature set.
VENDOR_AGGREGATE_COLUMNS: tuple[str, ...] = (
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
)

#: Strictly causal (past-only) features produced by the SQL window functions.
CAUSAL_VELOCITY_COLUMNS: tuple[str, ...] = (
    "card_txn_seq",
    "card_txn_count_past",
    "card_hours_since_prev",
    "card_txn_1h",
    "card_txn_24h",
    "card_amt_sum_24h",
    "card_amt_zscore_past",
    "card_distinct_addr_past",
    "addr_txn_count_past",
    "email_txn_count_past",
    "device_txn_count_past",
    "device_distinct_card_past",
    "device_hist_fraud_rate",
    "card_amt_vs_1h_mean",
    "card_hist_fraud_rate",
    "card_hist_txn_count",
    "addr_hist_fraud_rate",
)

#: Chargebacks take weeks to land. A label-based feature is only legitimate if
#: it is restricted to transactions old enough to have been resolved by the time
#: the transaction being scored arrived. Sixty days is the industry rule of thumb.
LABEL_RESOLUTION_LAG_DAYS: int = 60

NUMERIC_FEATURES: tuple[str, ...] = CAUSAL_VELOCITY_COLUMNS + (
    "TransactionAmt",
    "TransactionAmt_log",
    "card1_freq_code",
    "card2_missing",
    "card5_missing",
    "addr1_missing",
    "dist1_missing",
    "dist1_log",
    "identity_missing",
    "hour_of_day",
    "day_of_week",
    "days_since_first_seen",
    "is_night",
    "mismatch_count",
    "device_risk_code",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain_bucket",
    "R_emaildomain_bucket",
    "DeviceType",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "id_31_browser",
    "id_33_screen",
)

#: Columns deliberately removed after the EDA stage because they encode the
#: entity risk directly and would not exist at scoring time.
DROP_AFTER_EDA: tuple[str, ...] = (
    "card1",
    "addr1",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "TransactionID",
    "TransactionDT",
)


# ---------------------------------------------------------------------------
# Cost-sensitive decisioning
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CostModel:
    """Business costs that turn model scores into an operating decision.

    Three actions are available to the fraud team, so the policy is defined by
    two thresholds rather than one:

    * ``approve``  - score below the review threshold. A fraudulent transaction
      that is approved is a realised loss equal to its amount.
    * ``review``   - score in the middle band. A human looks at it; fraud is
      caught, at a fixed handling cost, and legitimate traffic is released.
    * ``decline``  - score at or above the decline threshold. Fraud is stopped
      but so is some legitimate traffic, which costs goodwill.

    Real fraud systems are built this way because auto-declining a customer is
    materially more expensive than asking an analyst to look at a queue.
    """

    #: Share of a fraudulent transaction's amount actually lost if approved.
    #: Below 1.0 when the acquirer recovers part of the value.
    fraud_loss_ratio: float = 1.0
    #: Fully-loaded cost of one manual review (analyst time, tooling).
    review_cost: float = 8.00
    #: Cost of wrongly declining a legitimate transaction (support call,
    #: customer churn, scheme penalties).
    decline_friction_cost: float = 25.00
    #: Minimum absolute amount worth a manual review. Below this the review
    #: costs more than the loss it could prevent, so tiny alerts are approved.
    min_review_amount: float = 5.00
    #: Share of reviewed fraud that is actually stopped before the money leaves.
    #: This is the real reason an issuer ever declines rather than reviews. A
    #: review is not a guaranteed save: the analyst queue has latency, and a
    #: transaction that has already authorised and settled cannot be recalled.
    #: With a catch rate of 0.70, reviewing a fraudulent 100-unit transaction
    #: costs 8 + 30 = 38, while declining it costs nothing - so the top of the
    #: ranking, where precision is high, is worth declining outright.
    review_catch_rate: float = 0.70
    #: Share of transaction volume the fraud team can physically work per day.
    #: Retained as a guard: no reported policy should ever be one a team cannot
    #: staff. Whether it actually binds at the fitted optimum is reported.
    max_review_rate: float = 0.02


COST_MODEL = CostModel()

#: Review capacity scenarios, as a share of daily transaction volume. Maps to
#: the "we can only look at N alerts a day" constraint that real fraud teams
#: operate under, and drives the Recall@K table.
REVIEW_CAPACITY_GRID: tuple[float, ...] = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)


# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
#: Share of the training set used to fit the resampler / calibrator, kept apart
#: from the fold used to fit the model. Without this, SMOTE-then-fit leaks
#: synthetic neighbours of validation rows into training.
CALIBRATION_FRACTION: float = 0.20

#: Number of expanding-window folds used for the resampling comparison.
N_TIME_FOLDS: int = 4

TARGET_FRAUD_RECALL: float = 0.60

#: The resampling sweep repeats the fit once per policy per fold, so it runs on a
#: contiguous tail of the training window rather than the whole thing. Time order
#: is preserved, which is what the folds depend on. Only the *ranking* of the
#: policies is read from that table - the absolute levels are not comparable to
#: the headline numbers, and the README says so.
SAMPLING_STUDY_MAX_ROWS: int = 120_000

#: Rows used for the global SHAP summary. Contributions are per-transaction and
#: do not need the full window to stabilise.
SHAP_SAMPLE_ROWS: int = 8_000

#: Permutation importance repeats, applied to a 20k-row sample.
PERMUTATION_REPEATS: int = 5

LIGHTGBM_PARAMS: dict[str, float | int | str] = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 40,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": -1,
}

#: Reduced configuration used only inside the resampling sweep, which repeats
#: the fit once per policy per fold.
RESAMPLING_STUDY_PARAMS: dict[str, float | int] = {
    "n_estimators": 250,
    "learning_rate": 0.08,
    "num_leaves": 31,
    "min_child_samples": 20,
}

#: Resampling policies compared in the imbalance study.
#:
#: The target ratios are deliberately partial. At a 0.35% base rate, sampling to
#: full balance would replace 99.3% of the training set with synthetic rows, so
#: the model would fit the generator rather than the population. Lifting the
#: minority to 5% of the majority is the standard compromise.
#:
#: ``unweighted_baseline`` and ``class_weight`` are both "no resampling" arms and
#: differ only in whether LightGBM reweights the loss. Including both matters:
#: without the unweighted arm there is no evidence that any of the machinery
#: earns its place.
RESAMPLING_STRATEGIES: dict[str, float | None] = {
    "unweighted_baseline": None,
    "class_weight": None,
    "random_undersample_10_to_1": 0.10,
    "smote_5pct": 0.05,
    "smote_enn_5pct": 0.05,
}
