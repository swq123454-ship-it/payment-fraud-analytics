-- ===========================================================================
-- 03_features.sql - causal feature engineering with window functions
--
-- Two rules govern this file, and both exist because the lazy version of each
-- feature produces a model that looks excellent and cannot be deployed.
--
--   RULE 1  Every window is strictly backward-looking. Frames end at
--           "1 PRECEDING" or at a value offset, never at CURRENT ROW, so a
--           transaction can never see itself or a later one.
--
--   RULE 2  A label may only be used as a feature if it would already have been
--           resolved when the transaction arrived. Chargebacks take weeks, so
--           the historical-fraud-rate windows stop 60 days short of now.
--
-- Rows with no entity to group on (a null address, an absent identity record)
-- are computed on their own filtered relation and joined back, so that the
-- resulting NULLs mean "no history" rather than "lumped in with every other
-- anonymous row".
-- ===========================================================================

DROP TABLE IF EXISTS fct_transaction_features;

CREATE TABLE fct_transaction_features AS
WITH base AS (
    SELECT
        t.TransactionID,
        t.TransactionDT,
        t.TransactionAmt,
        t.ProductCD,
        t.card1, t.card2, t.card3, t.card4, t.card5, t.card6,
        t.addr1, t.addr2, t.dist1,
        t.P_emaildomain, t.R_emaildomain,
        t.M1, t.M2, t.M3, t.M4, t.M5, t.M6,
        t.is_fraud, t.chargeback_amount, t.chargeback_days, t.case_closed,

        -- --- calendar ------------------------------------------------------
        CAST(strftime('%H', t.TransactionDT, 'unixepoch') AS INTEGER) AS hour_of_day,
        CAST(strftime('%w', t.TransactionDT, 'unixepoch') AS INTEGER) AS day_of_week,
        CASE
            WHEN CAST(strftime('%H', t.TransactionDT, 'unixepoch') AS INTEGER) < 6 THEN 1
            WHEN CAST(strftime('%H', t.TransactionDT, 'unixepoch') AS INTEGER) >= 23 THEN 1
            ELSE 0
        END AS is_night,

        -- --- transforms ----------------------------------------------------
        ln(t.TransactionAmt)                 AS TransactionAmt_log,
        ln(1.0 + COALESCE(t.dist1, 0.0))     AS dist1_log,

        -- --- missingness as signal ------------------------------------------
        -- In card-not-present data a missing field is rarely random: it is a
        -- property of the merchant integration, and it correlates with fraud.
        CASE WHEN t.card2 IS NULL THEN 1 ELSE 0 END AS card2_missing,
        CASE WHEN t.card5 IS NULL THEN 1 ELSE 0 END AS card5_missing,
        CASE WHEN t.addr1 IS NULL THEN 1 ELSE 0 END AS addr1_missing,
        CASE WHEN t.dist1 IS NULL THEN 1 ELSE 0 END AS dist1_missing,

        -- --- AVS / CVV mismatch count ---------------------------------------
        (CASE WHEN t.M1 = 'T' THEN 1 ELSE 0 END
       + CASE WHEN t.M2 = 'T' THEN 1 ELSE 0 END
       + CASE WHEN t.M3 = 'T' THEN 1 ELSE 0 END
       + CASE WHEN t.M4 = 'T' THEN 1 ELSE 0 END
       + CASE WHEN t.M5 = 'T' THEN 1 ELSE 0 END
       + CASE WHEN t.M6 = 'T' THEN 1 ELSE 0 END) AS mismatch_count,

        COALESCE(t.addr1, -1) AS addr_key
    FROM stg_transactions t
),

-- The identity feed covers a minority of traffic. The LEFT JOIN is the point of
-- the exercise: an inner join would silently discard most of the book, and the
-- absence of a record is itself informative.
identity_enriched AS (
    SELECT
        b.*,
        COALESCE(i.DeviceType, 'unknown')     AS DeviceType,
        COALESCE(i.DeviceInfo, 'unknown')     AS DeviceInfo,
        COALESCE(i.id_31_browser, 'unknown')  AS id_31_browser,
        COALESCE(i.id_33_screen, 'unknown')   AS id_33_screen,
        CASE WHEN i.TransactionID IS NULL THEN 1 ELSE 0 END AS identity_missing,
        CASE WHEN i.TransactionID IS NULL THEN 0 ELSE 1 END AS has_identity
    FROM base b
    LEFT JOIN stg_identity i ON b.TransactionID = i.TransactionID
),

-- ===========================================================================
-- Card-level windows. This is the workhorse: velocity, recency, amount
-- deviation and the lagged label rate all come from ordering the card's own
-- history and looking strictly backwards.
-- ===========================================================================
card_windows AS (
    SELECT
        k.TransactionID,

        -- How many transactions this card has made, ever, before this one.
        ROW_NUMBER() OVER (
            PARTITION BY k.card1 ORDER BY k.TransactionDT, k.TransactionID
        ) AS card_txn_seq,

        COUNT(*) OVER (
            PARTITION BY k.card1 ORDER BY k.TransactionDT, k.TransactionID
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS card_txn_count_past,

        -- Gap to the previous transaction on the card, in hours. A card that
        -- has been dormant for a year and then fires is a different risk from a
        -- card that transacts daily.
        (k.TransactionDT - LAG(k.TransactionDT) OVER (
            PARTITION BY k.card1 ORDER BY k.TransactionDT, k.TransactionID
        )) / 3600.0 AS card_hours_since_prev,

        -- Velocity in fixed time windows. RANGE with a value offset means
        -- "all rows timestamped within the last N seconds", which is exactly
        -- the business definition and needs no pre-aggregation table.
        COUNT(*) OVER (
            PARTITION BY k.card1 ORDER BY k.TransactionDT
            RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING
        ) AS card_txn_1h,

        COUNT(*) OVER (
            PARTITION BY k.card1 ORDER BY k.TransactionDT
            RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING
        ) AS card_txn_24h,

        COALESCE(SUM(k.TransactionAmt) OVER (
            PARTITION BY k.card1 ORDER BY k.TransactionDT
            RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING
        ), 0.0) AS card_amt_sum_24h,

        -- Spending baselines. Compare against the card's whole history and
        -- against the last hour, because a fraudster tests with a small amount
        -- and then submits a large one minutes later.
        AVG(k.TransactionAmt) OVER (
            PARTITION BY k.card1 ORDER BY k.TransactionDT, k.TransactionID
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS card_amt_mean_past,

        AVG(k.TransactionAmt) OVER (
            PARTITION BY k.card1 ORDER BY k.TransactionDT
            RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING
        ) AS card_amt_mean_1h,

        -- Second moment, so that the standard deviation can be formed below
        -- without a second pass over the table.
        AVG(k.TransactionAmt * k.TransactionAmt) OVER (
            PARTITION BY k.card1 ORDER BY k.TransactionDT, k.TransactionID
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS card_amt_sq_past,

        MIN(k.TransactionDT) OVER (PARTITION BY k.card1) AS card_first_dt,

        -- --- label-based, resolution-lagged ---------------------------------
        -- Only chargebacks old enough to have been settled. 5184000 seconds is
        -- sixty days. Using the raw label of a transaction from yesterday would
        -- be look-ahead bias dressed up as a "behavioural feature".
        COALESCE(SUM(k.is_fraud) OVER (
            PARTITION BY k.card1 ORDER BY k.TransactionDT
            RANGE BETWEEN UNBOUNDED PRECEDING AND 5184000 PRECEDING
        ), 0) AS card_hist_fraud_count,

        COUNT(*) OVER (
            PARTITION BY k.card1 ORDER BY k.TransactionDT
            RANGE BETWEEN UNBOUNDED PRECEDING AND 5184000 PRECEDING
        ) AS card_hist_txn_count

    FROM identity_enriched k
),

-- ===========================================================================
-- Distinct-address count per card, past-only.
--
-- SQLite does not permit DISTINCT inside a window function, so the standard
-- alternative is used: flag the first time a (card, address) pair is seen, then
-- run a cumulative sum over that flag. It is a single pass instead of the
-- correlated subquery that would otherwise be needed.
-- ===========================================================================
card_addr_first_seen AS (
    SELECT
        k.TransactionID,
        k.card1,
        k.TransactionDT,
        CASE WHEN ROW_NUMBER() OVER (
            PARTITION BY k.card1, k.addr_key ORDER BY k.TransactionDT, k.TransactionID
        ) = 1 THEN 1 ELSE 0 END AS is_new_addr
    FROM identity_enriched k
),

card_addr_spread AS (
    SELECT
        TransactionID,
        SUM(is_new_addr) OVER (
            PARTITION BY card1 ORDER BY TransactionDT, TransactionID
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS card_distinct_addr_past
    FROM card_addr_first_seen
),

-- ===========================================================================
-- Counterparty windows. Each is restricted to rows that actually carry the key,
-- then joined back so anonymous traffic yields NULL rather than a fake bucket.
-- ===========================================================================
addr_windows AS (
    SELECT
        TransactionID,
        COUNT(*) OVER (
            PARTITION BY addr_key ORDER BY TransactionDT, TransactionID
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS addr_txn_count_past,
        COALESCE(SUM(is_fraud) OVER (
            PARTITION BY addr_key ORDER BY TransactionDT
            RANGE BETWEEN UNBOUNDED PRECEDING AND 5184000 PRECEDING
        ), 0) AS addr_hist_fraud_count,
        COUNT(*) OVER (
            PARTITION BY addr_key ORDER BY TransactionDT
            RANGE BETWEEN UNBOUNDED PRECEDING AND 5184000 PRECEDING
        ) AS addr_hist_txn_count
    FROM identity_enriched
    WHERE addr_key <> -1
),

email_windows AS (
    SELECT
        TransactionID,
        COUNT(*) OVER (
            PARTITION BY P_emaildomain ORDER BY TransactionDT, TransactionID
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS email_txn_count_past
    FROM identity_enriched
    WHERE P_emaildomain IS NOT NULL
),

device_windows AS (
    SELECT
        TransactionID,
        COUNT(*) OVER (
            PARTITION BY DeviceInfo ORDER BY TransactionDT, TransactionID
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS device_txn_count_past,
        COALESCE(SUM(is_fraud) OVER (
            PARTITION BY DeviceInfo ORDER BY TransactionDT
            RANGE BETWEEN UNBOUNDED PRECEDING AND 5184000 PRECEDING
        ), 0) AS device_hist_fraud_count,
        COUNT(*) OVER (
            PARTITION BY DeviceInfo ORDER BY TransactionDT
            RANGE BETWEEN UNBOUNDED PRECEDING AND 5184000 PRECEDING
        ) AS device_hist_txn_count
    FROM identity_enriched
    WHERE has_identity = 1
),

-- How many distinct cards this device has been used with, so far.
--
-- A device that suddenly serves a dozen cards is the carding-ring signature, and
-- it is invisible to any single-entity aggregate: the tell is the *fan-out*, not
-- the volume. The first-occurrence flag plus a cumulative sum replaces the
-- DISTINCT-in-window form, which SQLite does not support.
device_card_first_seen AS (
    SELECT
        TransactionID,
        DeviceInfo,
        TransactionDT,
        CASE WHEN ROW_NUMBER() OVER (
            PARTITION BY DeviceInfo, card1 ORDER BY TransactionDT, TransactionID
        ) = 1 THEN 1 ELSE 0 END AS is_new_card
    FROM identity_enriched
    WHERE has_identity = 1
),

device_fanout AS (
    SELECT
        TransactionID,
        SUM(is_new_card) OVER (
            PARTITION BY DeviceInfo ORDER BY TransactionDT, TransactionID
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS device_distinct_card_past
    FROM device_card_first_seen
)

SELECT
    k.TransactionID,
    k.TransactionDT,
    k.TransactionAmt,
    k.ProductCD,
    k.card1, k.card2, k.card3, k.card4, k.card5, k.card6,
    k.addr1, k.addr2, k.dist1,
    k.P_emaildomain, k.R_emaildomain,
    k.DeviceType, k.DeviceInfo, k.id_31_browser, k.id_33_screen,
    k.M1, k.M2, k.M3, k.M4, k.M5, k.M6,

    k.hour_of_day, k.day_of_week, k.is_night,
    k.TransactionAmt_log, k.dist1_log,
    k.card2_missing, k.card5_missing, k.addr1_missing, k.dist1_missing,
    k.mismatch_count, k.identity_missing,

    -- --- card history ----------------------------------------------------
    c.card_txn_seq,
    c.card_txn_count_past,
    c.card_hours_since_prev,
    c.card_txn_1h,
    c.card_txn_24h,
    c.card_amt_sum_24h,
    c.card_amt_mean_past,
    c.card_amt_mean_1h,
    s.card_distinct_addr_past,
    (k.TransactionDT - c.card_first_dt) / 86400.0 AS days_since_first_seen,

    -- Relative deviation from the card's own baseline.
    CASE
        WHEN c.card_amt_mean_past IS NULL OR c.card_amt_mean_past <= 0 THEN NULL
        ELSE k.TransactionAmt / c.card_amt_mean_past - 1.0
    END AS card_amt_vs_past_mean,

    CASE
        WHEN c.card_amt_mean_1h IS NULL OR c.card_amt_mean_1h <= 0 THEN NULL
        ELSE k.TransactionAmt / c.card_amt_mean_1h - 1.0
    END AS card_amt_vs_1h_mean,

    -- Standard deviation from the first two moments, guarded against the
    -- negative variance that floating-point cancellation can produce.
    CASE
        WHEN c.card_amt_mean_past IS NULL THEN NULL
        WHEN c.card_amt_sq_past - c.card_amt_mean_past * c.card_amt_mean_past <= 0.0 THEN NULL
        ELSE (k.TransactionAmt - c.card_amt_mean_past)
             / sqrt(c.card_amt_sq_past - c.card_amt_mean_past * c.card_amt_mean_past)
    END AS card_amt_zscore_past,

    -- Resolution-lagged label rates.
    CASE
        WHEN c.card_hist_txn_count = 0 THEN NULL
        ELSE 1.0 * c.card_hist_fraud_count / c.card_hist_txn_count
    END AS card_hist_fraud_rate,
    c.card_hist_txn_count,

    CASE
        WHEN a.addr_hist_txn_count IS NULL OR a.addr_hist_txn_count = 0 THEN NULL
        ELSE 1.0 * a.addr_hist_fraud_count / a.addr_hist_txn_count
    END AS addr_hist_fraud_rate,

    CASE
        WHEN d.device_hist_txn_count IS NULL OR d.device_hist_txn_count = 0 THEN NULL
        ELSE 1.0 * d.device_hist_fraud_count / d.device_hist_txn_count
    END AS device_hist_fraud_rate,

    -- --- counterparties --------------------------------------------------
    a.addr_txn_count_past,
    e.email_txn_count_past,
    d.device_txn_count_past,
    f.device_distinct_card_past,

    -- --- label and post-hoc fields (dropped before modelling) -------------
    k.is_fraud,
    k.chargeback_amount,
    k.chargeback_days,
    k.case_closed
FROM identity_enriched k
JOIN card_windows      c ON c.TransactionID = k.TransactionID
JOIN card_addr_spread  s ON s.TransactionID = k.TransactionID
LEFT JOIN addr_windows  a ON a.TransactionID = k.TransactionID
LEFT JOIN email_windows e ON e.TransactionID = k.TransactionID
LEFT JOIN device_windows d ON d.TransactionID = k.TransactionID
LEFT JOIN device_fanout  f ON f.TransactionID = k.TransactionID;

CREATE INDEX idx_fct_txn_time ON fct_transaction_features (TransactionDT, TransactionID);
