-- ===========================================================================
-- 04_feature_mart.sql - frozen modelling sets
--
-- Three tables leave this file:
--
--   mart_features          production set. Strictly causal features only.
--   mart_features_vendor   the same rows plus the vendor's C1..C6 counters.
--   mart_split_summary     row and label counts per split, for the record.
--
-- The two marts exist to answer a question an interviewer will ask: "your AUC
-- looks high - is anything looking into the future?" The vendor counters are
-- all-time aggregates that include transactions after the one being scored.
-- Keeping them in a separate mart makes the size of that optimism measurable
-- instead of hypothetical.
-- ===========================================================================

DROP TABLE IF EXISTS mart_features;

CREATE TABLE mart_features AS
WITH ordered AS (
    SELECT
        f.*,
        ROW_NUMBER() OVER (ORDER BY f.TransactionDT, f.TransactionID) AS time_rank,
        COUNT(*) OVER () AS total_rows
    FROM fct_transaction_features f
),

split_assigned AS (
    SELECT
        o.*,
        CASE
            WHEN o.time_rank <= CAST(o.total_rows * 0.70 AS INTEGER) THEN 'train'
            WHEN o.time_rank <= CAST(o.total_rows * 0.85 AS INTEGER) THEN 'test'
            ELSE 'oot'
        END AS split
    FROM ordered o
),

-- Frequency encodings are fitted on the training window only. Computing them
-- over the whole file would let the model learn how often a card appears in
-- data it is not allowed to have seen.
train_card_freq AS (
    SELECT
        card1,
        COUNT(*) * 1.0 / (SELECT COUNT(*) FROM split_assigned WHERE split = 'train') AS card1_freq_code
    FROM split_assigned
    WHERE split = 'train'
    GROUP BY card1
),

train_device_freq AS (
    SELECT
        DeviceInfo,
        COUNT(*) * 1.0 / (SELECT COUNT(*) FROM split_assigned WHERE split = 'train') AS device_risk_code
    FROM split_assigned
    WHERE split = 'train'
    GROUP BY DeviceInfo
)

SELECT
    s.*,
    COALESCE(f.card1_freq_code, 0.0)   AS card1_freq_code,
    COALESCE(dv.device_risk_code, 0.0) AS device_risk_code
FROM split_assigned s
LEFT JOIN train_card_freq   f  ON s.card1 = f.card1
LEFT JOIN train_device_freq dv ON s.DeviceInfo = dv.DeviceInfo;

DROP TABLE IF EXISTS mart_features_vendor;

CREATE TABLE mart_features_vendor AS
SELECT
    m.*,
    t.C1, t.C2, t.C3, t.C4, t.C5, t.C6
FROM mart_features m
JOIN stg_transactions t ON t.TransactionID = m.TransactionID;

-- ===========================================================================

DROP TABLE IF EXISTS mart_split_summary;

CREATE TABLE mart_split_summary AS
SELECT
    split,
    COUNT(*)                          AS transactions,
    SUM(is_fraud)                     AS frauds,
    ROUND(AVG(is_fraud), 6)           AS fraud_rate,
    MIN(TransactionDT)                AS first_dt,
    MAX(TransactionDT)                AS last_dt,
    ROUND(SUM(TransactionAmt), 2)     AS gross_amount,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN TransactionAmt ELSE 0 END), 2) AS fraud_amount
FROM mart_features
GROUP BY split;

-- ===========================================================================
-- Governance view: which columns a model is allowed to see.
-- A reviewer can check the answer without reading the Python source.
-- ===========================================================================

DROP TABLE IF EXISTS feature_governance;

CREATE TABLE feature_governance AS
SELECT 'is_fraud' AS column_name, 'label' AS role, 'never a feature' AS note
UNION ALL SELECT 'chargeback_amount', 'post_hoc', 'known only after the chargeback is settled - dropped before modelling'
UNION ALL SELECT 'chargeback_days', 'post_hoc', 'known only after the chargeback is settled - dropped before modelling'
UNION ALL SELECT 'case_closed', 'post_hoc', 'case resolution status - dropped before modelling'
UNION ALL SELECT 'C1', 'vendor_aggregate', 'all-time counter that includes later transactions - sensitivity study only'
UNION ALL SELECT 'C2', 'vendor_aggregate', 'all-time counter that includes later transactions - sensitivity study only'
UNION ALL SELECT 'C3', 'vendor_aggregate', 'all-time counter that includes later transactions - sensitivity study only'
UNION ALL SELECT 'C4', 'vendor_aggregate', 'all-time counter that includes later transactions - sensitivity study only'
UNION ALL SELECT 'C5', 'vendor_aggregate', 'forward-looking 7-day counter - sensitivity study only'
UNION ALL SELECT 'C6', 'vendor_aggregate', 'deviation from an all-time mean - sensitivity study only'
UNION ALL SELECT 'TransactionID', 'identifier', 'row key, no predictive meaning'
UNION ALL SELECT 'card1', 'identifier', 'raw identifier, replaced by card1_freq_code fitted on train'
UNION ALL SELECT 'addr1', 'identifier', 'raw identifier, replaced by addr_txn_count_past'
UNION ALL SELECT 'TransactionDT', 'ordering', 'used for the temporal split only'
UNION ALL SELECT 'card_hist_fraud_rate', 'causal_label', 'lagged 60 days, assumes chargebacks are settled by then'
UNION ALL SELECT 'addr_hist_fraud_rate', 'causal_label', 'lagged 60 days, assumes chargebacks are settled by then';
