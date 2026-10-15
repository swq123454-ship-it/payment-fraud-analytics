-- ===========================================================================
-- 01_staging.sql - type hygiene, key normalisation, de-duplication
--
-- The raw extract is loaded verbatim from the generator (or from the Kaggle
-- CSV files). Nothing downstream should ever query `raw_*` directly, because
-- the raw tables carry vendor quirks that must be resolved once, in one place.
-- ===========================================================================

DROP TABLE IF EXISTS stg_transactions;

CREATE TABLE stg_transactions AS
WITH deduplicated AS (
    SELECT
        *,
        -- The upstream feed is at-least-once, so a transaction can appear twice
        -- after a retry. Keep the lowest TransactionID and drop the rest.
        ROW_NUMBER() OVER (PARTITION BY TransactionID ORDER BY TransactionID) AS duplicate_rank
    FROM raw_transactions
)
SELECT
    CAST(TransactionID AS INTEGER)                     AS TransactionID,
    CAST(TransactionDT AS INTEGER)                     AS TransactionDT,
    CAST(TransactionAmt AS REAL)                       AS TransactionAmt,
    UPPER(TRIM(ProductCD))                             AS ProductCD,
    CAST(card1 AS INTEGER)                             AS card1,
    CAST(card2 AS REAL)                                AS card2,
    CAST(card3 AS REAL)                                AS card3,
    LOWER(TRIM(card4))                                 AS card4,
    CAST(card5 AS REAL)                                AS card5,
    LOWER(TRIM(card6))                                 AS card6,
    CAST(addr1 AS REAL)                                AS addr1,
    CAST(addr2 AS REAL)                                AS addr2,
    CAST(dist1 AS REAL)                                AS dist1,
    -- Email domains arrive with mixed case and trailing whitespace.
    NULLIF(LOWER(TRIM(P_emaildomain)), '')             AS P_emaildomain,
    NULLIF(LOWER(TRIM(R_emaildomain)), '')             AS R_emaildomain,

    -- C1..C6 are vendor-supplied all-time counters. They are kept in the
    -- warehouse for the leakage study in 04_feature_mart.sql and are NOT part
    -- of the production feature set.
    CAST(C1 AS REAL) AS C1,
    CAST(C2 AS REAL) AS C2,
    CAST(C3 AS REAL) AS C3,
    CAST(C4 AS REAL) AS C4,
    CAST(C5 AS REAL) AS C5,
    CAST(C6 AS REAL) AS C6,

    -- Match flags. The vendor encodes them as T/F/'M0'/'M1' strings.
    CASE WHEN UPPER(TRIM(M1)) = 'T' THEN 'T' ELSE 'F' END AS M1,
    CASE WHEN UPPER(TRIM(M2)) = 'T' THEN 'T' ELSE 'F' END AS M2,
    CASE WHEN UPPER(TRIM(M3)) = 'T' THEN 'T' ELSE 'F' END AS M3,
    CASE WHEN UPPER(TRIM(M4)) = 'T' THEN 'T' ELSE 'F' END AS M4,
    CASE WHEN UPPER(TRIM(M5)) = 'T' THEN 'T' ELSE 'F' END AS M5,
    CASE WHEN UPPER(TRIM(M6)) = 'T' THEN 'T' ELSE 'F' END AS M6,

    CAST(is_fraud AS INTEGER) AS is_fraud,

    -- Post-chargeback fields are stored because a real warehouse has them, and
    -- they are removed before modelling. They are the single easiest way to
    -- produce a fraudulent model that scores 1.00 AUC and is worthless.
    CAST(chargeback_amount AS REAL) AS chargeback_amount,
    CAST(chargeback_days AS INTEGER) AS chargeback_days,
    CAST(case_closed AS INTEGER)     AS case_closed
FROM deduplicated
WHERE duplicate_rank = 1
  AND TransactionAmt > 0
  AND TransactionDT IS NOT NULL;

-- Negative and zero amounts, and rows with no timestamp, are dropped rather
-- than imputed: they represent a feed defect, not a legitimate observation.

CREATE INDEX idx_stg_txn_time ON stg_transactions (TransactionDT, TransactionID);
CREATE INDEX idx_stg_txn_card ON stg_transactions (card1, TransactionDT);

DROP TABLE IF EXISTS stg_identity;

CREATE TABLE stg_identity AS
WITH deduplicated AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY TransactionID ORDER BY TransactionID) AS duplicate_rank
    FROM raw_identity
)
SELECT
    CAST(TransactionID AS INTEGER) AS TransactionID,
    COALESCE(NULLIF(TRIM(DeviceType), ''), 'unknown')   AS DeviceType,
    COALESCE(NULLIF(TRIM(DeviceInfo), ''), 'unknown')   AS DeviceInfo,
    LOWER(COALESCE(NULLIF(TRIM(id_31_browser), ''), 'unknown')) AS id_31_browser,
    COALESCE(NULLIF(TRIM(id_33_screen), ''), 'unknown') AS id_33_screen
FROM deduplicated
WHERE duplicate_rank = 1;

CREATE UNIQUE INDEX idx_stg_identity_txn ON stg_identity (TransactionID);
