-- ===========================================================================
-- 02_data_quality.sql - assertions the pipeline refuses to proceed without
--
-- Every check materialises its own result row, so the quality report is a
-- queryable table rather than a paragraph in a README that nobody re-runs.
-- ===========================================================================

DROP TABLE IF EXISTS data_quality;

CREATE TABLE data_quality (
    check_name  TEXT,
    dimension   TEXT,
    metric      TEXT,
    value       REAL,
    status      TEXT,
    notes       TEXT
);

-- --- row counts -------------------------------------------------------------
INSERT INTO data_quality
SELECT 'row_count_raw', 'completeness', 'rows', COUNT(*),
       CASE WHEN COUNT(*) > 0 THEN 'PASS' ELSE 'FAIL' END,
       'raw_transactions as delivered'
FROM raw_transactions;

INSERT INTO data_quality
SELECT 'row_count_staging', 'completeness', 'rows', COUNT(*),
       CASE WHEN COUNT(*) > 0 THEN 'PASS' ELSE 'FAIL' END,
       'after de-duplication and amount filters'
FROM stg_transactions;

INSERT INTO data_quality
SELECT 'rows_dropped_share', 'completeness', 'share',
       ROUND(1.0 * ((SELECT COUNT(*) FROM raw_transactions) - (SELECT COUNT(*) FROM stg_transactions))
             / (SELECT COUNT(*) FROM raw_transactions), 6),
       CASE WHEN 1.0 * ((SELECT COUNT(*) FROM raw_transactions) - (SELECT COUNT(*) FROM stg_transactions))
                 / (SELECT COUNT(*) FROM raw_transactions) < 0.01 THEN 'PASS' ELSE 'WARN' END,
       'retention above 99% is expected'
WHERE (SELECT COUNT(*) FROM raw_transactions) > 0;

-- --- duplicates -------------------------------------------------------------
INSERT INTO data_quality
SELECT 'duplicate_transaction_ids', 'uniqueness', 'rows',
       (SELECT COUNT(*) FROM (SELECT TransactionID FROM stg_transactions GROUP BY TransactionID HAVING COUNT(*) > 1)),
       CASE WHEN (SELECT COUNT(*) FROM (SELECT TransactionID FROM stg_transactions GROUP BY TransactionID HAVING COUNT(*) > 1)) = 0
            THEN 'PASS' ELSE 'FAIL' END,
       'primary key must be unique after staging';

-- --- label distribution -----------------------------------------------------
INSERT INTO data_quality
SELECT 'overall_fraud_rate', 'label_balance', 'share', ROUND(AVG(is_fraud), 6),
       CASE WHEN AVG(is_fraud) BETWEEN 0.0005 AND 0.05 THEN 'PASS' ELSE 'WARN' END,
       'card-not-present fraud sits between 0.05% and 5%'
FROM stg_transactions;

INSERT INTO data_quality
SELECT 'majority_class_share', 'label_balance', 'share', ROUND(1 - AVG(is_fraud), 6),
       'INFO', 'accuracy of the all-negative classifier - the number this project refuses to report'
FROM stg_transactions;

-- Fraud rate by calendar month. Fraud is seasonal and a model validated on a
-- single quiet month is not validated.
INSERT INTO data_quality
SELECT
    'fraud_rate_month_' || strftime('%Y-%m', TransactionDT, 'unixepoch'),
    'label_balance', 'share', ROUND(AVG(is_fraud), 6),
    CASE WHEN AVG(is_fraud) BETWEEN 0.0003 AND 0.08 THEN 'PASS' ELSE 'WARN' END,
    'monthly label rate - drift here invalidates a static threshold'
FROM stg_transactions
GROUP BY strftime('%Y-%m', TransactionDT, 'unixepoch');

-- --- missingness ------------------------------------------------------------
INSERT INTO data_quality
SELECT 'missing_share_card2', 'completeness', 'share',
       ROUND(1.0 * SUM(CASE WHEN card2 IS NULL THEN 1 ELSE 0 END) / COUNT(*), 6), 'INFO',
       'missingness is itself predictive and is kept as a feature'
FROM stg_transactions;

INSERT INTO data_quality
SELECT 'missing_share_addr1', 'completeness', 'share',
       ROUND(1.0 * SUM(CASE WHEN addr1 IS NULL THEN 1 ELSE 0 END) / COUNT(*), 6), 'INFO',
       'billing address'
FROM stg_transactions;

INSERT INTO data_quality
SELECT 'missing_share_dist1', 'completeness', 'share',
       ROUND(1.0 * SUM(CASE WHEN dist1 IS NULL THEN 1 ELSE 0 END) / COUNT(*), 6), 'INFO',
       'distance between cardholder and merchant'
FROM stg_transactions;

-- Missingness that differs by label is signal, not noise. Report the gap so
-- nobody "cleans" it away.
INSERT INTO data_quality
SELECT 'identity_coverage_fraud', 'completeness', 'share',
       ROUND(1.0 * (SELECT COUNT(*) FROM stg_transactions t JOIN stg_identity i USING (TransactionID)
                    WHERE t.is_fraud = 1) / NULLIF((SELECT COUNT(*) FROM stg_transactions WHERE is_fraud = 1), 0), 6),
       'INFO', 'identity table coverage among fraudulent transactions'
;

INSERT INTO data_quality
SELECT 'identity_coverage_legitimate', 'completeness', 'share',
       ROUND(1.0 * (SELECT COUNT(*) FROM stg_transactions t JOIN stg_identity i USING (TransactionID)
                    WHERE t.is_fraud = 0) / NULLIF((SELECT COUNT(*) FROM stg_transactions WHERE is_fraud = 0), 0), 6),
       'INFO', 'identity table coverage among legitimate transactions'
;

-- --- referential integrity --------------------------------------------------
INSERT INTO data_quality
SELECT 'orphan_identity_rows', 'integrity', 'rows',
       (SELECT COUNT(*) FROM stg_identity i
        WHERE NOT EXISTS (SELECT 1 FROM stg_transactions t WHERE t.TransactionID = i.TransactionID)),
       CASE WHEN (SELECT COUNT(*) FROM stg_identity i
                  WHERE NOT EXISTS (SELECT 1 FROM stg_transactions t WHERE t.TransactionID = i.TransactionID)) = 0
            THEN 'PASS' ELSE 'WARN' END,
       'identity rows that cannot be joined to a transaction';

-- --- range checks -----------------------------------------------------------
INSERT INTO data_quality
SELECT 'non_positive_amounts', 'validity', 'rows',
       SUM(CASE WHEN TransactionAmt <= 0 THEN 1 ELSE 0 END),
       CASE WHEN SUM(CASE WHEN TransactionAmt <= 0 THEN 1 ELSE 0 END) = 0 THEN 'PASS' ELSE 'FAIL' END,
       'zero and negative amounts must not reach the model'
FROM stg_transactions;

INSERT INTO data_quality
SELECT 'amount_p99', 'distribution', 'currency',
       (SELECT TransactionAmt FROM stg_transactions ORDER BY TransactionAmt
        LIMIT 1 OFFSET (SELECT CAST(COUNT(*) * 0.99 AS INTEGER) FROM stg_transactions)),
       'INFO', '99th percentile amount - the tail that drives loss-weighted cost'
;

INSERT INTO data_quality
SELECT 'duplicate_timestamp_share', 'uniqueness', 'share',
       ROUND(1.0 * (SELECT COUNT(*) FROM (SELECT TransactionDT FROM stg_transactions
                                           GROUP BY TransactionDT HAVING COUNT(*) > 1))
             / (SELECT COUNT(*) FROM stg_transactions), 6), 'INFO',
       'same-second transactions: bounded velocity resolution'
;
