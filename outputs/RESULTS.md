# Results

Every number below is produced by `scripts/run_pipeline.py` on the synthetic extract described in the README. Nothing is hand-entered.

## Data

- Transactions: **300,001**  (train 210,000 / test 45,000 / out-of-time 45,001)
- Fraud rate: **0.350%** (185 frauds in the out-of-time window)
- Identity-table coverage: **22.4%**

| split   |   transactions |   frauds |   fraud_rate |   first_dt |    last_dt |   gross_amount |   fraud_amount |
|:--------|---------------:|---------:|-------------:|-----------:|-----------:|---------------:|---------------:|
| oot     |          45001 |      185 |     0.004111 | 1528132657 | 1530585596 |    4.06032e+06 |        20386.7 |
| test    |          45000 |      154 |     0.003422 | 1525775323 | 1528132640 |    3.99535e+06 |        16934.4 |
| train   |         210000 |      711 |     0.003386 | 1514766151 | 1525775306 |    1.79617e+07 |        72259   |

## Headline performance (out-of-time window)

- **PR-AUC 0.5782** (95% bootstrap interval 0.5002 - 0.6638) against a random baseline of 0.4111%
- ROC-AUC 0.9511, KS 0.8887
Capacity-constrained performance on the same window:

| capacity   | recall   | precision   | lift   |
|:-----------|:---------|:------------|:-------|
| 0.100%     | 15.1%    | 62.2%       | 151.3x |
| 0.200%     | 34.1%    | 70.0%       | 170.3x |
| 0.500%     | 76.8%    | 63.1%       | 153.5x |
| 1.000%     | 88.6%    | 36.4%       | 88.7x  |
| 2.000%     | 90.3%    | 18.6%       | 45.1x  |
| 5.000%     | 90.3%    | 7.4%        | 18.1x  |

- Recall is the share of fraud inside the top slice; precision is how much of that slice is fraud; lift is precision divided by the base rate.
- Accuracy is not reported: an all-negative classifier is 99.59% accurate and catches nothing.

## Model comparison

| model               |   test_pr_auc |   test_roc_auc |   oot_pr_auc |   oot_roc_auc |   oot_ks |   oot_pr_auc_ci_low |   oot_pr_auc_ci_high |
|:--------------------|--------------:|---------------:|-------------:|--------------:|---------:|--------------------:|---------------------:|
| lightgbm            |        0.4614 |         0.9596 |       0.6044 |        0.95   |   0.8927 |              0.5166 |                0.691 |
| logistic_regression |        0.4427 |         0.9625 |       0.5879 |        0.95   |   0.8813 |            nan      |              nan     |
| isolation_forest    |        0.1458 |         0.9256 |       0.2221 |        0.9567 |   0.8694 |            nan      |              nan     |
| autoencoder         |        0.1047 |         0.928  |       0.1838 |        0.9579 |   0.8484 |            nan      |              nan     |

## Calibration

- Expected calibration error, raw: **0.00232**; after isotonic calibration: **0.00090**
- The cost-optimal threshold is a function of the predicted probability, so an uncalibrated model produces a threshold that is wrong by the same factor.

## Where the risk sits

Fraud rate by the card's *lagged* resolved fraud history. The feature is
deliberately excluded from its own bucket boundary - the lag means the
current transaction cannot contribute to the number it is bucketed by.

| bucket              |   fraud_rate |     n |
|:--------------------|-------------:|------:|
| no resolved history |  0.000304013 | 39472 |
| 0%                  |  0.00122812  |  3257 |
| 0-1%                |  0.00234192  |   427 |
| 1-5%                |  0.0926641   |  1813 |
| 5-15%               |  0           |    12 |

## Imbalance handling

| strategy                   |   target_minority_share |   mean_pr_auc |   sd_pr_auc |   mean_roc_auc |   mean_train_rows |   share_of_data_used |   mean_score |
|:---------------------------|------------------------:|--------------:|------------:|---------------:|------------------:|---------------------:|-------------:|
| smote_enn_5pct             |                    0.05 |        0.5145 |      0.0602 |         0.9386 |             85523 |               0.7127 |     0.005534 |
| random_undersample_10_to_1 |                    0.1  |        0.505  |      0.0506 |         0.9476 |              3443 |               0.0287 |     0.012014 |
| smote_5pct                 |                    0.05 |        0.4899 |      0.03   |         0.9435 |             86295 |               0.7191 |     0.002562 |
| unweighted_baseline        |                  nan    |        0.443  |      0.0305 |         0.9364 |             82500 |               0.6875 |     0.002233 |
| class_weight               |                  nan    |        0.1123 |      0.0601 |         0.5647 |             82500 |               0.6875 |     0.013456 |

**Where the optimism comes from** (same data, same model, four protocols):

| variant                               | resampling_placement         | fold_protocol                                      |   mean_pr_auc |
|:--------------------------------------|:-----------------------------|:---------------------------------------------------|--------------:|
| A_resample_before_split_random_folds  | whole window, before folding | random stratified 5-fold                           |        0.9948 |
| B_resample_inside_fold_random_folds   | inside each fold             | random stratified 5-fold                           |        0.4884 |
| C_resample_inside_fold_time_folds     | inside each fold             | forward-chaining, 4 folds                          |        0.4899 |
| D_no_resampling_time_folds            | none                         | forward-chaining, 4 folds                          |        0.443  |
| A_minus_B_resampling_placement_effect | -                            | where the resampling happened, folds held fixed    |        0.5065 |
| B_minus_C_random_vs_time_fold_effect  | -                            | the fold protocol, resampling placement held fixed |       -0.0015 |
| A_minus_C_total_optimism              | -                            | what a typical tutorial would report               |        0.505  |
| C_minus_D_value_of_resampling         | -                            | what the resampling machinery is worth             |        0.0469 |

A protocol that resamples before splitting reports PR-AUC **+0.5050** higher than the honest one. Almost all of that - +0.5065 of it - is the *placement* of the resampling, not the sampler: synthesising points across the split boundary lets a validation row's nearest neighbours include the row itself. Holding the placement fixed and swapping random folds for forward-chaining folds moves the number by -0.0015, and dropping resampling altogether - the honest baseline - costs +0.0469. So the sampler is worth about 0.0469 here, and the protocol is worth roughly 0.5065. Reporting the first number as if it were the second is the single most common way a fraud benchmark is overstated.

## Vendor counter leakage study

| feature_set          |   n_features |   test_pr_auc |   oot_pr_auc |   oot_roc_auc |   pr_auc_gap |
|:---------------------|-------------:|--------------:|-------------:|--------------:|-------------:|
| causal features only |           46 |        0.4614 |       0.6044 |        0.95   |       0      |
| plus vendor counters |           52 |        0.4945 |       0.6181 |        0.9507 |       0.0137 |

## Cost-sensitive decisioning

Cost model: {"fraud_loss_ratio": 1.0, "review_cost": 8.0, "decline_friction_cost": 25.0, "min_review_amount": 5.0, "review_catch_rate": 0.7, "max_review_rate": 0.02}

Fitted on the validation window (thresholds are chosen here and never re-tuned):

- **Two-threshold policy**: review at 0.1218, decline at 0.1218 - expected cost **5,741**. On this window it puts 0.000% of transactions through the analyst queue and refuses 0.622% outright.
- Single-threshold policy: approve or decline at 0.1218 (expected cost 5,741)
- The review band is what the third action buys: **0** cheaper on the window it was fitted on. It is deliberately described that way, because the thresholds are then frozen - an advantage measured on the window that chose them is not yet evidence.

Applied unchanged to the out-of-time window:

| policy                                    |   expected_cost |   cost_per_10k |   cost_as_pct_of_processed_amount |   review_rate |   decline_rate |   fraud_capture_rate |   loss_approved_fraud |   review_operating_cost |   decline_friction_cost |   n_approved |   n_reviewed |   n_declined |   fraud_caught |   fraud_total |   false_positive_rate |   value_saved_vs_approve_all |
|:------------------------------------------|----------------:|---------------:|----------------------------------:|--------------:|---------------:|---------------------:|----------------------:|------------------------:|------------------------:|-------------:|-------------:|-------------:|---------------:|--------------:|----------------------:|-----------------------------:|
| approve everything (baseline)             |        20386.7  |        4530.27 |                            0.5021 |             0 |       0        |               0      |                nan    |                     nan |                     nan |          nan |          nan |          nan |            nan |           nan |            nan        |                       nan    |
| fixed 0.5 threshold                       |        14994    |        3331.93 |                            0.3693 |             0 |       0.0018   |               0.3135 |              14419    |                       0 |                     575 |        44920 |            0 |           81 |             58 |           185 |              0.000513 |                      5967.65 |
| single threshold fitted on validation     |         5561.47 |        1235.85 |                            0.137  |             0 |       0.006222 |               0.8432 |               2461.47 |                       0 |                    3100 |        44721 |            0 |          280 |            156 |           185 |              0.002767 |                     17925.2  |
| two-threshold policy fitted on validation |         5561.47 |        1235.85 |                            0.137  |             0 |       0.006222 |               0.8432 |               2461.47 |                       0 |                    3100 |        44721 |            0 |          280 |            156 |           185 |              0.002767 |                     17925.2  |

Reading the table: every row is the same model on the same 45,001 unseen transactions, and only the decision rule differs. `approve everything` is the do-nothing benchmark; `fixed 0.5` is the default anyone reaches for, and it leaves most of the available saving on the table because a raw score of 0.5 is not a calibrated probability and has no reason to be the right cut.

### Is the manual queue worth funding?

At the configured 70% catch rate the review band pays for itself only while an alert costs **4 or less**, where it is worth **3** on the validation window over approve-or-decline. The shipped price is 8, which is on the other side of that line. At 8 per alert the band closes again, so the break-even sits between 4 and 8.

Across the whole grid the most the band is ever worth is **2,078**, at 1 per alert and a 90% catch rate. Note which lever matters more: halving the analyst cost buys far less than making the review actually stop the transaction, because a review that only catches part of what it finds has already lost most of the money by the time anyone looks.

|   review_cost |   review_catch_rate | review_band_open   |   n_reviewed |   n_declined |   review_rate |   two_threshold_cost |   single_threshold_cost |   review_band_saving |
|--------------:|--------------------:|:-------------------|-------------:|-------------:|--------------:|---------------------:|------------------------:|---------------------:|
|             1 |                 0.3 | False              |            0 |          280 |      0        |              5740.88 |                 5740.88 |                 0    |
|             1 |                 0.5 | True               |           37 |          280 |      0.000822 |              5719.12 |                 5740.88 |                21.76 |
|             1 |                 0.7 | True               |          208 |          182 |      0.004622 |              5432.68 |                 5740.88 |               308.2  |
|             1 |                 0.9 | True               |          339 |            0 |      0.007533 |              3663.37 |                 5740.88 |              2077.51 |
|             2 |                 0.3 | False              |            0 |          280 |      0        |              5740.88 |                 5740.88 |                 0    |
|             2 |                 0.5 | False              |            0 |          280 |      0        |              5740.88 |                 5740.88 |                 0    |
|             2 |                 0.7 | True               |          119 |          182 |      0.002644 |              5565.28 |                 5740.88 |               175.6  |
|             2 |                 0.9 | True               |          250 |            0 |      0.005556 |              3956.29 |                 5740.88 |              1784.59 |
|             4 |                 0.3 | False              |            0 |          280 |      0        |              5740.88 |                 5740.88 |                 0    |
|             4 |                 0.5 | False              |            0 |          280 |      0        |              5740.88 |                 5740.88 |                 0    |
|             4 |                 0.7 | True               |           82 |          182 |      0.001822 |              5737.55 |                 5740.88 |                 3.33 |
|             4 |                 0.9 | True               |          213 |            0 |      0.004733 |              4414.05 |                 5740.88 |              1326.83 |
|             8 |                 0.3 | False              |            0 |          280 |      0        |              5740.88 |                 5740.88 |                 0    |
|             8 |                 0.5 | False              |            0 |          280 |      0        |              5740.88 |                 5740.88 |                 0    |
|             8 |                 0.7 | False              |            0 |          280 |      0        |              5740.88 |                 5740.88 |                 0    |
|             8 |                 0.9 | True               |           82 |          182 |      0.001822 |              5204.16 |                 5740.88 |               536.72 |
|            16 |                 0.3 | False              |            0 |          280 |      0        |              5740.88 |                 5740.88 |                 0    |
|            16 |                 0.5 | False              |            0 |          280 |      0        |              5740.88 |                 5740.88 |                 0    |
|            16 |                 0.7 | False              |            0 |          280 |      0        |              5740.88 |                 5740.88 |                 0    |
|            16 |                 0.9 | False              |            0 |          280 |      0        |              5740.88 |                 5740.88 |                 0    |
|            25 |                 0.3 | False              |            0 |          280 |      0        |              5740.88 |                 5740.88 |                 0    |
|            25 |                 0.5 | False              |            0 |          280 |      0        |              5740.88 |                 5740.88 |                 0    |
|            25 |                 0.7 | False              |            0 |          280 |      0        |              5740.88 |                 5740.88 |                 0    |
|            25 |                 0.9 | False              |            0 |          280 |      0        |              5740.88 |                 5740.88 |                 0    |
|            50 |                 0.3 | False              |            0 |          280 |      0        |              5740.88 |                 5740.88 |                 0    |
|            50 |                 0.5 | False              |            0 |          280 |      0        |              5740.88 |                 5740.88 |                 0    |
|            50 |                 0.7 | False              |            0 |          280 |      0        |              5740.88 |                 5740.88 |                 0    |
|            50 |                 0.9 | False              |            0 |          280 |      0        |              5740.88 |                 5740.88 |                 0    |

## Drift monitoring

| Metric | Value |
|---|---|
| psi | 10.5737 |
| verdict | retrain |
| ks_statistic | 0.9315 |
| ks_pvalue | 0.0 |
| reference_mean_score | 0.001948 |
| actual_mean_score | 0.003311 |
| reference_flag_rate | 0.001622 |
| actual_flag_rate | 0.0018 |

Retraining recommended by the monitor: **True**

## Top drivers (mean absolute SHAP)

| feature               |   mean_abs_shap |   mean_shap |   max_abs_shap |   share_of_total |
|:----------------------|----------------:|------------:|---------------:|-----------------:|
| days_since_first_seen |        0.437714 | -0.419      |       1.37772  |         0.103589 |
| addr_txn_count_past   |        0.391115 | -0.0325419  |       2.38454  |         0.092561 |
| email_txn_count_past  |        0.220042 | -0.100663   |       1.77978  |         0.052075 |
| card_hours_since_prev |        0.21924  |  0.0226454  |       4.15047  |         0.051885 |
| mismatch_count        |        0.206322 |  0.00208372 |       3.03398  |         0.048828 |
| card1_freq_code       |        0.205493 |  0.0275285  |       3.61372  |         0.048632 |
| addr_hist_fraud_rate  |        0.199719 | -0.0812772  |       1.05099  |         0.047265 |
| card_amt_zscore_past  |        0.196646 |  0.0416419  |       1.14236  |         0.046538 |
| dist1_log             |        0.186143 | -0.00347641 |       2.10491  |         0.044052 |
| hour_of_day           |        0.165792 |  0.00925119 |       1.6933   |         0.039236 |
| TransactionAmt        |        0.165652 |  0.00514721 |       1.9373   |         0.039203 |
| TransactionAmt_log    |        0.128067 |  0.00394982 |       0.699438 |         0.030308 |

## Worked examples

**highest_scoring_fraud** - score 0.666667, amount 138.78, actual fraud 1

| feature               | value     |   shap_contribution | direction      |
|:----------------------|:----------|--------------------:|:---------------|
| card1_freq_code       | 0.0374238 |            3.43142  | increases risk |
| mismatch_count        | 6         |            2.83383  | increases risk |
| email_txn_count_past  | 1223      |            1.13604  | increases risk |
| addr_txn_count_past   | missing   |            1.02431  | increases risk |
| card_txn_24h          | 70        |            0.90204  | increases risk |
| card_hours_since_prev | 0.0541667 |            0.871034 | increases risk |

**highest_scoring_false_positive** - score 0.666667, amount 218.17, actual fraud 0

| feature              |        value |   shap_contribution | direction      |
|:---------------------|-------------:|--------------------:|:---------------|
| card1_freq_code      |    0.0374238 |            3.52613  | increases risk |
| mismatch_count       |    5         |            2.89426  | increases risk |
| email_txn_count_past | 2663         |            1.08283  | increases risk |
| dist1_log            |    5.78074   |            0.824993 | increases risk |
| is_night             |    1         |            0.700391 | increases risk |
| card_hist_fraud_rate |    0.0732684 |            0.670386 | increases risk |

## Data quality checks

| check_name                   | dimension     | metric   |         value | status   | notes                                                                               |
|:-----------------------------|:--------------|:---------|--------------:|:---------|:------------------------------------------------------------------------------------|
| row_count_raw                | completeness  | rows     | 300001        | PASS     | raw_transactions as delivered                                                       |
| row_count_staging            | completeness  | rows     | 300001        | PASS     | after de-duplication and amount filters                                             |
| rows_dropped_share           | completeness  | share    |      0        | PASS     | retention above 99% is expected                                                     |
| duplicate_transaction_ids    | uniqueness    | rows     |      0        | PASS     | primary key must be unique after staging                                            |
| overall_fraud_rate           | label_balance | share    |      0.0035   | PASS     | card-not-present fraud sits between 0.05% and 5%                                    |
| majority_class_share         | label_balance | share    |      0.9965   | INFO     | accuracy of the all-negative classifier - the number this project refuses to report |
| fraud_rate_month_2018-01     | label_balance | share    |      0.002798 | PASS     | monthly label rate - drift here invalidates a static threshold                      |
| fraud_rate_month_2018-02     | label_balance | share    |      0.003328 | PASS     | monthly label rate - drift here invalidates a static threshold                      |
| fraud_rate_month_2018-03     | label_balance | share    |      0.003625 | PASS     | monthly label rate - drift here invalidates a static threshold                      |
| fraud_rate_month_2018-04     | label_balance | share    |      0.003977 | PASS     | monthly label rate - drift here invalidates a static threshold                      |
| fraud_rate_month_2018-05     | label_balance | share    |      0.002788 | PASS     | monthly label rate - drift here invalidates a static threshold                      |
| fraud_rate_month_2018-06     | label_balance | share    |      0.004314 | PASS     | monthly label rate - drift here invalidates a static threshold                      |
| fraud_rate_month_2018-07     | label_balance | share    |      0.009642 | PASS     | monthly label rate - drift here invalidates a static threshold                      |
| missing_share_card2          | completeness  | share    |      0.093126 | INFO     | missingness is itself predictive and is kept as a feature                           |
| missing_share_addr1          | completeness  | share    |      0.091763 | INFO     | billing address                                                                     |
| missing_share_dist1          | completeness  | share    |      0.347856 | INFO     | distance between cardholder and merchant                                            |
| identity_coverage_fraud      | completeness  | share    |      0.390476 | INFO     | identity table coverage among fraudulent transactions                               |
| identity_coverage_legitimate | completeness  | share    |      0.223013 | INFO     | identity table coverage among legitimate transactions                               |
| orphan_identity_rows         | integrity     | rows     |      0        | PASS     | identity rows that cannot be joined to a transaction                                |
| non_positive_amounts         | validity      | rows     |      0        | PASS     | zero and negative amounts must not reach the model                                  |
| amount_p99                   | distribution  | currency |    675.84     | INFO     | 99th percentile amount - the tail that drives loss-weighted cost                    |
| duplicate_timestamp_share    | uniqueness    | share    |      0.013037 | INFO     | same-second transactions: bounded velocity resolution                               |
