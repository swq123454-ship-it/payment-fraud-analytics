"""Encoder tests.

The encoder is where silent leakage is easiest to introduce, so the tests here
are mostly about what the encoder must *refuse* to do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frauddetect import config as cfg
from frauddetect.features import (
    FeaturePrep,
    assert_no_forbidden_features,
    bucket_email,
    simple_bucket,
    split_frames,
)

# ``frame`` - the complete engineered feature matrix - lives in conftest.py so
# that the encoder tests and the scoring-service tests exercise the same schema.


def test_email_bucketing():
    values = pd.Series(["gmail.com", "GMAIL.COM", "anonymous.com", None, "someshop.ie"])
    assert list(bucket_email(values)) == ["major", "major", "relay", "__missing__", "other"]


def test_simple_bucket_lowercases_and_marks_missing():
    values = pd.Series([" Visa ", "", None, "MasterCard"])
    assert list(simple_bucket(values)) == ["visa", "__missing__", "__missing__", "mastercard"]


def test_post_hoc_columns_never_reach_the_design_matrix(frame):
    prep = FeaturePrep().fit(frame)
    assert_no_forbidden_features(prep.feature_names)
    for column in cfg.POST_HOC_COLUMNS:
        assert column not in prep.feature_names


def test_vendor_aggregates_are_excluded_by_default(frame):
    prep = FeaturePrep().fit(frame)
    for column in cfg.VENDOR_AGGREGATE_COLUMNS:
        assert column not in prep.feature_names


def test_vendor_aggregates_can_be_opted_in(frame):
    prep = FeaturePrep(include_vendor_aggregates=True).fit(frame)
    for column in cfg.VENDOR_AGGREGATE_COLUMNS:
        assert column in prep.feature_names


def test_assert_no_forbidden_features_raises_on_a_post_hoc_column():
    with pytest.raises(ValueError, match="chargeback_amount"):
        assert_no_forbidden_features(["TransactionAmt", "chargeback_amount"])


def test_vocabulary_is_frozen_so_unseen_levels_collapse(frame):
    prep = FeaturePrep(max_categories=2).fit(frame)
    unseen = frame.copy()
    unseen["ProductCD"] = "BRAND_NEW_CODE"
    matrix = prep.transform(unseen)
    assert matrix["ProductCD"].isna().sum() == 0
    assert set(matrix["ProductCD"].cat.categories) == set(prep.vocabularies_["ProductCD"])
    assert (matrix["ProductCD"].astype(str) == "__rare__").all()


def test_transform_tolerates_absent_columns(frame):
    """The scoring service receives partial payloads; that must not crash."""
    prep = FeaturePrep().fit(frame)
    partial = frame[["TransactionAmt", "ProductCD"]].copy()
    matrix = prep.transform(partial)
    assert list(matrix.columns) == prep.feature_names
    assert matrix["card_txn_1h"].isna().all()


def test_transform_output_shape_is_stable_across_splits(frame):
    prep = FeaturePrep().fit(frame)
    assert list(prep.transform(frame).columns) == prep.feature_names
    assert len(prep.transform(frame.iloc[:5])) == 5


def test_categorical_columns_are_categorical_dtype(frame):
    prep = FeaturePrep().fit(frame)
    matrix = prep.transform(frame)
    for column in prep.vocabularies_:
        assert isinstance(matrix[column].dtype, pd.CategoricalDtype), column


def test_linear_matrix_is_dense_and_finite(frame):
    prep = FeaturePrep().fit(frame)
    linear = prep.transform_linear(frame)
    assert len(linear.columns) == len(prep.linear_feature_names)
    assert np.isfinite(linear.to_numpy(dtype=float)).all()


def test_linear_matrix_has_no_missing_values(frame):
    prep = FeaturePrep().fit(frame)
    with_extra_nans = frame.copy()
    with_extra_nans.loc[:100, "TransactionAmt"] = np.nan
    linear = prep.transform_linear(with_extra_nans)
    assert not np.isnan(linear.to_numpy(dtype=float)).any()


def test_split_frames_partitions_the_row_set(frame):
    frames = split_frames(frame)
    assert set(frames) == {"train", "test", "oot"}
    assert sum(len(part) for part in frames.values()) == len(frame)


def test_split_frames_rejects_a_frame_without_a_split_column(frame):
    with pytest.raises(ValueError, match="split"):
        split_frames(frame.drop(columns=["split"]))


def test_encoder_audit_describes_every_feature(frame):
    prep = FeaturePrep().fit(frame)
    audit = prep.audit()
    assert set(audit["feature"]) == set(prep.feature_names)
    assert (audit["kind"].isin(["numeric", "categorical"])).all()
