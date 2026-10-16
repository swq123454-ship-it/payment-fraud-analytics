"""Feature preprocessing: bucketing, vocabulary freezing, design-matrix build.

The SQL layer already produced the relational features (velocity, entity counts,
lagged label rates). This module owns the remaining decisions:

* which raw categorical values survive as their own level and which collapse
  into ``__rare__``;
* how email domains and device strings are grouped into business-meaningful
  buckets;
* which columns are structurally forbidden from ever entering the model.

Every vocabulary is fitted on the training window only and then frozen. Fitting
it on the full file would leak information about the future into the encoder -
the same class of mistake as scaling before splitting, just less visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as cfg

#: Disposable and privacy-relay mail providers. In production these accounts are
#: trivially created in bulk, so they carry a different risk profile from a
#: fifteen-year-old ISP mailbox.
RELAY_DOMAINS: frozenset[str] = frozenset(
    {
        "anonymous.com",
        "mailinator.com",
        "guerrillamail.com",
        "tempmail.com",
        "protonmail.com",
        "mail.com",
        "gmx.com",
        "yandex.ru",
    }
)

#: Consumer mailbox providers worth keeping as their own level.
MAJOR_DOMAINS: frozenset[str] = frozenset(
    {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "comcast.net", "icloud.com"}
)


def bucket_email(series: pd.Series) -> pd.Series:
    """Collapse the long tail of mail domains into three business buckets."""
    domain = series.astype("string").str.lower().str.strip()
    result = pd.Series("__missing__", index=series.index, dtype=object)
    result[domain.notna()] = "other"
    result[domain.isin(MAJOR_DOMAINS)] = "major"
    result[domain.isin(RELAY_DOMAINS)] = "relay"
    return result


def simple_bucket(series: pd.Series) -> pd.Series:
    """Case-fold a raw categorical and mark missing values explicitly."""
    values = series.astype("string").str.lower().str.strip()
    return values.where(values.notna() & (values != ""), "__missing__").astype(object)


@dataclass
class FeaturePrep:
    """Frozen, reusable encoder for the fraud design matrix.

    Two output flavours are offered because the two model families need
    different treatment of missing values:

    * ``transform`` keeps NaNs and emits pandas ``category`` columns, which is
      what LightGBM's native categorical handling expects.
    * ``transform_linear`` median-imputes and standardises, which is what
      logistic regression needs.
    """

    categorical_features: tuple[str, ...] = cfg.CATEGORICAL_FEATURES
    numeric_features: tuple[str, ...] = cfg.NUMERIC_FEATURES
    max_categories: int = 25
    include_vendor_aggregates: bool = False

    vocabularies_: dict[str, list] = field(default_factory=dict, init=False)
    numeric_columns_: list[str] = field(default_factory=list, init=False)
    medians_: pd.Series | None = field(default=None, init=False)
    scaler_mean_: pd.Series | None = field(default=None, init=False)
    scaler_std_: pd.Series | None = field(default=None, init=False)
    dropped_columns_: list[str] = field(default_factory=list, init=False)

    # -- derivation --------------------------------------------------------
    def _derive(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if "P_emaildomain" in out.columns:
            out["P_emaildomain_bucket"] = bucket_email(out["P_emaildomain"])
        if "R_emaildomain" in out.columns:
            out["R_emaildomain_bucket"] = bucket_email(out["R_emaildomain"])
        for column in out.columns:
            if column.startswith("id_") or column in ("ProductCD", "card4", "card6", "DeviceType"):
                out[column] = simple_bucket(out[column])
        return out

    def _available_numeric(self, frame: pd.DataFrame) -> list[str]:
        wanted = list(self.numeric_features)
        if self.include_vendor_aggregates:
            wanted = wanted + list(cfg.VENDOR_AGGREGATE_COLUMNS)
        return [c for c in wanted if c in frame.columns]

    # -- fitting -----------------------------------------------------------
    def fit(self, frame: pd.DataFrame) -> FeaturePrep:
        derived = self._derive(frame)

        self.vocabularies_ = {}
        for column in self.categorical_features:
            if column not in derived.columns:
                continue
            counts = derived[column].value_counts(dropna=False)
            kept = list(counts.index[: self.max_categories])
            # Anything outside the top levels collapses to one bucket, so the
            # category set cannot grow between fitting and scoring.
            self.vocabularies_[column] = [str(v) for v in kept] + ["__rare__"]

        self.numeric_columns_ = self._available_numeric(derived)
        numeric_block = derived[self.numeric_columns_].apply(pd.to_numeric, errors="coerce")
        self.medians_ = numeric_block.median()
        self.scaler_mean_ = numeric_block.mean()
        # A zero standard deviation would blow up the linear transform.
        self.scaler_std_ = numeric_block.std().replace(0.0, 1.0).fillna(1.0)

        forbidden = set(cfg.POST_HOC_COLUMNS) | set(cfg.DROP_AFTER_EDA) | {cfg.TARGET_COLUMN, "split"}
        if not self.include_vendor_aggregates:
            forbidden |= set(cfg.VENDOR_AGGREGATE_COLUMNS)
        self.dropped_columns_ = sorted(c for c in frame.columns if c in forbidden)
        return self

    # -- transforming ------------------------------------------------------
    def _encode_categoricals(self, derived: pd.DataFrame) -> pd.DataFrame:
        encoded: dict[str, pd.Series] = {}
        for column, vocabulary in self.vocabularies_.items():
            if column not in derived.columns:
                # Scoring a single request will not carry every column the
                # training set had. An absent categorical is a missing value,
                # never a silent zero.
                encoded[column] = pd.Categorical(
                    np.full(len(derived), "__missing__", dtype=object), categories=vocabulary
                )
                continue
            values = derived[column].astype(object).where(derived[column].notna(), "__missing__").astype(str)
            values = values.where(values.isin(vocabulary), "__rare__")
            encoded[column] = pd.Categorical(values, categories=vocabulary)
        return pd.DataFrame(encoded, index=derived.index)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Tree-model design matrix: NaNs preserved, categoricals as ``category``."""
        if not self.vocabularies_ and not self.numeric_columns_:
            raise RuntimeError("FeaturePrep.fit must be called before transform.")
        derived = self._derive(frame)
        # reindex rather than index: an absent column becomes NaN, which the
        # booster reads as missing instead of crashing at scoring time.
        numeric = derived.reindex(columns=self.numeric_columns_).apply(pd.to_numeric, errors="coerce").astype(float)
        categorical = self._encode_categoricals(derived)
        return pd.concat([numeric, categorical], axis=1)[self.feature_names]

    def transform_linear(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Linear-model design matrix: median imputation and z-scaling."""
        matrix = self.transform(frame)
        numeric = matrix[self.numeric_columns_].fillna(self.medians_).fillna(0.0)
        numeric = (numeric - self.scaler_mean_) / self.scaler_std_
        parts = [numeric]
        for column in self.vocabularies_:
            dummies = pd.get_dummies(matrix[column].astype(str), prefix=column, dtype=float)
            # Level set is guaranteed by the frozen vocabulary, so this reindex
            # can only restore columns, never invent them.
            expected = [f"{column}_{level}" for level in self.vocabularies_[column]]
            dummies = dummies.reindex(columns=expected, fill_value=0.0)
            parts.append(dummies)
        return pd.concat(parts, axis=1)

    @property
    def feature_names(self) -> list[str]:
        return list(self.numeric_columns_) + list(self.vocabularies_.keys())

    @property
    def linear_feature_names(self) -> list[str]:
        names = list(self.numeric_columns_)
        for column, vocabulary in self.vocabularies_.items():
            names.extend(f"{column}_{level}" for level in vocabulary)
        return names

    def audit(self) -> pd.DataFrame:
        """Machine-readable record of what the encoder decided."""
        rows = []
        for column in self.numeric_columns_:
            rows.append({"feature": column, "kind": "numeric", "levels": np.nan})
        for column, levels in self.vocabularies_.items():
            rows.append({"feature": column, "kind": "categorical", "levels": len(levels)})
        return pd.DataFrame(rows)


def split_frames(mart: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the frozen mart into train / test / out-of-time."""
    missing = {"split"} - set(mart.columns)
    if missing:
        raise ValueError("mart is missing the split column; run the SQL layer first")
    return {name: mart[mart["split"] == name].copy() for name in cfg.SPLIT_ORDER}


def assert_no_forbidden_features(feature_names) -> None:
    """Hard guard against the two easiest ways to ruin this project.

    A model that consumes ``chargeback_amount`` scores ~1.00 and is useless,
    because at scoring time no chargeback exists yet. This raises rather than
    warns, so the failure mode cannot reach a reported metric.
    """
    forbidden = set(cfg.POST_HOC_COLUMNS)
    offending = sorted(forbidden & set(feature_names))
    if offending:
        raise ValueError(
            f"Post-chargeback columns reached the design matrix: {offending}. These are only knowable after the fact."
        )
