"""Scoring-service tests.

The service is the only part of the repository a fraud analyst or a downstream
system actually touches, and it was previously the only part with no tests. The
two properties that matter are that a bundle round-trips through disk intact,
and that a partial payload - which is what a real request looks like, because
half the features are entity history that a thin client will not have - is
scored and explained rather than rejected.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from frauddetect.api import ScoringBundle, ScoringService, create_app, load_bundle, save_bundle
from frauddetect.evaluate import Calibrator, optimise_policy
from frauddetect.features import FeaturePrep, split_frames
from frauddetect.models import fit_lightgbm


def _payload(row) -> dict:
    """A JSON-safe request body for one transaction.

    JSON has no ``NaN``, so a client cannot send "this card has no previous
    transaction" as a numeric missing marker - it has to *omit* the key. That is
    the documented contract, and omitting is also what a thin client naturally
    does, so the tests exercise it rather than papering over it.
    """
    body = {}
    for key, value in row.drop(labels=["split"]).items():
        if isinstance(value, float) and np.isnan(value):
            continue
        body[key] = value.item() if isinstance(value, np.generic) else value
    return body


@pytest.fixture(scope="module")
def fitted(frame):
    """A fitted bundle, built exactly the way the pipeline builds it."""
    parts = split_frames(frame)
    train, test = parts["train"], parts["test"]
    y_train = train["is_fraud"].to_numpy()
    y_test = test["is_fraud"].to_numpy()

    prep = FeaturePrep().fit(train)
    model = fit_lightgbm(prep.transform(train), y_train, seed=7)
    calibrator = Calibrator().fit(y_test, model.predict_proba(prep.transform(test))[:, 1])

    policy = optimise_policy(
        y_test,
        test["TransactionAmt"].to_numpy(dtype=float),
        calibrator.apply(model.predict_proba(prep.transform(test))[:, 1]),
    )
    bundle = ScoringBundle(
        prep=prep,
        model=model,
        calibrator=calibrator,
        feature_names=prep.feature_names,
        review_threshold=policy.review_threshold,
        decline_threshold=policy.decline_threshold,
        metadata={"version": "test-bundle"},
    )
    return bundle, parts["oot"]


def test_bundle_round_trips_through_disk(fitted, tmp_path):
    bundle, _ = fitted
    path = save_bundle(bundle, tmp_path / "bundle.joblib")
    restored = load_bundle(path)
    assert restored.feature_names == bundle.feature_names
    assert restored.review_threshold == bundle.review_threshold
    assert restored.decline_threshold == bundle.decline_threshold
    assert restored.metadata["version"] == "test-bundle"


def test_loading_a_missing_bundle_names_the_command_that_makes_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_pipeline"):
        load_bundle(tmp_path / "absent.joblib")


def test_score_one_returns_a_decision_and_the_drivers_behind_it(fitted):
    bundle, oot = fitted
    service = ScoringService(bundle)
    payload = _payload(oot.iloc[0])

    response = service.score_one(payload)

    assert 0.0 <= response["score"] <= 1.0
    assert response["decision"] in {"approve", "review", "decline"}
    assert response["model_version"] == "test-bundle"
    assert len(response["top_drivers"]) == 5
    for driver in response["top_drivers"]:
        assert driver["feature"] in bundle.feature_names
        assert driver["direction"] in {"increases risk", "decreases risk"}
    # The narrative is what an analyst reads instead of the number, so it has to
    # actually name features rather than be an empty string.
    assert "Largest drivers:" in response["narrative"]


def test_score_frame_partitions_every_row_into_exactly_one_action(fitted):
    bundle, oot = fitted
    scored = ScoringService(bundle).score_frame(oot.drop(columns=["split"]).head(200), explain=False)
    assert len(scored) == 200
    assert set(scored["decision"]).issubset({"approve", "review", "decline"})
    assert scored["score"].between(0.0, 1.0).all()


def test_a_thin_payload_is_scored_rather_than_rejected(fitted):
    """Most of the design matrix is entity history a thin client will not hold.

    Sending only the fields a checkout page knows must still produce a score and
    an explanation; the absent history has to be read as "unknown", not as zero.
    """
    bundle, oot = fitted
    thin = oot.iloc[0][["TransactionAmt", "ProductCD", "card1", "card4", "DeviceType"]].to_dict()

    response = ScoringService(bundle).score_one(thin)

    assert 0.0 <= response["score"] <= 1.0
    assert response["decision"] in {"approve", "review", "decline"}
    assert len(response["top_drivers"]) == 5


def test_absent_history_is_returned_as_null_not_a_broken_response(fitted):
    """The service must render missing history as JSON ``null``.

    Most of the design matrix is entity history, so nearly every real
    transaction has at least one missing feature, and the strongest signal in
    the model is usually a feature that is missing. ``NaN`` is the correct
    in-process representation and is invalid on the wire; letting it escape
    turns the alert endpoint into a 500, which is the worst possible failure
    because the explanation is the reason the endpoint exists.
    """
    bundle, oot = fitted
    row = oot.iloc[0]
    assert row.isna().any(), "fixture no longer contains a missing feature"

    response = ScoringService(bundle).score_one(_payload(row))

    json.dumps(response, allow_nan=False)  # raises if any NaN survived


def test_scoring_a_frame_preserves_explanations_row_for_row(fitted):
    bundle, oot = fitted
    scored = ScoringService(bundle).score_frame(oot.drop(columns=["split"]).head(3), explain=True)
    assert len(scored["top_drivers"]) == 3
    assert all(len(drivers) == 3 for drivers in scored["top_drivers"])


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client(fitted, tmp_path_factory):
    from fastapi.testclient import TestClient

    bundle, _ = fitted
    path = save_bundle(bundle, tmp_path_factory.mktemp("api") / "bundle.joblib")
    return TestClient(create_app(bundle_path=path))


def test_health_reports_the_bundle_is_present(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["bundle_present"] is True


def test_model_info_exposes_the_thresholds_and_cost_model(client, fitted):
    bundle, _ = fitted
    body = client.get("/model/info").json()
    assert body["version"] == "test-bundle"
    assert body["review_threshold"] == pytest.approx(bundle.review_threshold)
    assert body["n_features"] == len(bundle.feature_names)
    # The served cost model has to be the same object the policy was fitted
    # against, or the thresholds and the money no longer belong together.
    assert body["cost_model"]["review_catch_rate"] > 0


def test_score_endpoint_returns_a_full_explanation(client, fitted):
    _, oot = fitted
    payload = _payload(oot.iloc[0])
    response = client.post("/score", json={"features": payload})
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] in {"approve", "review", "decline"}
    assert "narrative" in body


def test_batch_endpoint_counts_match_the_rows_it_returns(client, fitted):
    _, oot = fitted
    rows = [_payload(row) for _, row in oot.head(40).iterrows()]
    body = client.post("/score/batch", json={"rows": rows}).json()
    assert body["n"] == 40
    assert body["declined"] + body["reviewed"] <= 40
    assert len(body["results"]) == 40


def test_batch_endpoint_reports_a_malformed_request_as_422(client):
    response = client.post("/score/batch", json={"rows": "not-a-list"})
    assert response.status_code == 422


def test_scores_are_reproducible_across_calls(client, fitted):
    """The same transaction must not score differently on the second request."""
    _, oot = fitted
    payload = _payload(oot.iloc[5])
    first = client.post("/score", json={"features": payload, "explain": False}).json()
    second = client.post("/score", json={"features": payload, "explain": False}).json()
    assert first["score"] == pytest.approx(second["score"], abs=1e-12)
    assert np.isfinite(first["score"])
