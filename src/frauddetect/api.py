"""Serving layer: the fitted pipeline as a scoring service.

An analyst queue is only useful if a score arrives with a reason, so the service
returns the calibrated probability, the decision the cost model implies, and the
largest SHAP contributions for that specific transaction.

The endpoint takes the **engineered feature vector**, not a raw transaction. That
is deliberate and matches how these systems are built: velocity and entity
history come from a streaming feature store, and a model server that quietly
recomputed them from a single request would produce a different number from the
one the warehouse produced. Point-in-time features are a pipeline concern, and
pretending otherwise is how training/serving skew gets introduced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config as cfg
from .evaluate import Calibrator, apply_policy
from .explain import ShapExplainer, case_attribution, case_narrative
from .features import FeaturePrep

BUNDLE_PATH: Path = cfg.OUTPUT_DIR / "model" / "scoring_bundle.joblib"


def _json_scalar(value):
    """Make a feature value safe to serialise.

    Most of the design matrix is entity history, and history is frequently
    absent - a brand-new card has no previous transaction, so the feature is
    genuinely missing rather than zero. ``NaN`` is the correct in-process
    representation of that, but it is not valid JSON, so it has to become
    ``null`` on the way out. Without this the service returns a 500 for almost
    every real transaction, which is the worst possible failure for an alert
    queue: the explanation is the reason the endpoint exists.
    """
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass
class ScoringBundle:
    """Everything needed to reproduce a score, versioned together."""

    prep: FeaturePrep
    model: object
    calibrator: Calibrator
    feature_names: list[str]
    review_threshold: float
    decline_threshold: float
    metadata: dict = field(default_factory=dict)


def save_bundle(bundle: ScoringBundle, path: Path | None = None) -> Path:
    path = Path(path or BUNDLE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: Path | None = None) -> ScoringBundle:
    path = Path(path or BUNDLE_PATH)
    if not path.exists():
        raise FileNotFoundError(f"No scoring bundle at {path}. Run scripts/run_pipeline.py first.")
    return joblib.load(path)


class ScoringService:
    """Wraps a bundle so that a dictionary in gives a decision and a reason out."""

    def __init__(self, bundle: ScoringBundle):
        self.bundle = bundle
        self.explainer = ShapExplainer.from_model(bundle.model, bundle.feature_names)

    def _frame(self, payload: dict) -> pd.DataFrame:
        # Any column the payload omits is restored as NaN by the encoder's
        # reindex, and LightGBM then treats it as *missing* rather than as zero -
        # which is the correct semantics for an entity with no history.
        return pd.DataFrame([payload])

    def score_frame(self, frame: pd.DataFrame, explain: bool = True) -> pd.DataFrame:
        matrix = self.bundle.prep.transform(frame)
        raw = self.bundle.model.predict_proba(matrix)[:, 1]
        calibrated = self.bundle.calibrator.apply(raw)

        amounts = frame["TransactionAmt"].to_numpy(dtype=float) if "TransactionAmt" in frame.columns else None
        masks = apply_policy(
            calibrated,
            self.bundle.review_threshold,
            self.bundle.decline_threshold,
            amounts,
        )
        decision = np.where(masks["declined"], "decline", np.where(masks["reviewed"], "review", "approve"))

        output = pd.DataFrame(
            {"raw_score": raw, "score": calibrated, "decision": decision},
            index=frame.index,
        )
        if explain:
            contributions = self.explainer.contributions(matrix)
            top = contributions.abs().to_numpy().argsort(axis=1)[:, ::-1][:, :3]
            drivers = []
            for row_position in range(len(matrix)):
                names = [self.bundle.feature_names[i] for i in top[row_position]]
                drivers.append(
                    [
                        {
                            "feature": name,
                            "value": _json_scalar(matrix.iloc[row_position][name]),
                            "contribution": round(
                                float(contributions.iloc[row_position][name]), 4
                            ),
                        }
                        for name in names
                    ]
                )
            output["top_drivers"] = drivers
        return output

    def score_one(self, payload: dict, explain: bool = True) -> dict:
        frame = self._frame(payload)
        result = self.score_frame(frame, explain=explain).iloc[0]
        response = {
            "score": round(float(result["score"]), 6),
            "raw_score": round(float(result["raw_score"]), 6),
            "decision": str(result["decision"]),
            "review_threshold": round(float(self.bundle.review_threshold), 6),
            "decline_threshold": round(float(self.bundle.decline_threshold), 6),
            "model_version": self.bundle.metadata.get("version", "unknown"),
        }
        if explain:
            matrix = self.bundle.prep.transform(frame)
            single = matrix.iloc[[0]]
            attribution = case_attribution(self.explainer, single, top_k=5)
            response["top_drivers"] = [
                {
                    "feature": entry["feature"],
                    "value": _json_scalar(entry["value"]),
                    "contribution": round(float(entry["shap_contribution"]), 4),
                    "direction": entry["direction"],
                }
                for _, entry in attribution.iterrows()
            ]
            response["narrative"] = case_narrative(self.explainer, single, top_k=4)
        return response


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------
class ScoreRequest(BaseModel):
    features: dict = Field(
        ...,
        description=(
            "Engineered feature vector for one transaction, as produced by "
            "sql/04_feature_mart.sql. Omit features whose history does not exist "
            "rather than sending 0 or NaN: JSON cannot represent NaN, and a "
            "missing entity history is a different statement from a zero."
        ),
    )
    explain: bool = Field(True, description="Attach the largest SHAP contributions.")


class BatchScoreRequest(BaseModel):
    rows: list[dict] = Field(..., description="List of engineered feature vectors.")
    explain: bool = Field(False, description="Explanations are disabled for batch scoring.")


def create_app(bundle_path: Path | None = None) -> FastAPI:
    app = FastAPI(
        title="Payment fraud scoring service",
        version="1.0.0",
        description=(
            "Real-time scoring for card-not-present transactions. Returns a calibrated "
            "fraud probability, the cost-optimal decision, and the drivers behind it."
        ),
    )
    state: dict = {"service": None}

    def service() -> ScoringService:
        if state["service"] is None:
            state["service"] = ScoringService(load_bundle(bundle_path))
        return state["service"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "bundle_present": Path(bundle_path or BUNDLE_PATH).exists()}

    @app.get("/model/info")
    def model_info() -> dict:
        current = service()
        return {
            "version": current.bundle.metadata.get("version", "unknown"),
            "review_threshold": current.bundle.review_threshold,
            "decline_threshold": current.bundle.decline_threshold,
            "n_features": len(current.bundle.feature_names),
            "metrics": current.bundle.metadata.get("metrics", {}),
            "cost_model": asdict(cfg.COST_MODEL),
        }

    @app.post("/score")
    def score(request: ScoreRequest) -> dict:
        try:
            return service().score_one(request.features, explain=request.explain)
        except Exception as error:  # pragma: no cover - defensive boundary
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/score/batch")
    def score_batch(request: BatchScoreRequest) -> dict:
        try:
            current = service()
            frame = pd.DataFrame(request.rows)
            scored = current.score_frame(frame, explain=False)
            return {
                "n": int(len(scored)),
                "declined": int((scored["decision"] == "decline").sum()),
                "reviewed": int((scored["decision"] == "review").sum()),
                "results": scored[["raw_score", "score", "decision"]].round(6).to_dict(orient="records"),
            }
        except Exception as error:  # pragma: no cover - defensive boundary
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app


app = None  # populated by scripts/serve.py when a bundle exists
