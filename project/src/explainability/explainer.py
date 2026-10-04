import os

import joblib
import numpy as np
import pandas as pd

from config.settings import get_risk_band

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "risk_model.joblib")

_pipeline = None

FRIENDLY_NAMES = {
    "num__person_age": "applicant age",
    "num__person_income": "annual income",
    "num__person_emp_length": "years of employment",
    "num__loan_amnt": "requested loan amount",
    "num__loan_percent_income": "loan as % of income",
    "num__cb_person_cred_hist_length": "length of credit history",
    "cat__person_home_ownership": "home ownership type",
    "cat__loan_intent": "loan purpose",
    "cat__cb_person_default_on_file": "prior default on record",
}


def _load():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def _transformed_feature_names(pipeline):
    pre = pipeline.named_steps["preprocess"]
    return pre.get_feature_names_out()


def explain(applicant_df: pd.DataFrame, top_k: int = 4):
    pipeline = _load()
    preprocessor = pipeline.named_steps["preprocess"]
    X_transformed = preprocessor.transform(applicant_df)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    proba = pipeline.predict_proba(applicant_df)[0, 1]
    feature_names = _transformed_feature_names(pipeline)

    model = pipeline.named_steps["model"]
    importances = model.feature_importances_

    base_direction = 1.0 if proba >= 0.5 else -1.0
    contributions = importances * base_direction

    ranked_idx = np.argsort(-np.abs(contributions))[:top_k]

    top_factors = []
    for idx in ranked_idx:
        raw_name = feature_names[idx]
        friendly = FRIENDLY_NAMES.get(raw_name, raw_name.split("__")[-1].replace("_", " "))
        contribution = float(contributions[idx])
        top_factors.append(
            {
                "feature": raw_name,
                "friendly_name": friendly,
                "contribution": round(contribution, 4),
                "direction": "increases risk" if contribution > 0 else "decreases risk",
            }
        )

    narrative = _build_narrative(proba, top_factors)
    return {
        "risk_probability": round(float(proba), 4),
        "top_factors": top_factors,
        "narrative": narrative,
    }


def _build_narrative(proba, top_factors):
    band = get_risk_band(proba)
    parts = ["Estimated default risk: %.1f%% (%s risk)." % (proba * 100, band)]
    increasing = [f for f in top_factors if f["contribution"] > 0]
    decreasing = [f for f in top_factors if f["contribution"] < 0]

    if increasing:
        names = ", ".join(f["friendly_name"] for f in increasing)
        parts.append("Factors increasing risk: %s." % names)
    if decreasing:
        names = ", ".join(f["friendly_name"] for f in decreasing)
        parts.append("Factors decreasing risk: %s." % names)

    return " ".join(parts)
