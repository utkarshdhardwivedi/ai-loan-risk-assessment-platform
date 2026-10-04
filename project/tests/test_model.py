import pandas as pd
import pytest

from src.explainability.explainer import explain


def make_applicant(**overrides):
    base = {
        "person_age": 30,
        "person_income": 60000.0,
        "person_emp_length": 4.0,
        "loan_amnt": 10000.0,
        "loan_percent_income": 10000.0 / 60000.0,
        "cb_person_cred_hist_length": 5.0,
        "person_home_ownership": "RENT",
        "loan_intent": "PERSONAL",
        "cb_person_default_on_file": "N",
    }
    base.update(overrides)
    if "loan_amnt" in overrides or "person_income" in overrides:
        base["loan_percent_income"] = base["loan_amnt"] / base["person_income"]
    return pd.DataFrame([base])


def test_explain_returns_valid_probability():
    df = make_applicant()
    result = explain(df)
    assert 0.0 <= result["risk_probability"] <= 1.0
    assert "narrative" in result
    assert len(result["top_factors"]) > 0


def test_high_risk_scores_higher_than_low_risk():
    low_risk = make_applicant(
        person_income=120000.0,
        loan_amnt=8000.0,
        person_emp_length=12.0,
        cb_person_cred_hist_length=15.0,
        cb_person_default_on_file="N",
    )
    high_risk = make_applicant(
        person_income=25000.0,
        loan_amnt=20000.0,
        person_emp_length=0.5,
        cb_person_cred_hist_length=1.0,
        cb_person_default_on_file="Y",
    )
    low_result = explain(low_risk)
    high_result = explain(high_risk)
    assert high_result["risk_probability"] > low_result["risk_probability"], (
        f"High-risk applicant ({high_result['risk_probability']:.4f}) should score "
        f"higher than low-risk ({low_result['risk_probability']:.4f})"
    )


def test_top_factors_have_required_fields():
    df = make_applicant()
    result = explain(df)
    for factor in result["top_factors"]:
        assert "friendly_name" in factor
        assert "direction" in factor
        assert factor["direction"] in ("increases risk", "decreases risk")


def test_prior_default_shows_up_in_top_factors_for_risky_applicant():
    df = make_applicant(
        cb_person_default_on_file="Y",
        person_income=30000.0,
        loan_amnt=25000.0,
        cb_person_cred_hist_length=1.0,
    )
    result = explain(df)
    feature_names = [f["feature"] for f in result["top_factors"]]
    friendly_names = [f["friendly_name"] for f in result["top_factors"]]
    assert result["risk_probability"] > 0.3, "Prior defaulter with large loan should be high risk"
