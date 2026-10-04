import json
import os
import shutil
import tempfile

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from config.settings import get_decision, get_risk_band
from src.db import database
from src.document_ai.parser import parse_document
from src.explainability.explainer import explain
from src.models.fraud_checks import check_application
from src.assistant.advisor import answer_question
from src.assistant.llm_client import call_llm, is_configured

app = FastAPI(title="AI Loan Risk Assessment Platform", version="2.0.0")

database.init_db()


class ApplicantInput(BaseModel):
    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., gt=0)
    employment_years: float = Field(..., ge=0)
    loan_amount: float = Field(..., gt=0)
    credit_history_years: float = Field(..., ge=0)
    home_ownership: str = Field(..., description="RENT | OWN | MORTGAGE | OTHER")
    loan_intent: str = Field(
        ...,
        description="PERSONAL | EDUCATION | MEDICAL | VENTURE | HOMEIMPROVEMENT | DEBTCONSOLIDATION",
    )
    prior_default: str = Field(..., description="Y | N")


class AssistantQuestion(BaseModel):
    application_id: int
    question: str


def _applicant_to_df(applicant: ApplicantInput) -> pd.DataFrame:
    loan_percent_income = applicant.loan_amount / applicant.annual_income
    return pd.DataFrame(
        [
            {
                "person_age": applicant.age,
                "person_income": applicant.annual_income,
                "person_emp_length": applicant.employment_years,
                "loan_amnt": applicant.loan_amount,
                "loan_percent_income": loan_percent_income,
                "cb_person_cred_hist_length": applicant.credit_history_years,
                "person_home_ownership": applicant.home_ownership.upper(),
                "loan_intent": applicant.loan_intent.upper(),
                "cb_person_default_on_file": applicant.prior_default.upper(),
            }
        ]
    )


def _decision_from_probability(p: float) -> str:
    return get_decision(p)


@app.get("/")
def root():
    return {"status": "ok", "service": "AI Loan Risk Assessment Platform"}


@app.get("/health")
@app.get("/healthz")
def health():
    health_status = {
        "status": "healthy",
        "service": "AI Loan Risk Assessment Platform",
        "database": "unknown",
        "model": "unknown",
    }

    try:
        with database.get_conn() as conn:
            conn.execute("SELECT 1").fetchone()
        health_status["database"] = "connected"
    except Exception as exc:
        health_status["status"] = "unhealthy"
        health_status["database"] = f"error: {exc}"

    try:
        from src.explainability.explainer import MODEL_PATH
        if os.path.exists(MODEL_PATH):
            health_status["model"] = "loaded"
        else:
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    except Exception as exc:
        health_status["status"] = "unhealthy"
        health_status["model"] = f"error: {exc}"

    if health_status["status"] != "healthy":
        raise HTTPException(status_code=503, detail=health_status)

    return health_status


@app.post("/assess")
def assess(applicant: ApplicantInput):
    df = _applicant_to_df(applicant)
    result = explain(df)
    result["decision"] = _decision_from_probability(result["risk_probability"])
    return result


@app.post("/assess/document")
async def assess_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        llm_fn = call_llm if is_configured() else None
        result = parse_document(tmp_path, call_llm_fn=llm_fn)
    finally:
        os.unlink(tmp_path)

    return result


@app.post("/assess/full")
def assess_full(applicant: ApplicantInput, extracted_income: float | None = None):
    df = _applicant_to_df(applicant)
    result = explain(df)
    decision = _decision_from_probability(result["risk_probability"])

    fraud_input = {
        "annual_income": applicant.annual_income,
        "employment_years": applicant.employment_years,
        "loan_amount": applicant.loan_amount,
        "loan_percent_income": applicant.loan_amount / applicant.annual_income,
        "prior_default": applicant.prior_default,
        "risk_probability": result["risk_probability"],
    }
    fraud_flags = check_application(fraud_input, extracted_income=extracted_income)

    band = get_risk_band(result["risk_probability"])

    record = {
        "age": applicant.age,
        "annual_income": applicant.annual_income,
        "employment_years": applicant.employment_years,
        "loan_amount": applicant.loan_amount,
        "credit_history_years": applicant.credit_history_years,
        "home_ownership": applicant.home_ownership,
        "loan_intent": applicant.loan_intent,
        "prior_default": applicant.prior_default,
        "loan_percent_income": applicant.loan_amount / applicant.annual_income,
        "risk_probability": result["risk_probability"],
        "risk_band": band,
        "decision": decision,
        "narrative": result["narrative"],
        "fraud_flags": "; ".join(fraud_flags) if fraud_flags else "",
        "top_factors": json.dumps(result["top_factors"]),
    }
    app_id = database.save_application(record)

    return {
        "application_id": app_id,
        "decision": decision,
        "risk_band": band,
        **result,
        "fraud_flags": fraud_flags,
    }


@app.get("/applications")
def list_applications(limit: int = 200):
    return database.list_applications(limit=limit)


@app.get("/applications/{app_id}")
def get_application(app_id: int):
    record = database.get_application(app_id)
    if not record:
        raise HTTPException(404, "Application not found.")
    return record


@app.post("/assistant/ask")
def assistant_ask(payload: AssistantQuestion):
    record = database.get_application(payload.application_id)
    if not record:
        raise HTTPException(404, "Application not found.")

    try:
        stored_factors = json.loads(record.get("top_factors") or "[]")
    except (json.JSONDecodeError, TypeError):
        stored_factors = []

    assessment = {
        "risk_probability": record["risk_probability"],
        "narrative": record["narrative"],
        "top_factors": stored_factors,
    }
    answer = answer_question(payload.question, assessment)
    return {"answer": answer}
