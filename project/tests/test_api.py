import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.db import database


@pytest.fixture(autouse=True)
def init_test_db(monkeypatch, tmp_path):
    test_db = str(tmp_path / "test_api.db")
    monkeypatch.setattr(database, "DB_PATH", test_db)
    database.init_db()


@pytest.fixture
def client():
    return TestClient(app)


def sample_applicant_payload(**overrides):
    payload = {
        "age": 30,
        "annual_income": 65000.0,
        "employment_years": 5.0,
        "loan_amount": 10000.0,
        "credit_history_years": 6.0,
        "home_ownership": "RENT",
        "loan_intent": "PERSONAL",
        "prior_default": "N",
    }
    payload.update(overrides)
    return payload


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data


def test_health_endpoint_healthy(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert data["model"] == "loaded"


def test_health_endpoint_db_failure(client, monkeypatch):
    def broken_conn():
        raise RuntimeError("DB connection lost")

    monkeypatch.setattr(database, "get_conn", broken_conn)
    response = client.get("/health")
    assert response.status_code == 503
    data = response.json()["detail"]
    assert data["status"] == "unhealthy"
    assert "DB connection lost" in data["database"]


def test_healthz_alias(client):
    response = client.get("/healthz")
    assert response.status_code == 200


def test_assess_endpoint(client):
    response = client.post("/assess", json=sample_applicant_payload())
    assert response.status_code == 200
    data = response.json()
    assert "risk_probability" in data
    assert "decision" in data
    assert data["decision"] in ["APPROVE", "REVIEW", "REJECT"]
    assert "top_factors" in data
    assert "narrative" in data


def test_assess_full_persists_and_returns_id(client):
    response = client.post("/assess/full", json=sample_applicant_payload())
    assert response.status_code == 200
    data = response.json()
    assert "application_id" in data
    assert data["application_id"] > 0
    assert "risk_band" in data
    assert data["risk_band"] in ["Low", "Moderate", "High"]
    assert "fraud_flags" in data

    app_id = data["application_id"]
    get_resp = client.get(f"/applications/{app_id}")
    assert get_resp.status_code == 200
    record = get_resp.json()
    assert record["id"] == app_id
    assert record["decision"] == data["decision"]


def test_get_application_not_found(client):
    response = client.get("/applications/999999")
    assert response.status_code == 404


def test_high_loan_ratio_produces_elevated_risk(client):
    high_ratio = client.post(
        "/assess",
        json=sample_applicant_payload(annual_income=30000.0, loan_amount=20000.0),
    )
    low_ratio = client.post(
        "/assess",
        json=sample_applicant_payload(annual_income=120000.0, loan_amount=5000.0),
    )
    assert high_ratio.status_code == 200
    assert low_ratio.status_code == 200
    assert high_ratio.json()["risk_probability"] > low_ratio.json()["risk_probability"], (
        "A loan at 67% of income should score higher risk than one at 4% of income"
    )
