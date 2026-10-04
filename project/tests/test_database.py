import os
import tempfile

import pytest

from src.db import database


@pytest.fixture(autouse=True)
def temp_db(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setattr(database, "DB_PATH", path)
    database.init_db()
    yield
    os.unlink(path)


def sample_record(**overrides):
    record = {
        "age": 30,
        "annual_income": 60000.0,
        "employment_years": 4.0,
        "loan_amount": 10000.0,
        "credit_history_years": 5.0,
        "home_ownership": "RENT",
        "loan_intent": "PERSONAL",
        "prior_default": "N",
        "loan_percent_income": 10000.0 / 60000.0,
        "risk_probability": 0.2,
        "risk_band": "Low",
        "decision": "APPROVE",
        "narrative": "Test narrative.",
        "top_factors": "[]",
    }
    record.update(overrides)
    return record


def test_save_and_retrieve_application():
    app_id = database.save_application(sample_record())
    record = database.get_application(app_id)
    assert record is not None
    assert record["decision"] == "APPROVE"
    assert record["loan_intent"] == "PERSONAL"
    assert record["prior_default"] == "N"


def test_list_applications_returns_most_recent_first():
    id1 = database.save_application(sample_record(decision="APPROVE"))
    id2 = database.save_application(sample_record(decision="REJECT"))
    records = database.list_applications()
    assert records[0]["id"] == id2
    assert records[1]["id"] == id1
