import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "..", "loan_platform.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    age INTEGER,
    annual_income REAL,
    employment_years REAL,
    loan_amount REAL,
    credit_history_years REAL,
    home_ownership TEXT,
    loan_intent TEXT,
    prior_default TEXT,
    loan_percent_income REAL,
    risk_probability REAL,
    risk_band TEXT,
    decision TEXT,
    narrative TEXT,
    fraud_flags TEXT,
    top_factors TEXT
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(SCHEMA)


def save_application(record: dict) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO applications (
                created_at, age, annual_income, employment_years,
                loan_amount, credit_history_years, home_ownership,
                loan_intent, prior_default, loan_percent_income,
                risk_probability, risk_band, decision, narrative, fraud_flags,
                top_factors
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                record["age"],
                record["annual_income"],
                record["employment_years"],
                record["loan_amount"],
                record["credit_history_years"],
                record["home_ownership"],
                record["loan_intent"],
                record["prior_default"],
                record.get("loan_percent_income", 0.0),
                record["risk_probability"],
                record["risk_band"],
                record["decision"],
                record["narrative"],
                record.get("fraud_flags", ""),
                record.get("top_factors", ""),
            ),
        )
        return cur.lastrowid


def list_applications(limit: int = 200):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM applications ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_application(app_id: int):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM applications WHERE id = ?", (app_id,)
        ).fetchone()
        return dict(row) if row else None
