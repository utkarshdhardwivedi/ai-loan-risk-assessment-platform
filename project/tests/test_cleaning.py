import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset.clean_data import clean


def _make_fixture_csv(tmp_path) -> str:
    rows = [
        {
            "person_age": 30,
            "person_income": 60000,
            "person_home_ownership": "RENT",
            "person_emp_length": 4.0,
            "loan_intent": "PERSONAL",
            "loan_grade": "B",
            "loan_amnt": 10000,
            "loan_int_rate": 11.5,
            "loan_status": 0,
            "loan_percent_income": 0.17,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 5,
        },
        {
            "person_age": 30,
            "person_income": 60000,
            "person_home_ownership": "RENT",
            "person_emp_length": 4.0,
            "loan_intent": "PERSONAL",
            "loan_grade": "B",
            "loan_amnt": 10000,
            "loan_int_rate": 11.5,
            "loan_status": 0,
            "loan_percent_income": 0.17,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 5,
        },
        {
            "person_age": 144,
            "person_income": 45000,
            "person_home_ownership": "OWN",
            "person_emp_length": 3.0,
            "loan_intent": "EDUCATION",
            "loan_grade": "C",
            "loan_amnt": 8000,
            "loan_int_rate": 13.0,
            "loan_status": 1,
            "loan_percent_income": 0.18,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 3,
        },
        {
            "person_age": 40,
            "person_income": 80000,
            "person_home_ownership": "MORTGAGE",
            "person_emp_length": 123.0,
            "loan_intent": "MEDICAL",
            "loan_grade": "A",
            "loan_amnt": 15000,
            "loan_int_rate": 9.5,
            "loan_status": 0,
            "loan_percent_income": 0.19,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 10,
        },
    ]
    df = pd.DataFrame(rows)
    path = str(tmp_path / "fixture.csv")
    df.to_csv(path, index=False)
    return path


def test_duplicate_row_is_dropped(tmp_path):
    raw = _make_fixture_csv(tmp_path)
    out = str(tmp_path / "clean.csv")
    report = str(tmp_path / "report.txt")
    df = clean(raw_path=raw, out_path=out, report_path=report)
    assert len(df) == 2


def test_age_outlier_is_dropped(tmp_path):
    raw = _make_fixture_csv(tmp_path)
    out = str(tmp_path / "clean.csv")
    report = str(tmp_path / "report.txt")
    df = clean(raw_path=raw, out_path=out, report_path=report)
    assert df["person_age"].max() <= 80


def test_impossible_emp_length_is_imputed(tmp_path):
    raw = _make_fixture_csv(tmp_path)
    out = str(tmp_path / "clean.csv")
    report = str(tmp_path / "report.txt")
    df = clean(raw_path=raw, out_path=out, report_path=report)
    assert df["person_emp_length"].isnull().sum() == 0
    assert df["person_emp_length"].max() <= 60


def test_loan_grade_and_int_rate_dropped(tmp_path):
    raw = _make_fixture_csv(tmp_path)
    out = str(tmp_path / "clean.csv")
    report = str(tmp_path / "report.txt")
    df = clean(raw_path=raw, out_path=out, report_path=report)
    assert "loan_grade" not in df.columns
    assert "loan_int_rate" not in df.columns


def test_report_file_written(tmp_path):
    raw = _make_fixture_csv(tmp_path)
    out = str(tmp_path / "clean.csv")
    report = str(tmp_path / "report.txt")
    clean(raw_path=raw, out_path=out, report_path=report)
    assert os.path.exists(report)
