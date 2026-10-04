from src.models.fraud_checks import check_application


def base_applicant(**overrides):
    applicant = {
        "annual_income": 60000.0,
        "employment_years": 4.0,
        "loan_amount": 15000.0,
        "loan_percent_income": 15000.0 / 60000.0,
        "prior_default": "N",
        "risk_probability": 0.15,
    }
    applicant.update(overrides)
    return applicant


def test_no_flags_for_normal_application():
    flags = check_application(base_applicant())
    assert flags == []


def test_flags_high_loan_percent_income():
    applicant = base_applicant(loan_amount=40000.0, loan_percent_income=40000.0 / 60000.0)
    flags = check_application(applicant)
    assert any("loan-to-income" in f.lower() for f in flags)


def test_flags_prior_default_with_low_model_risk():
    applicant = base_applicant(prior_default="Y", risk_probability=0.10)
    flags = check_application(applicant)
    assert any("prior default" in f.lower() for f in flags)


def test_no_flag_prior_default_when_model_risk_is_high():
    applicant = base_applicant(prior_default="Y", risk_probability=0.60)
    flags = check_application(applicant)
    assert not any("prior default" in f.lower() for f in flags)


def test_flags_high_income_short_employment():
    applicant = base_applicant(employment_years=0.5, annual_income=200000.0)
    flags = check_application(applicant)
    assert any("employment" in f.lower() for f in flags)


def test_flags_income_document_mismatch():
    applicant = base_applicant(annual_income=100000.0)
    flags = check_application(applicant, extracted_income=40000.0)
    assert any("document-extracted" in f for f in flags)
