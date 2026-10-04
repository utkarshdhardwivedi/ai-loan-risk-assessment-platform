from src.document_ai.parser import extract_income_heuristic


def test_extract_income_annual_pattern():
    text = "Applicant Details\nAnnual income: $75,000.00\nEmployer: Acme Corp"
    income = extract_income_heuristic(text)
    assert income == 75000.0


def test_extract_income_low_annual_not_multiplied():
    text = "Tax Summary\nGross annual income: $18,500.00\nStatus: Part-time"
    income = extract_income_heuristic(text)
    assert income == 18500.0


def test_extract_income_monthly():
    text = "Earnings Statement\nMonthly salary: $4,500.00\nDepartment: Engineering"
    income = extract_income_heuristic(text)
    assert income == 54000.0


def test_extract_income_biweekly():
    text = "Pay stub\nBi-weekly pay: $2,000.00\nDirect Deposit"
    income = extract_income_heuristic(text)
    assert income == 52000.0


def test_extract_income_frequency_in_context():
    text = "Payroll Voucher\nPay Frequency: Monthly\nNet Pay: $3,200.00"
    income = extract_income_heuristic(text)
    assert income == 38400.0


def test_extract_income_none_when_no_pattern():
    text = "General agreement terms and conditions with no monetary amounts."
    income = extract_income_heuristic(text)
    assert income is None
