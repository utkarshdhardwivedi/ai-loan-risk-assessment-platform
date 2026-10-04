def check_application(applicant: dict, extracted_income: float | None = None):
    flags = []

    if applicant.get("loan_percent_income", 0) > 0.5:
        flags.append(
            f"Loan-to-income ratio is high ({applicant['loan_percent_income']:.0%}), "
            "exceeding 50% of annual income."
        )

    if (
        applicant.get("prior_default") == "Y"
        and applicant.get("risk_probability", 1.0) < 0.25
    ):
        flags.append(
            "Applicant has a prior default on record but model risk score is low — "
            "recommend manual review for consistency."
        )

    if applicant.get("employment_years", 1) < 1 and applicant.get("annual_income", 0) > 150000:
        flags.append("High reported income with very limited employment history.")

    if extracted_income is not None:
        reported = applicant.get("annual_income", 0)
        if reported > 0:
            diff_ratio = abs(reported - extracted_income) / reported
            if diff_ratio > 0.25:
                flags.append(
                    f"Self-reported income differs from document-extracted income by "
                    f"{diff_ratio:.0%}."
                )

    return flags
