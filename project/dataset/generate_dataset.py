import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 5000


def generate(n=N):
    age = RNG.integers(21, 65, n)
    annual_income = np.round(RNG.lognormal(mean=10.8, sigma=0.45, size=n), -3)
    annual_income = np.clip(annual_income, 15000, 400000)

    employment_years = np.clip(RNG.normal(6, 4.5, n), 0, 40)

    existing_debt = np.round(annual_income * RNG.uniform(0.0, 0.6, n), -2)
    loan_amount = np.round(annual_income * RNG.uniform(0.05, 1.2, n), -2)

    credit_history_years = np.clip(RNG.normal(8, 5, n), 0, 40)
    num_delinquencies = RNG.poisson(0.6, n)
    num_delinquencies = np.clip(num_delinquencies - (credit_history_years > 10).astype(int), 0, None)

    dti = (existing_debt + loan_amount * 0.12) / annual_income
    loan_to_income = loan_amount / annual_income

    home_ownership = RNG.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.4, 0.4, 0.2])
    purpose = RNG.choice(
        ["debt_consolidation", "home_improvement", "education", "medical", "business", "other"],
        n,
    )

    z = (
        -3.2
        + 3.1 * dti
        + 1.4 * loan_to_income
        + 0.55 * num_delinquencies
        - 0.05 * credit_history_years
        - 0.03 * employment_years
        - 0.000006 * annual_income
        + RNG.normal(0, 0.4, n)
    )
    prob_default = 1 / (1 + np.exp(-z))
    default = (RNG.uniform(0, 1, n) < prob_default).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income,
            "employment_years": np.round(employment_years, 1),
            "existing_debt": existing_debt,
            "loan_amount": loan_amount,
            "credit_history_years": np.round(credit_history_years, 1),
            "num_delinquencies": num_delinquencies,
            "debt_to_income": np.round(dti, 3),
            "loan_to_income": np.round(loan_to_income, 3),
            "home_ownership": home_ownership,
            "purpose": purpose,
            "default": default,
        }
    )
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("loan_applications.csv", index=False)
    print(f"Generated {len(df)} rows. Default rate: {df['default'].mean():.2%}")
