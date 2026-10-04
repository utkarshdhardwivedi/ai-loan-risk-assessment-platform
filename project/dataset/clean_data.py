import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "raw", "credit_risk_dataset.csv")
OUT_PATH = os.path.join(BASE_DIR, "loan_applications_clean.csv")
REPORT_PATH = os.path.join(BASE_DIR, "cleaning_report.txt")


def clean(raw_path: str = RAW_PATH, out_path: str = OUT_PATH, report_path: str = REPORT_PATH):
    lines = []

    def log(msg: str):
        print(msg)
        lines.append(msg)

    if not os.path.exists(raw_path):
        print(f"ERROR: raw CSV not found at {raw_path}")
        sys.exit(1)

    df = pd.read_csv(raw_path)
    start = len(df)

    df = df.drop_duplicates()
    df = df[df["person_age"] <= 80]

    impossible_mask = df["person_emp_length"] > 60
    df.loc[impossible_mask, "person_emp_length"] = float("nan")

    median_emp = df["person_emp_length"].median()
    df["person_emp_length"] = df["person_emp_length"].fillna(median_emp)

    cols_to_drop = [c for c in ["loan_grade", "loan_int_rate"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    df.to_csv(out_path, index=False)

    log(f"Cleaned {start} rows -> {len(df)} rows. Saved to {out_path}")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return df


if __name__ == "__main__":
    clean()
