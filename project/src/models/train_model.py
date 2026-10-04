import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "..", "dataset", "loan_applications_clean.csv")

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file",
]
TARGET = "loan_status"


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def main():
    df = pd.read_csv(DATASET_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    minority_metrics = report.get("1", {})
    print(f"ROC-AUC: {auc:.4f}")
    print(
        f"Minority class (default=1): "
        f"precision={minority_metrics.get('precision', 0):.4f}  "
        f"recall={minority_metrics.get('recall', 0):.4f}  "
        f"f1={minority_metrics.get('f1-score', 0):.4f}"
    )
    print(
        "NOTE: accuracy alone is misleading for imbalanced data (21.8% positive rate). "
        "ROC-AUC and minority-class F1/recall are the primary model quality indicators."
    )

    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    imps = pipeline.named_steps["model"].feature_importances_
    print("\nFeature importances:")
    for n, i in sorted(zip(names, imps), key=lambda x: -x[1]):
        print(f"  {i:.4f}  {n}")

    metrics = {
        "roc_auc": auc,
        "minority_class_precision": minority_metrics.get("precision"),
        "minority_class_recall": minority_metrics.get("recall"),
        "minority_class_f1": minority_metrics.get("f1-score"),
        "classification_report": report,
    }

    joblib.dump(pipeline, os.path.join(BASE_DIR, "risk_model.joblib"))
    with open(os.path.join(BASE_DIR, "feature_names.json"), "w") as f:
        json.dump(
            {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES}, f, indent=2
        )
    with open(os.path.join(BASE_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel, feature list, and metrics saved.")


if __name__ == "__main__":
    main()
