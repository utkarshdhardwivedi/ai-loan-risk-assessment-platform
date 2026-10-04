# System Architecture

## Data Flow

1. Applicant submits financial details via Streamlit UI or API (`POST /assess/full`).
2. Optional supporting PDF processed via `src/document_ai/parser.py`.
3. Financial ratios evaluated through Scikit-Learn pipeline (`src/models/train_model.py`).
4. Risk score computed and SHAP / Feature importances converted into narrative (`src/explainability/explainer.py`).
5. Anomaly flags evaluated using rule-based checks (`src/models/fraud_checks.py`).
6. Decision and assessment record persisted into SQLite database (`src/db/database.py`).
7. User queries past assessments via LLM advisor assistant (`src/assistant/advisor.py`).

## Component Breakdown

| Module | Purpose |
|---|---|
| `src/api/main.py` | FastAPI application endpoints |
| `src/dashboard.py` | Streamlit interactive frontend |
| `src/models/train_model.py` | Model training pipeline |
| `src/models/fraud_checks.py` | Rule-based fraud triage |
| `src/explainability/explainer.py` | SHAP/Feature importance narratives |
| `src/document_ai/parser.py` | PDF income parsing engine |
| `src/assistant/llm_client.py` | NIM / Anthropic provider router |
| `src/assistant/advisor.py` | Grounded Q&A assistant |
| `src/db/database.py` | SQLite persistence layer |
| `config/settings.py` | Environment configuration |
