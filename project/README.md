# AI Loan Risk Assessment Platform

## Problem Being Solved

Manual credit underwriting can be slow, inconsistent, and opaque to applicants. This platform automates initial loan risk assessment, providing default probability scoring, plain-language explanations of contributing risk factors, automated document verification, and rule-based anomaly detection.

---

## Features

- **Credit Risk Scoring**: Automated risk assessment categorizing applicants into APPROVE, REVIEW, or REJECT decisions.
- **Explainable AI**: Plain-language breakdown of top factors driving risk predictions using model feature attributions.
- **Financial Document Verification**: PDF payslip and bank statement income extraction to cross-check self-reported figures.
- **Fraud & Anomaly Detection**: Rule-based checks for loan-to-income excesses, short employment histories, and document income mismatches.
- **AI Advisory Assistant**: Grounded Q&A assistant to answer applicant questions based on individual assessment results.
- **Reviewer Dashboard**: Streamlit web interface for submitting applications, reviewing history, and asking the assistant questions.
- **Persistence**: SQLite storage for tracking application history, decisions, and risk factors.

---

## Technology Stack

- **Language**: Python 3.13
- **Backend API**: FastAPI, Uvicorn, Pydantic
- **ML & Explainability**: Scikit-Learn (GradientBoostingClassifier), Joblib, SHAP
- **Document AI**: pdfplumber, Regex
- **Database**: SQLite
- **Frontend**: Streamlit
- **LLM Integration**: NVIDIA NIM / Anthropic API (optional)
- **Testing**: pytest, HTTPX

---

## Architecture

```text
Streamlit Dashboard (src/dashboard.py)
       │
       ▼
FastAPI Backend (src/api/main.py)
       │
       ├── ML Model & Explainer (src/models/train_model.py, src/explainability/explainer.py)
       ├── Fraud Checks (src/models/fraud_checks.py)
       ├── Document Parser (src/document_ai/parser.py)
       ├── SQLite DB (src/db/database.py)
       └── LLM Client & Advisor (src/assistant/llm_client.py, src/assistant/advisor.py)
```

See `docs/architecture.md` for full component details.

---

## Installation

1. Clone the repository and navigate to the project directory:

```bash
cd project
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
cp config/.env.example .env
```

---

## How to Use It

1. Open the **New Application** tab in the dashboard.
2. Fill in applicant financial parameters (age, income, employment years, loan amount, credit history).
3. Optionally upload a PDF payslip or bank statement.
4. Click **Run Risk Assessment** to receive a decision, risk score, narrative explanation, and fraud flags.
5. Use the **Applications** tab to view saved application history.
6. Use the **Ask the Assistant** tab to query specific application outcomes.

---

## Example Output

```json
{
  "application_id": 1,
  "decision": "REJECT",
  "risk_band": "High",
  "risk_probability": 0.765,
  "top_factors": [
    {"friendly_name": "loan as % of income", "direction": "increases risk", "contribution": 0.3394},
    {"friendly_name": "annual income", "direction": "increases risk", "contribution": 0.3064}
  ],
  "narrative": "Estimated default risk: 76.5% (High risk). Factors increasing risk: loan as % of income, annual income.",
  "fraud_flags": ["Loan-to-income ratio is high (82%), exceeding 50% of annual income."]
}
```

---

## Limitations

- Prototype model trained on synthetic and standard benchmark financial datasets.
- Single-node SQLite database intended for development and demo environments.
- LLM advisor features require optional API key configuration; degrades gracefully to template responses offline.

---

## Dataset & Training Pipeline

To download the Kaggle dataset, clean it, and retrain the model:

```bash
python dataset/download_kaggle_data.py
python dataset/clean_data.py
python src/models/train_model.py
```

---

## Running Tests

Execute the unit and integration test suite:

```bash
pytest tests/ -v
```

---

## How to Run the Application / Instructions to Start the Project

To start the full platform, launch both the API backend and the Streamlit dashboard:

### 1. Start the API Service

```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000` (Interactive docs at `http://localhost:8000/docs`).

### 2. Start the Dashboard

In a separate terminal window:

```bash
streamlit run src/dashboard.py
```

Access the user interface in your browser at `http://localhost:8501`.
