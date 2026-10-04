import os

for p in [
    os.path.join(os.path.dirname(__file__), "..", ".env"),
    os.path.join(os.path.dirname(__file__), ".env"),
    os.path.join(os.path.dirname(__file__), ".env.example"),
]:
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k not in os.environ:
                        os.environ[k] = v
        break

NIM_API_KEY = os.environ.get("NIM_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
NIM_MODEL = os.environ.get("NIM_MODEL", os.environ.get("ANTHROPIC_MODEL", "deepseek-ai/deepseek-v4.1-flash"))
NIM_BASE_URL = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
ANTHROPIC_API_KEY = NIM_API_KEY

API_URL = os.environ.get("API_URL", "http://localhost:8000")
DECISION_THRESHOLDS = {
    "approve_below": 0.25,
    "reject_above": 0.50,
}


def get_decision(risk_probability: float) -> str:
    if risk_probability < DECISION_THRESHOLDS["approve_below"]:
        return "APPROVE"
    if risk_probability < DECISION_THRESHOLDS["reject_above"]:
        return "REVIEW"
    return "REJECT"


def get_risk_band(risk_probability: float) -> str:
    if risk_probability >= DECISION_THRESHOLDS["reject_above"]:
        return "High"
    if risk_probability >= DECISION_THRESHOLDS["approve_below"]:
        return "Moderate"
    return "Low"
