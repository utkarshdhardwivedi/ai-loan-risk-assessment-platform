from src.assistant.llm_client import call_llm, is_configured

SYSTEM_PROMPT = (
    "You are a loan advisory assistant for a bank. You are given a "
    "structured risk assessment (probability, contributing factors) for a "
    "specific applicant. Answer the user's question using ONLY that data. "
    "Be concise, factual, and avoid making promises about approval. "
    "Suggest concrete, actionable steps where relevant (e.g. reduce "
    "debt-to-income ratio, build credit history)."
)


def answer_question(question: str, assessment: dict) -> str:
    context = (
        f"Risk probability: {assessment['risk_probability']:.1%}\n"
        f"Narrative: {assessment['narrative']}\n"
        f"Top factors: {assessment['top_factors']}\n"
    )

    if not is_configured():
        return _template_answer(question, assessment)

    prompt = f"Assessment data:\n{context}\n\nUser question: {question}"
    try:
        return call_llm(prompt, system=SYSTEM_PROMPT, max_tokens=4096)
    except Exception as exc:
        print(f"[advisor] LLM call failed ({type(exc).__name__}: {exc}), using template answer")
        return _template_answer(question, assessment)


def _template_answer(question: str, assessment: dict) -> str:
    increasing = [f for f in assessment.get("top_factors") or [] if f["contribution"] > 0]
    if increasing:
        tips = "; ".join(f"work on your {f['friendly_name']}" for f in increasing[:3])
        return f"{assessment['narrative']} To improve your standing, consider: {tips}."
    return assessment["narrative"]
