import os
import re

import pdfplumber

ANNUAL_PATTERNS = [
    r"(?:annual\s+income|gross\s+annual\s+income|annual\s+salary|total\s+annual\s+income)[:\s]*\$?([\d,]+(?:\.\d{2})?)",
    r"(?:base\s+salary|gross\s+pay|net\s+pay|salary)[:\s]*\$?([\d,]+(?:\.\d{2})?)\s*(?:/\s*yr|/\s*year|per\s+year|annually)",
]

MONTHLY_PATTERNS = [
    r"(?:monthly\s+salary|monthly\s+pay|monthly\s+income)[:\s]*\$?([\d,]+(?:\.\d{2})?)",
    r"(?:gross\s+pay|net\s+pay|base\s+pay|total\s+pay)[:\s]*\$?([\d,]+(?:\.\d{2})?)\s*(?:/\s*mo|/\s*month|per\s+month)",
]

BIWEEKLY_PATTERNS = [
    r"(?:bi-?weekly\s+salary|bi-?weekly\s+pay|bi-?weekly\s+income)[:\s]*\$?([\d,]+(?:\.\d{2})?)",
    r"(?:gross\s+pay|net\s+pay|base\s+pay|total\s+pay)[:\s]*\$?([\d,]+(?:\.\d{2})?)\s*(?:/\s*bi-?week|per\s+bi-?week)",
]

WEEKLY_PATTERNS = [
    r"(?:weekly\s+salary|weekly\s+pay|weekly\s+income)[:\s]*\$?([\d,]+(?:\.\d{2})?)",
    r"(?:gross\s+pay|net\s+pay|base\s+pay|total\s+pay)[:\s]*\$?([\d,]+(?:\.\d{2})?)\s*(?:/\s*wk|/\s*week|per\s+week)",
]

GENERIC_PAY_PATTERNS = [
    r"(?:net\s+pay|gross\s+pay|total\s+income)[:\s]*\$?([\d,]+(?:\.\d{2})?)",
]


def extract_text(pdf_path: str) -> str:
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
    return "\n".join(text_chunks)


def _detect_pay_frequency(lowered: str) -> int:
    if re.search(r"pay\s+frequency[:\s]*(?:bi-?weekly|every\s+two\s+weeks)", lowered):
        return 26
    if re.search(r"pay\s+frequency[:\s]*weekly", lowered):
        return 52
    if re.search(r"pay\s+frequency[:\s]*monthly", lowered):
        return 12
    if re.search(r"pay\s+frequency[:\s]*(?:annual|annually|yearly)", lowered):
        return 1
    if "bi-weekly" in lowered or "biweekly" in lowered:
        return 26
    if "per month" in lowered or "monthly" in lowered:
        return 12
    if "per week" in lowered or "weekly" in lowered:
        return 52
    return 1


def extract_income_heuristic(text: str):
    lowered = text.lower()

    for pattern in ANNUAL_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            return round(float(match.group(1).replace(",", "")), 2)

    for pattern in MONTHLY_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            return round(float(match.group(1).replace(",", "")) * 12, 2)

    for pattern in BIWEEKLY_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            return round(float(match.group(1).replace(",", "")) * 26, 2)

    for pattern in WEEKLY_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            return round(float(match.group(1).replace(",", "")) * 52, 2)

    for pattern in GENERIC_PAY_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            val = float(match.group(1).replace(",", ""))
            multiplier = _detect_pay_frequency(lowered)
            return round(val * multiplier, 2)

    return None


def extract_income_llm(text: str, call_llm_fn):
    prompt = (
        "Extract the applicant's estimated ANNUAL income (in numbers only, "
        "no currency symbols or commas) from the following financial "
        "document text. If multiple figures appear, use the most relevant "
        "salary/income figure and annualize it if it looks monthly. "
        "Respond with ONLY the number.\n\n"
        f"Document text:\n{text[:4000]}"
    )
    response = call_llm_fn(prompt)
    digits = re.sub(r"[^\d.]", "", response)
    try:
        return float(digits) if digits else None
    except ValueError:
        return None


def parse_document(pdf_path: str, call_llm_fn=None):
    text = extract_text(pdf_path)
    income = extract_income_heuristic(text)
    method = "regex"

    if income is None and call_llm_fn is not None:
        income = extract_income_llm(text, call_llm_fn)
        method = "llm"

    return {"extracted_text_preview": text[:500], "extracted_income": income, "method": method}
