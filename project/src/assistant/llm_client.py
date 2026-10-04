import os
import re
import requests

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def get_api_key() -> str:
    return os.environ.get("NIM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or ""


def is_nim() -> bool:
    key = get_api_key()
    return bool(os.environ.get("NIM_API_KEY") or key.startswith("nvapi-"))


def get_model() -> str:
    if is_nim():
        return os.environ.get("NIM_MODEL") or "deepseek-ai/deepseek-v4.1-flash"
    return os.environ.get("ANTHROPIC_MODEL") or "claude-3-5-sonnet-20241022"


def is_configured() -> bool:
    return bool(get_api_key())


def _strip_reasoning(text: str) -> str:
    if not text:
        return text

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    for marker in [
        "**final answer:**",
        "**recommendation:**",
        "**conclusion:**",
        "**response:**",
        "**answer:**",
    ]:
        lower = text.lower()
        idx = lower.rfind(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()

    if re.match(r"(here'?s? a thinking|here is a thinking|\d+\.\s+\*\*)", text, re.IGNORECASE):
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(parts) > 1:
            return parts[-1]

    return text


def call_llm(prompt: str, system: str = "", max_tokens: int = 500) -> str:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("No LLM API key configured (set NIM_API_KEY or ANTHROPIC_API_KEY).")

    if is_nim():
        url = os.environ.get("NIM_BASE_URL", NIM_URL)
        model = get_model()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=240)
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""

        if reasoning and not content:
            content = reasoning

        return _strip_reasoning(content.strip())
    else:
        url = os.environ.get("ANTHROPIC_API_URL", ANTHROPIC_URL)
        model = get_model()
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return "".join(block.get("text", "") for block in data.get("content", []))
