from unittest.mock import MagicMock, patch
from src.assistant import llm_client


def test_provider_detection_nim(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "nvapi-test12345")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("NIM_MODEL", raising=False)
    assert llm_client.is_nim() is True
    assert llm_client.get_api_key() == "nvapi-test12345"
    model = llm_client.get_model()
    assert model and isinstance(model, str), "get_model() should return a non-empty string"


def test_provider_detection_anthropic(monkeypatch):
    monkeypatch.delenv("NIM_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test12345")
    assert llm_client.is_nim() is False
    assert llm_client.get_api_key() == "sk-ant-test12345"
    assert "claude" in llm_client.get_model()


def test_call_llm_nim_content(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "nvapi-test12345")
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Loan risk is evaluated by debt ratio.",
                    "reasoning_content": "Chain of thought...",
                }
            }
        ]
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = llm_client.call_llm("Explain risk", max_tokens=100)
        assert result == "Loan risk is evaluated by debt ratio."
        assert mock_post.called
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer nvapi-test12345"


def test_call_llm_nim_fallback_to_reasoning_if_content_empty(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "nvapi-test12345")
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "DeepSeek reasoning chain content.",
                }
            }
        ]
    }

    with patch("requests.post", return_value=mock_resp):
        result = llm_client.call_llm("Explain risk", max_tokens=100)
        assert result == "DeepSeek reasoning chain content."
