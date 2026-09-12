import asyncio

from embodied_ai.context import AgentContext
from embodied_ai.providers import GeminiProvider


class Response:
    def __init__(self, text: str, usage_metadata=None): self.text = text; self.usage_metadata = usage_metadata


class Models:
    def __init__(self, responses): self.responses = iter(responses); self.calls = 0
    def generate_content(self, **_kwargs):
        self.calls += 1
        result = next(self.responses)
        if isinstance(result, Exception): raise result
        return result


def provider_with(responses):
    provider = object.__new__(GeminiProvider)
    provider.model = "fake"; provider.timeout_seconds = 1
    models = Models(responses)
    provider.client = type("Client", (), {"models": models})()
    return provider, models


def test_gemini_provider_retries_transient_failures_without_network():
    provider, models = provider_with([ConnectionError("temporary"), ConnectionError("temporary"), Response('{"action":{"type":"wait"},"decision_summary":"safe"}')])
    decision, _latency = asyncio.run(provider.choose({"visible_cells": []}, AgentContext()))
    assert decision.action.type == "wait" and decision.decision_summary == "safe" and models.calls == 3
    assert provider.last_attempts == 3
    assert provider.last_backoff_ms == [250, 500]


def test_gemini_provider_rejects_malformed_structured_output():
    provider, _models = provider_with([Response('{"action":{"type":"teleport"}}')])
    try: asyncio.run(provider.choose({"visible_cells": []}, AgentContext()))
    except RuntimeError as error: assert "ProviderMalformedResponse" in str(error)
    else: raise AssertionError("malformed provider output was accepted")


def test_gemini_provider_records_reported_token_usage():
    usage = type("Usage", (), {"prompt_token_count": 12, "candidates_token_count": 7, "cached_content_token_count": None, "total_token_count": 19})()
    provider, _ = provider_with([Response('{"action":{"type":"wait"},"decision_summary":"safe"}', usage)])
    asyncio.run(provider.choose({"visible_cells": []}, AgentContext()))
    assert provider.last_token_usage == {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19}
