from __future__ import annotations

import os
import types
import unittest
from unittest.mock import patch

import model_query


class _ChatCompletions:
    def __init__(self) -> None:
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=" risposta "),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )


class _OpenAIClient:
    def __init__(self) -> None:
        self.chat = types.SimpleNamespace(completions=_ChatCompletions())


class ModelQueryBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = {
            "LLM_BACKEND": "bedrock",
            "AWS_BEARER_TOKEN_BEDROCK": "test-token",
            "BEDROCK_REGION": "us-east-2",
            "BEDROCK_ANTHROPIC_SCOPE": "us",
        }

    def test_openai_evaluated_model_uses_bedrock_id_and_keeps_logical_metrics(self) -> None:
        client = _OpenAIClient()
        with patch.dict(os.environ, self.env, clear=True):
            with patch("model_query.create_openai_client", return_value=client):
                result = model_query.query_openai_with_metrics("gpt-5.5", "domanda")

        request = client.chat.completions.last_request
        self.assertEqual(request["model"], "openai.gpt-5.5")
        self.assertEqual(request["messages"][0]["content"], "domanda")
        self.assertEqual(result.text, "risposta")
        self.assertEqual(result.metrics["model"], "gpt-5.5")

    def test_anthropic_evaluated_model_uses_bedrock_id_and_keeps_logical_metrics(self) -> None:
        response = types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=" risposta ")],
            usage=None,
        )
        with patch.dict(os.environ, self.env, clear=True):
            with patch("model_query.create_anthropic_client", return_value=object()):
                with patch("model_query.stream_anthropic_message", return_value=response) as stream:
                    result = model_query.query_anthropic_with_metrics(
                        "claude-sonnet-4-6",
                        "domanda",
                    )

        request = stream.call_args.kwargs["request"]
        self.assertEqual(request["model"], "us.anthropic.claude-sonnet-4-6")
        self.assertEqual(request["messages"][0]["content"], "domanda")
        self.assertEqual(result.text, "risposta")
        self.assertEqual(result.metrics["model"], "claude-sonnet-4-6")


if __name__ == "__main__":
    unittest.main()
