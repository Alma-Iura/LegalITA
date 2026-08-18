from __future__ import annotations

import os
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import model_query


REGISTRY_YAML = textwrap.dedent(
    """
    anthropic:
      interface: anthropic
      api_key_env: ANTHROPIC_API_KEY
    bedrock-anthropic:
      interface: bedrock-anthropic
      api_key_env: AWS_BEARER_TOKEN_BEDROCK
      region: us-east-2
      prefix: us.anthropic.
    openai:
      interface: openai
      api_key_env: OPENAI_API_KEY
    bedrock-openai:
      interface: bedrock-openai
      api_key_env: AWS_BEARER_TOKEN_BEDROCK
      region: us-east-2
      prefix: openai.
    """
).strip()


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
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = Path(self.tmpdir.name) / ".env.providers.yaml"
        self.registry_path.write_text(REGISTRY_YAML, encoding="utf-8")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_openai_namespaced_model_uses_bedrock_api_model_and_logical_metrics(self) -> None:
        env = {
            "LLM_PROVIDER_REGISTRY": str(self.registry_path),
            "AWS_BEARER_TOKEN_BEDROCK": "test-token",
        }
        client = _OpenAIClient()
        with patch.dict(os.environ, env, clear=True):
            with patch("model_query.create_openai_client", return_value=client):
                result = model_query.query_model_with_metrics("bedrock-openai:gpt-5.5", "domanda")

        request = client.chat.completions.last_request
        self.assertEqual(request["model"], "openai.gpt-5.5")
        self.assertEqual(request["messages"][0]["content"], "domanda")
        self.assertEqual(result.text, "risposta")
        self.assertEqual(result.metrics["provider"], "bedrock-openai")
        self.assertEqual(result.metrics["model"], "gpt-5.5")

    def test_anthropic_namespaced_model_uses_bedrock_api_model_and_logical_metrics(self) -> None:
        response = types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=" risposta ")],
            usage=None,
        )
        env = {
            "LLM_PROVIDER_REGISTRY": str(self.registry_path),
            "AWS_BEARER_TOKEN_BEDROCK": "test-token",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("model_query.create_anthropic_client", return_value=object()):
                with patch("model_query.stream_anthropic_message", return_value=response) as stream:
                    result = model_query.query_model_with_metrics(
                        "bedrock-anthropic:claude-sonnet-4-6",
                        "domanda",
                    )

        request = stream.call_args.kwargs["request"]
        self.assertEqual(request["model"], "us.anthropic.claude-sonnet-4-6")
        self.assertEqual(request["messages"][0]["content"], "domanda")
        self.assertEqual(result.text, "risposta")
        self.assertEqual(result.metrics["provider"], "bedrock-anthropic")
        self.assertEqual(result.metrics["model"], "claude-sonnet-4-6")


if __name__ == "__main__":
    unittest.main()
