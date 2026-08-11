from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import provider_runtime


class ProviderRuntimeTests(unittest.TestCase):
    def test_native_keeps_logical_model_ids(self) -> None:
        with patch.dict(os.environ, {"LLM_BACKEND": "native"}, clear=True):
            claude = provider_runtime.resolve_model("anthropic", "claude-sonnet-4-6")
            gpt = provider_runtime.resolve_model("openai", "gpt-5.5")

        self.assertEqual(claude.api_model, "claude-sonnet-4-6")
        self.assertEqual(claude.transport, "anthropic_native")
        self.assertEqual(gpt.api_model, "gpt-5.5")
        self.assertEqual(gpt.transport, "openai_native")

    def test_bedrock_required_model_mappings(self) -> None:
        env = {
            "LLM_BACKEND": "bedrock",
            "AWS_BEARER_TOKEN_BEDROCK": "test-token",
            "BEDROCK_REGION": "us-east-2",
            "BEDROCK_ANTHROPIC_SCOPE": "us",
        }
        with patch.dict(os.environ, env, clear=True):
            sonnet = provider_runtime.resolve_model("anthropic", "claude-sonnet-4-6")
            opus = provider_runtime.resolve_model("anthropic", "claude-opus-4-8")
            gpt = provider_runtime.resolve_model("openai", "gpt-5.5")

        self.assertEqual(sonnet.api_model, "us.anthropic.claude-sonnet-4-6")
        self.assertEqual(opus.api_model, "us.anthropic.claude-opus-4-8")
        self.assertEqual(gpt.api_model, "openai.gpt-5.5")
        self.assertEqual(
            gpt.base_url,
            "https://bedrock-mantle.us-east-2.api.aws/openai/v1",
        )

    def test_bedrock_anthropic_scope_is_configurable(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_BACKEND": "bedrock",
                "BEDROCK_REGION": "eu-south-1",
                "BEDROCK_ANTHROPIC_SCOPE": "eu",
            },
            clear=True,
        ):
            target = provider_runtime.resolve_model("anthropic", "claude-sonnet-4-6")
        self.assertEqual(target.api_model, "eu.anthropic.claude-sonnet-4-6")

    def test_unsupported_bedrock_model_fails_fast(self) -> None:
        with patch.dict(os.environ, {"LLM_BACKEND": "bedrock"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "non mappato su Amazon Bedrock"):
                provider_runtime.resolve_model("openai", "gpt-4o")

    def test_anthropic_bedrock_client_uses_bearer_key_and_region(self) -> None:
        env = {
            "LLM_BACKEND": "bedrock",
            "AWS_BEARER_TOKEN_BEDROCK": "test-token",
            "BEDROCK_REGION": "us-east-2",
        }
        with patch.dict(os.environ, env, clear=True):
            target = provider_runtime.resolve_model("anthropic", "claude-sonnet-4-6")
            with patch.object(provider_runtime.anthropic, "AnthropicBedrock") as bedrock_cls:
                provider_runtime.create_anthropic_client(target)

        bedrock_cls.assert_called_once_with(
            api_key="test-token",
            aws_region="us-east-2",
        )

    def test_openai_bedrock_client_uses_bearer_key_and_base_url(self) -> None:
        env = {
            "LLM_BACKEND": "bedrock",
            "AWS_BEARER_TOKEN_BEDROCK": "test-token",
            "BEDROCK_REGION": "us-east-2",
        }
        with patch.dict(os.environ, env, clear=True):
            target = provider_runtime.resolve_model("openai", "gpt-5.5")
            with patch("provider_runtime.OpenAI") as openai_cls:
                provider_runtime.create_openai_client(target)

        openai_cls.assert_called_once_with(
            api_key="test-token",
            base_url="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
        )


if __name__ == "__main__":
    unittest.main()
