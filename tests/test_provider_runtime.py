from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import provider_runtime


REGISTRY_YAML = textwrap.dedent(
    """
    anthropic:
      interface: anthropic
      api_key_env: ANTHROPIC_API_KEY
    openai:
      interface: openai
      api_key_env: OPENAI_API_KEY
    bedrock-anthropic:
      interface: bedrock-anthropic
      api_key_env: AWS_BEARER_TOKEN_BEDROCK
      region: us-east-2
      prefix: us.anthropic.
    bedrock-openai:
      interface: bedrock-openai
      api_key_env: AWS_BEARER_TOKEN_BEDROCK
      region: us-east-2
      prefix: openai.
    gpu-office:
      interface: openai
      api_key_env: GPU_OFFICE_API_KEY
      base_url: https://llm-gateway.example.internal/v1
      timeout_s: 120
      verify_tls: false
      max_retries: 1
      pricing:
        mode: hourly
        models:
          qwen2.5:72b:
            cost_per_hour: 1.8
          default:
            cost_per_hour: 1.2
    """
).strip()


class ProviderRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = Path(self.tmpdir.name) / ".env.providers.yaml"
        self.registry_path.write_text(REGISTRY_YAML, encoding="utf-8")
        provider_runtime.reset_provider_registry_cache()

    def tearDown(self) -> None:
        provider_runtime.reset_provider_registry_cache()
        self.tmpdir.cleanup()

    def test_parse_model_target_splits_on_first_colon(self) -> None:
        namespace, model = provider_runtime.parse_model_target("gpu-office:qwen2.5:72b")
        self.assertEqual(namespace, "gpu-office")
        self.assertEqual(model, "qwen2.5:72b")

    def test_non_namespaced_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "namespace:model"):
            provider_runtime.parse_model_target("claude-sonnet-4-6")

    def test_openai_namespace_resolution_uses_registry(self) -> None:
        env = {
            "LLM_PROVIDER_REGISTRY": str(self.registry_path),
            "OPENAI_API_KEY": "openai-test",
        }
        with patch.dict(os.environ, env, clear=True):
            target = provider_runtime.resolve_model("openai:gpt-5.5")

        self.assertEqual(target.namespace, "openai")
        self.assertEqual(target.interface, "openai")
        self.assertEqual(target.api_model, "gpt-5.5")

    def test_bedrock_prefix_and_region_are_applied(self) -> None:
        env = {
            "LLM_PROVIDER_REGISTRY": str(self.registry_path),
            "AWS_BEARER_TOKEN_BEDROCK": "test-token",
        }
        with patch.dict(os.environ, env, clear=True):
            sonnet = provider_runtime.resolve_model("bedrock-anthropic:claude-sonnet-4-6")
            gpt = provider_runtime.resolve_model("bedrock-openai:gpt-5.5")

        self.assertEqual(sonnet.api_model, "us.anthropic.claude-sonnet-4-6")
        self.assertEqual(sonnet.region, "us-east-2")
        self.assertEqual(gpt.api_model, "openai.gpt-5.5")
        self.assertEqual(gpt.base_url, "https://bedrock-mantle.us-east-2.api.aws/openai/v1")

    def test_unknown_namespace_fails_with_registry_path(self) -> None:
        env = {
            "LLM_PROVIDER_REGISTRY": str(self.registry_path),
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Namespace provider sconosciuto"):
                provider_runtime.resolve_model("unknown-lab:model")

    def test_missing_credential_env_fails_fast(self) -> None:
        env = {
            "LLM_PROVIDER_REGISTRY": str(self.registry_path),
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                provider_runtime.resolve_model("openai:gpt-5.5")

    def test_bedrock_anthropic_client_uses_registry_credential_and_region(self) -> None:
        env = {
            "LLM_PROVIDER_REGISTRY": str(self.registry_path),
            "AWS_BEARER_TOKEN_BEDROCK": "test-token",
        }
        with patch.dict(os.environ, env, clear=True):
            target = provider_runtime.resolve_model("bedrock-anthropic:claude-sonnet-4-6")
            with patch.object(provider_runtime.anthropic, "AnthropicBedrock") as bedrock_cls:
                provider_runtime.create_anthropic_client(target)

        bedrock_cls.assert_called_once_with(
            api_key="test-token",
            aws_region="us-east-2",
        )

    def test_openai_client_uses_registry_options(self) -> None:
        env = {
            "LLM_PROVIDER_REGISTRY": str(self.registry_path),
            "GPU_OFFICE_API_KEY": "gpu-key",
        }
        with patch.dict(os.environ, env, clear=True):
            target = provider_runtime.resolve_model("gpu-office:qwen2.5:72b")
            with patch("provider_runtime.OpenAI") as openai_cls:
                provider_runtime.create_openai_client(target)

        openai_cls.assert_called_once_with(
            api_key="gpu-key",
            base_url="https://llm-gateway.example.internal/v1",
            timeout=120.0,
            max_retries=1,
        )


if __name__ == "__main__":
    unittest.main()
