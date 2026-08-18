from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from config import build_judge_runtime_config, validate_judge_runtime_config


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
    """
).strip()


class ConfigBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = Path(self.tmpdir.name) / ".env.providers.yaml"
        self.registry_path.write_text(REGISTRY_YAML, encoding="utf-8")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_namespaced_models_validate(self) -> None:
        env = {
            "LLM_PROVIDER_REGISTRY": str(self.registry_path),
            "ANTHROPIC_API_KEY": "anthropic-test",
            "OPENAI_API_KEY": "openai-test",
            "JUDGE_A_MODEL": "anthropic:claude-sonnet-4-6",
            "JUDGE_B_MODEL": "openai:gpt-5.5",
            "JUDGE_C_MODEL": "bedrock-anthropic:claude-opus-4-8",
            "AWS_BEARER_TOKEN_BEDROCK": "bedrock-test",
        }
        with patch.dict(os.environ, env, clear=True):
            validate_judge_runtime_config(build_judge_runtime_config())

    def test_legacy_provider_inputs_are_rejected(self) -> None:
        env = {
            "LLM_PROVIDER_REGISTRY": str(self.registry_path),
            "JUDGE_A_PROVIDER": "anthropic",
            "JUDGE_A_MODEL": "anthropic:claude-sonnet-4-6",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, r"JUDGE_\[A\|B\|C\]_PROVIDER"):
                build_judge_runtime_config()

    def test_non_namespaced_model_is_rejected(self) -> None:
        env = {
            "LLM_PROVIDER_REGISTRY": str(self.registry_path),
            "JUDGE_A_MODEL": "claude-sonnet-4-6",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "namespace:model"):
                build_judge_runtime_config()


if __name__ == "__main__":
    unittest.main()
