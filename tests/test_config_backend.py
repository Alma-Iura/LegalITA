from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from config import build_judge_runtime_config, validate_judge_runtime_config


class ConfigBackendTests(unittest.TestCase):
    def test_native_validation_preserves_existing_api_key_requirements(self) -> None:
        env = {
            "LLM_BACKEND": "native",
            "ANTHROPIC_API_KEY": "anthropic-test",
            "OPENAI_API_KEY": "openai-test",
        }
        with patch.dict(os.environ, env, clear=True):
            validate_judge_runtime_config(build_judge_runtime_config())

    def test_bedrock_validation_uses_one_bedrock_key(self) -> None:
        env = {
            "LLM_BACKEND": "bedrock",
            "AWS_BEARER_TOKEN_BEDROCK": "bedrock-test",
            "BEDROCK_REGION": "us-east-2",
            "BEDROCK_ANTHROPIC_SCOPE": "us",
        }
        with patch.dict(os.environ, env, clear=True):
            validate_judge_runtime_config(build_judge_runtime_config())

    def test_bedrock_validation_rejects_missing_key(self) -> None:
        env = {
            "LLM_BACKEND": "bedrock",
            "BEDROCK_REGION": "us-east-2",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "AWS_BEARER_TOKEN_BEDROCK"):
                validate_judge_runtime_config(build_judge_runtime_config())

    def test_bedrock_validation_rejects_unmapped_judge(self) -> None:
        env = {
            "LLM_BACKEND": "bedrock",
            "AWS_BEARER_TOKEN_BEDROCK": "bedrock-test",
            "BEDROCK_REGION": "us-east-2",
        }
        with patch.dict(os.environ, env, clear=True):
            config = build_judge_runtime_config(judge_b_model="gpt-4o")
            with self.assertRaisesRegex(RuntimeError, "non mappato su Amazon Bedrock"):
                validate_judge_runtime_config(config)


if __name__ == "__main__":
    unittest.main()
