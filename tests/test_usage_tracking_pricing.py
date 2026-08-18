from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from usage_tracking import estimate_cost_usd


REGISTRY_YAML = textwrap.dedent(
    """
    openai:
      interface: openai
      api_key_env: OPENAI_API_KEY
      pricing:
        mode: token
        models:
          gpt-5.5:
            input_per_1m: 5.0
            output_per_1m: 15.0
            cached_input_per_1m: 1.0
    gpu-office:
      interface: openai
      api_key_env: GPU_OFFICE_API_KEY
      pricing:
        mode: hourly
        models:
          qwen2.5:72b:
            cost_per_hour: 2.0
          default:
            cost_per_hour: 1.5
    bare:
      interface: openai
      api_key_env: BARE_API_KEY
    """
).strip()


class UsageTrackingPricingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = Path(self.tmpdir.name) / ".env.providers.yaml"
        self.registry_path.write_text(REGISTRY_YAML, encoding="utf-8")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_token_pricing_is_computed_from_tokens(self) -> None:
        env = {"LLM_PROVIDER_REGISTRY": str(self.registry_path)}
        usage = {
            "input_tokens": 100_000,
            "output_tokens": 50_000,
            "cached_input_tokens": 20_000,
        }
        with patch.dict(os.environ, env, clear=True):
            cost, source = estimate_cost_usd(
                provider="openai",
                model="gpt-5.5",
                token_usage=usage,
                namespace="openai",
                latency_ms_value=200,
            )

        self.assertEqual(source, "registry:openai")
        self.assertAlmostEqual(cost, 1.17)

    def test_hourly_pricing_uses_exact_model_then_default(self) -> None:
        env = {"LLM_PROVIDER_REGISTRY": str(self.registry_path)}
        usage = {
            "input_tokens": None,
            "output_tokens": None,
            "cached_input_tokens": None,
        }
        with patch.dict(os.environ, env, clear=True):
            exact_cost, _ = estimate_cost_usd(
                provider="gpu-office",
                model="qwen2.5:72b",
                token_usage=usage,
                namespace="gpu-office",
                latency_ms_value=1_800_000,
            )
            fallback_cost, _ = estimate_cost_usd(
                provider="gpu-office",
                model="another-model",
                token_usage=usage,
                namespace="gpu-office",
                latency_ms_value=3_600_000,
            )

        self.assertAlmostEqual(exact_cost or 0.0, 1.0)
        self.assertAlmostEqual(fallback_cost or 0.0, 1.5)

    def test_unknown_pricing_returns_none(self) -> None:
        env = {"LLM_PROVIDER_REGISTRY": str(self.registry_path)}
        with patch.dict(os.environ, env, clear=True):
            cost, source = estimate_cost_usd(
                provider="bare",
                model="model-x",
                token_usage={"input_tokens": 1, "output_tokens": 1, "cached_input_tokens": 0},
                namespace="bare",
                latency_ms_value=100,
            )
        self.assertIsNone(cost)
        self.assertIsNone(source)


if __name__ == "__main__":
    unittest.main()
