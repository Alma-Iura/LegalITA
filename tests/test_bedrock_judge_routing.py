from __future__ import annotations

import os
import types
import unittest
from unittest.mock import patch

from evaluation.judge import AnthropicJudge, OpenAIJudge


class _AnthropicMessages:
    def __init__(self) -> None:
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(text='{"verdict":"pass","reasoning":"ok"}')]
        )


class _AnthropicClient:
    def __init__(self) -> None:
        self.messages = _AnthropicMessages()


class _OpenAIResponses:
    def __init__(self) -> None:
        self.create_request = None
        self.parse_called = False

    def create(self, **kwargs):
        self.create_request = kwargs
        return types.SimpleNamespace(
            output_text='{"verdict":"pass","reasoning":"ok"}',
            output=[],
        )

    def parse(self, **kwargs):
        self.parse_called = True
        raise AssertionError("Bedrock path must not call responses.parse")


class _OpenAIClient:
    def __init__(self) -> None:
        self.responses = _OpenAIResponses()


class BedrockJudgeRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = {
            "LLM_BACKEND": "bedrock",
            "AWS_BEARER_TOKEN_BEDROCK": "test-token",
            "BEDROCK_REGION": "us-east-2",
            "BEDROCK_ANTHROPIC_SCOPE": "us",
        }

    def test_anthropic_request_uses_bedrock_id_but_vote_keeps_logical_name(self) -> None:
        client = _AnthropicClient()
        with patch.dict(os.environ, self.env, clear=True):
            judge = AnthropicJudge(model="claude-sonnet-4-6", client=client, max_retries=1)
            vote = judge.evaluate("q", "a", "criterion", "PASS se corretto")

        self.assertEqual(
            client.messages.last_request["model"],
            "us.anthropic.claude-sonnet-4-6",
        )
        self.assertEqual(vote.model, "claude-sonnet-4-6")
        self.assertEqual(vote.verdict, "pass")

    def test_openai_bedrock_judge_uses_responses_create(self) -> None:
        client = _OpenAIClient()
        with patch.dict(os.environ, self.env, clear=True):
            judge = OpenAIJudge(judge_id="B", model="gpt-5.5", client=client, max_retries=1)
            vote = judge.evaluate("q", "a", "criterion", "PASS se corretto")

        self.assertFalse(client.responses.parse_called)
        self.assertEqual(client.responses.create_request["model"], "openai.gpt-5.5")
        self.assertEqual(vote.model, "gpt-5.5")
        self.assertEqual(vote.verdict, "pass")


if __name__ == "__main__":
    unittest.main()
