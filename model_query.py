"""Shared model querying layer using namespaced provider registry targets."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Protocol

import config as benchmark_config

from config import MODEL_MAX_TOKENS
from model_request_config import anthropic_message_kwargs, openai_completion_kwargs
from model_runtime import (
    anthropic_response_text,
    is_non_retryable_model_error,
    stream_anthropic_message,
)
from provider_runtime import ResolvedModel, create_anthropic_client, create_openai_client, resolve_model
from usage_tracking import (
    ModelCallResult,
    build_model_call_metrics,
    extract_usage,
    latency_ms,
)

log = logging.getLogger(__name__)


class ModelProviderAdapter(Protocol):
    interface: str

    def query_with_metrics(self, target: ResolvedModel, query: str) -> ModelCallResult:
        """Execute one model query and return text + call metrics."""

    def request_kwargs_for_summary(self, target: ResolvedModel) -> dict[str, object]:
        """Build stable request config summary for reporting."""


def run_query_with_retries(
    adapter: Callable[[str, str], object],
    model: str,
    query: str,
    *,
    max_retries: int,
    log: logging.Logger,
    validate: Callable[[object], None] | None = None,
) -> object | None:
    for attempt in range(1, max_retries + 1):
        try:
            result = adapter(model, query)
            if validate is not None:
                validate(result)
            return result
        except Exception as exc:
            if is_non_retryable_model_error(exc):
                log.error(
                    "Errore non recuperabile query %s (tentativo %d/%d): %s",
                    model,
                    attempt,
                    max_retries,
                    exc,
                )
                return None
            if attempt >= max_retries:
                log.warning(
                    "Errore ritentabile query %s (tentativo %d/%d): %s; tentativi esauriti",
                    model,
                    attempt,
                    max_retries,
                    exc,
                )
                break
            delay = 2.0 * (2 ** (attempt - 1))
            log.warning(
                "Errore ritentabile query %s (tentativo %d/%d): %s; attendo %.1fs",
                model,
                attempt,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)

    log.error("Query fallita dopo %d tentativi: %s", max_retries, model)
    return None


def require_result_text(result: ModelCallResult) -> None:
    if not result.text:
        raise ValueError("Risposta vuota dal modello.")


def require_answer_text(answer: str) -> None:
    if not answer:
        raise ValueError("Risposta vuota dal modello.")


class AnthropicAdapter:
    interface = "anthropic"

    def query_with_metrics(self, target: ResolvedModel, query: str) -> ModelCallResult:
        client = create_anthropic_client(target)
        started_at = time.perf_counter()
        request = anthropic_message_kwargs(target.logical_model, query, MODEL_MAX_TOKENS)
        request["model"] = target.api_model
        response = stream_anthropic_message(
            client,
            logger=log,
            config_module=benchmark_config,
            model=target.logical_model,
            model_max_tokens=MODEL_MAX_TOKENS,
            request=request,
        )
        text = anthropic_response_text(response)
        if not text:
            content_types = [
                getattr(block, "type", type(block).__name__)
                for block in getattr(response, "content", []) or []
            ]
            raise ValueError(
                "Risposta Anthropic priva di blocchi testuali finali "
                f"(content_types={content_types})."
            )
        return ModelCallResult(
            text=text,
            metrics=build_model_call_metrics(
                provider=target.namespace,
                model=target.logical_model,
                latency_ms_value=latency_ms(started_at),
                usage=extract_usage(response),
                namespace=target.namespace,
                latency_ms_for_hourly=latency_ms(started_at),
            ),
        )

    def request_kwargs_for_summary(self, target: ResolvedModel) -> dict[str, object]:
        request = anthropic_message_kwargs(target.logical_model, "", MODEL_MAX_TOKENS)
        request["model"] = target.api_model
        return request


class BedrockAnthropicAdapter(AnthropicAdapter):
    interface = "bedrock-anthropic"


class OpenAIAdapter:
    interface = "openai"

    def query_with_metrics(self, target: ResolvedModel, query: str) -> ModelCallResult:
        client = create_openai_client(target)
        started_at = time.perf_counter()
        request = openai_completion_kwargs(target.logical_model, query, MODEL_MAX_TOKENS)
        request["model"] = target.api_model
        response = client.chat.completions.create(**request)
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Risposta OpenAI priva di contenuto testuale.")
        elapsed = latency_ms(started_at)
        return ModelCallResult(
            text=content.strip(),
            metrics=build_model_call_metrics(
                provider=target.namespace,
                model=target.logical_model,
                latency_ms_value=elapsed,
                usage=extract_usage(response),
                namespace=target.namespace,
                latency_ms_for_hourly=elapsed,
            ),
        )

    def request_kwargs_for_summary(self, target: ResolvedModel) -> dict[str, object]:
        request = openai_completion_kwargs(target.logical_model, "", MODEL_MAX_TOKENS)
        request["model"] = target.api_model
        return request


class BedrockOpenAIAdapter(OpenAIAdapter):
    interface = "bedrock-openai"


_ADAPTER_REGISTRY: dict[str, ModelProviderAdapter] = {
    "anthropic": AnthropicAdapter(),
    "openai": OpenAIAdapter(),
    "bedrock-anthropic": BedrockAnthropicAdapter(),
    "bedrock-openai": BedrockOpenAIAdapter(),
}


def get_model_adapter(target: ResolvedModel) -> ModelProviderAdapter:
    adapter = _ADAPTER_REGISTRY.get(target.interface)
    if adapter is None:
        raise RuntimeError(f"Interfaccia provider non supportata: {target.interface}")
    return adapter


def query_model_with_metrics(model: str, query: str) -> ModelCallResult:
    target = resolve_model(model)
    adapter = get_model_adapter(target)
    return adapter.query_with_metrics(target, query)


def query_model(model: str, query: str) -> str:
    return query_model_with_metrics(model, query).text


def model_request_kwargs_for_summary(model: str) -> dict[str, object]:
    target = resolve_model(model)
    adapter = get_model_adapter(target)
    return adapter.request_kwargs_for_summary(target)
