"""Provider/backend resolution and client factories.

LegalITA keeps provider semantics (``anthropic`` / ``openai``) separate from
transport. ``LLM_BACKEND=native`` uses the first-party APIs exactly as before;
``LLM_BACKEND=bedrock`` maps logical model names to Amazon Bedrock model IDs
and creates Bedrock-compatible clients.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import anthropic
from openai import OpenAI

LLMBackend = Literal["native", "bedrock"]
ProviderName = Literal["anthropic", "openai"]
TransportName = Literal["anthropic_native", "bedrock_runtime", "openai_native", "bedrock_mantle"]

SUPPORTED_LLM_BACKENDS: tuple[str, ...] = ("native", "bedrock")
DEFAULT_BEDROCK_REGION = "us-east-2"
DEFAULT_BEDROCK_ANTHROPIC_SCOPE = "us"
SUPPORTED_BEDROCK_ANTHROPIC_SCOPES: tuple[str, ...] = ("global", "us", "eu", "au", "jp")

# Logical LegalITA model -> unqualified Bedrock provider model ID.
# The regional/global inference prefix for Anthropic is applied separately.
_BEDROCK_MODEL_IDS: dict[tuple[str, str], str] = {
    ("anthropic", "claude-sonnet-4-6"): "anthropic.claude-sonnet-4-6",
    ("anthropic", "claude-opus-4-8"): "anthropic.claude-opus-4-8",
    ("openai", "gpt-5.5"): "openai.gpt-5.5",
}


@dataclass(frozen=True)
class ResolvedModel:
    provider: ProviderName
    logical_model: str
    api_model: str
    backend: LLMBackend
    transport: TransportName
    region: str | None = None
    base_url: str | None = None


def get_llm_backend() -> LLMBackend:
    value = os.environ.get("LLM_BACKEND", "native").strip().lower()
    if value not in SUPPORTED_LLM_BACKENDS:
        raise RuntimeError(
            f"LLM_BACKEND={value!r} non supportato. "
            f"Valori ammessi: {', '.join(SUPPORTED_LLM_BACKENDS)}"
        )
    return value  # type: ignore[return-value]


def get_bedrock_region() -> str:
    return (
        os.environ.get("BEDROCK_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or DEFAULT_BEDROCK_REGION
    ).strip()


def get_bedrock_anthropic_scope() -> str:
    scope = os.environ.get(
        "BEDROCK_ANTHROPIC_SCOPE",
        DEFAULT_BEDROCK_ANTHROPIC_SCOPE,
    ).strip().lower()
    if scope not in SUPPORTED_BEDROCK_ANTHROPIC_SCOPES:
        raise RuntimeError(
            f"BEDROCK_ANTHROPIC_SCOPE={scope!r} non supportato. "
            f"Valori ammessi: {', '.join(SUPPORTED_BEDROCK_ANTHROPIC_SCOPES)}"
        )
    return scope


def bedrock_openai_base_url(region: str | None = None) -> str:
    override = os.environ.get("BEDROCK_OPENAI_BASE_URL", "").strip()
    if override:
        return override.rstrip("/")
    resolved_region = region or get_bedrock_region()
    return f"https://bedrock-mantle.{resolved_region}.api.aws/openai/v1"


def bedrock_api_key() -> str:
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip()
    if not token:
        raise RuntimeError(
            "AWS_BEARER_TOKEN_BEDROCK non impostata per LLM_BACKEND=bedrock"
        )
    return token


def supported_bedrock_models() -> tuple[tuple[str, str], ...]:
    return tuple(sorted(_BEDROCK_MODEL_IDS))


def is_bedrock_model_supported(provider: str, model: str) -> bool:
    return (provider.strip().lower(), model.strip()) in _BEDROCK_MODEL_IDS


def resolve_model(
    provider: ProviderName,
    model: str,
    *,
    backend: LLMBackend | None = None,
) -> ResolvedModel:
    logical_model = model.strip()
    resolved_backend = backend or get_llm_backend()

    if resolved_backend == "native":
        transport: TransportName = (
            "anthropic_native" if provider == "anthropic" else "openai_native"
        )
        return ResolvedModel(
            provider=provider,
            logical_model=logical_model,
            api_model=logical_model,
            backend="native",
            transport=transport,
        )

    key = (provider, logical_model)
    api_model = _BEDROCK_MODEL_IDS.get(key)
    if api_model is None:
        supported = ", ".join(
            f"{item_provider}/{item_model}"
            for item_provider, item_model in supported_bedrock_models()
        )
        raise RuntimeError(
            f"Modello {provider}/{logical_model} non mappato su Amazon Bedrock. "
            f"Mapping inclusi: {supported}"
        )

    region = get_bedrock_region()
    if provider == "anthropic":
        scope = get_bedrock_anthropic_scope()
        return ResolvedModel(
            provider=provider,
            logical_model=logical_model,
            api_model=f"{scope}.{api_model}",
            backend="bedrock",
            transport="bedrock_runtime",
            region=region,
        )

    return ResolvedModel(
        provider=provider,
        logical_model=logical_model,
        api_model=api_model,
        backend="bedrock",
        transport="bedrock_mantle",
        region=region,
        base_url=bedrock_openai_base_url(region),
    )


def create_anthropic_client(target: ResolvedModel) -> Any:
    if target.provider != "anthropic":
        raise ValueError("Target non Anthropic passato a create_anthropic_client")
    if target.backend == "native":
        return anthropic.Anthropic()

    # AnthropicBedrock accepts a Bedrock bearer API key directly. Read the
    # shared token explicitly so both Anthropic and OpenAI Bedrock clients use
    # the same credential and missing configuration fails before a request.
    api_key = bedrock_api_key()
    bedrock_cls = getattr(anthropic, "AnthropicBedrock", None)
    if bedrock_cls is None:
        raise RuntimeError(
            "AnthropicBedrock non disponibile. Installa le dipendenze con "
            "pip install -e . (anthropic[bedrock])."
        )
    return bedrock_cls(
        api_key=api_key,
        aws_region=target.region or get_bedrock_region(),
    )


def create_openai_client(target: ResolvedModel) -> OpenAI:
    if target.provider != "openai":
        raise ValueError("Target non OpenAI passato a create_openai_client")
    if target.backend == "native":
        return OpenAI()

    return OpenAI(
        api_key=bedrock_api_key(),
        base_url=target.base_url or bedrock_openai_base_url(target.region),
    )
