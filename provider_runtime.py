"""Namespaced provider registry resolution and client factories."""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Literal

import yaml

import anthropic
from openai import OpenAI

ProviderInterface = Literal[
    "anthropic",
    "openai",
    "bedrock-anthropic",
    "bedrock-openai",
]
SUPPORTED_PROVIDER_INTERFACES: tuple[str, ...] = (
    "anthropic",
    "openai",
    "bedrock-anthropic",
    "bedrock-openai",
)
_REGISTRY_ENV_VAR = "LLM_PROVIDER_REGISTRY"
_REGISTRY_FILENAME = ".env.providers.yaml"
_REGISTRY_CACHE: dict[str, "RegistryNamespace"] | None = None
_REGISTRY_CACHE_PATH: Path | None = None


@dataclass(frozen=True)
class RegistryNamespace:
    namespace: str
    interface: ProviderInterface
    base_url: str | None = None
    api_key_env: str | None = None
    prefix: str | None = None
    region: str | None = None
    timeout_s: float | None = None
    verify_tls: bool | None = None
    max_retries: int | None = None
    pricing: dict[str, Any] | None = None


@dataclass(frozen=True)
class ResolvedModel:
    provider: str
    namespace: str
    logical_model: str
    api_model: str
    interface: ProviderInterface
    region: str | None = None
    base_url: str | None = None
    timeout_s: float | None = None
    verify_tls: bool | None = None
    max_retries: int | None = None
    api_key_env: str | None = None
    api_key: str | None = None


def _registry_path() -> Path:
    custom_path = os.environ.get(_REGISTRY_ENV_VAR, "").strip()
    if custom_path:
        return Path(custom_path).expanduser().resolve()
    return (Path(__file__).parent / _REGISTRY_FILENAME).resolve()


def reset_provider_registry_cache() -> None:
    global _REGISTRY_CACHE, _REGISTRY_CACHE_PATH
    _REGISTRY_CACHE = None
    _REGISTRY_CACHE_PATH = None


def parse_model_target(target: str) -> tuple[str, str]:
    value = target.strip()
    if not value or ":" not in value:
        raise RuntimeError(
            f"Target modello non valido {target!r}: atteso formato 'namespace:model'"
        )
    namespace, logical_model = value.split(":", 1)
    namespace = namespace.strip()
    logical_model = logical_model.strip()
    if not namespace or not logical_model:
        raise RuntimeError(
            f"Target modello non valido {target!r}: namespace e model sono obbligatori"
        )
    return namespace, logical_model


def bedrock_openai_base_url(region: str) -> str:
    resolved_region = region.strip()
    return f"https://bedrock-mantle.{resolved_region}.api.aws/openai/v1"


def _error(path: Path, namespace: str, field: str, message: str) -> RuntimeError:
    return RuntimeError(f"{path} [{namespace}] {field}: {message}")


def _as_optional_str(path: Path, namespace: str, field: str, value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _error(path, namespace, field, "attesa stringa")
    stripped = value.strip()
    if stripped == "":
        raise _error(path, namespace, field, "stringa vuota non ammessa")
    return stripped


def _as_optional_bool(path: Path, namespace: str, field: str, value: Any) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise _error(path, namespace, field, "atteso boolean")
    return value


def _as_optional_float(path: Path, namespace: str, field: str, value: Any) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise _error(path, namespace, field, "atteso numero")
    out = float(value)
    if out <= 0:
        raise _error(path, namespace, field, "deve essere > 0")
    return out


def _as_optional_int(path: Path, namespace: str, field: str, value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise _error(path, namespace, field, "atteso intero")
    if value < 0:
        raise _error(path, namespace, field, "deve essere >= 0")
    return value


def _validate_pricing(path: Path, namespace: str, pricing: Any) -> dict[str, Any] | None:
    if pricing is None:
        return None
    if not isinstance(pricing, dict):
        raise _error(path, namespace, "pricing", "atteso oggetto")

    mode = pricing.get("mode")
    models = pricing.get("models")
    if mode not in ("token", "hourly"):
        raise _error(path, namespace, "pricing.mode", "valori ammessi: token, hourly")
    if not isinstance(models, dict) or not models:
        raise _error(path, namespace, "pricing.models", "attesa mappa non vuota")

    if mode == "token" and "default" in models:
        raise _error(path, namespace, "pricing.models.default", "non ammesso in mode=token")

    for model_key, model_pricing in models.items():
        if not isinstance(model_key, str) or not model_key.strip():
            raise _error(path, namespace, "pricing.models", "chiave modello non valida")
        if not isinstance(model_pricing, dict):
            raise _error(path, namespace, f"pricing.models.{model_key}", "atteso oggetto")

        if mode == "token":
            allowed = {"input_per_1m", "output_per_1m", "cached_input_per_1m"}
            unknown = set(model_pricing) - allowed
            if unknown:
                raise _error(
                    path,
                    namespace,
                    f"pricing.models.{model_key}",
                    f"campi non ammessi in mode=token: {', '.join(sorted(unknown))}",
                )
            for field in allowed:
                value = model_pricing.get(field)
                if value is None:
                    continue
                if not isinstance(value, (int, float)):
                    raise _error(
                        path,
                        namespace,
                        f"pricing.models.{model_key}.{field}",
                        "atteso numero",
                    )
        else:
            allowed = {"cost_per_hour"}
            unknown = set(model_pricing) - allowed
            if unknown:
                raise _error(
                    path,
                    namespace,
                    f"pricing.models.{model_key}",
                    f"campi non ammessi in mode=hourly: {', '.join(sorted(unknown))}",
                )
            value = model_pricing.get("cost_per_hour")
            if value is None or not isinstance(value, (int, float)):
                raise _error(
                    path,
                    namespace,
                    f"pricing.models.{model_key}.cost_per_hour",
                    "atteso numero",
                )

    return pricing


def _parse_namespace(path: Path, namespace: str, raw_entry: Any) -> RegistryNamespace:
    if not isinstance(raw_entry, dict):
        raise _error(path, namespace, "entry", "atteso oggetto")

    interface = raw_entry.get("interface")
    if interface not in SUPPORTED_PROVIDER_INTERFACES:
        raise _error(
            path,
            namespace,
            "interface",
            f"valori ammessi: {', '.join(SUPPORTED_PROVIDER_INTERFACES)}",
        )

    parsed = RegistryNamespace(
        namespace=namespace,
        interface=interface,
        base_url=_as_optional_str(path, namespace, "base_url", raw_entry.get("base_url")),
        api_key_env=_as_optional_str(path, namespace, "api_key_env", raw_entry.get("api_key_env")),
        prefix=_as_optional_str(path, namespace, "prefix", raw_entry.get("prefix")),
        region=_as_optional_str(path, namespace, "region", raw_entry.get("region")),
        timeout_s=_as_optional_float(path, namespace, "timeout_s", raw_entry.get("timeout_s")),
        verify_tls=_as_optional_bool(path, namespace, "verify_tls", raw_entry.get("verify_tls")),
        max_retries=_as_optional_int(path, namespace, "max_retries", raw_entry.get("max_retries")),
        pricing=_validate_pricing(path, namespace, raw_entry.get("pricing")),
    )

    if parsed.interface in ("bedrock-anthropic", "bedrock-openai"):
        if not parsed.region:
            raise _error(path, namespace, "region", "obbligatorio per interfacce bedrock")

    return parsed


def load_provider_registry() -> dict[str, RegistryNamespace]:
    global _REGISTRY_CACHE, _REGISTRY_CACHE_PATH
    path = _registry_path()
    if _REGISTRY_CACHE is not None and _REGISTRY_CACHE_PATH == path:
        return _REGISTRY_CACHE

    if not path.exists():
        raise RuntimeError(f"Registry provider non trovato: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Errore parsing YAML registry provider {path}: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise RuntimeError(f"Registry provider non valido {path}: attesa mappa namespace->config")

    parsed: dict[str, RegistryNamespace] = {}
    for namespace, raw_entry in raw.items():
        if not isinstance(namespace, str) or not namespace.strip():
            raise RuntimeError(f"{path}: namespace non valido {namespace!r}")
        key = namespace.strip()
        parsed[key] = _parse_namespace(path, key, raw_entry)

    _REGISTRY_CACHE = parsed
    _REGISTRY_CACHE_PATH = path
    return parsed


def get_namespace_pricing(namespace: str) -> dict[str, Any] | None:
    entry = load_provider_registry().get(namespace)
    return entry.pricing if entry is not None else None


def resolve_model(target: str) -> ResolvedModel:
    namespace, logical_model = parse_model_target(target)
    path = _registry_path()
    registry = load_provider_registry()
    entry = registry.get(namespace)
    if entry is None:
        raise RuntimeError(f"Namespace provider sconosciuto {namespace!r} nel registry {path}")

    api_model = logical_model
    if entry.interface in ("bedrock-anthropic", "bedrock-openai"):
        prefix = entry.prefix or ""
        if prefix and logical_model.startswith(prefix):
            api_model = logical_model
        else:
            api_model = f"{prefix}{logical_model}" if prefix else logical_model

    resolved_base_url = entry.base_url
    if entry.interface == "bedrock-openai" and not resolved_base_url:
        resolved_base_url = bedrock_openai_base_url(entry.region or "")

    api_key: str | None = None
    if entry.api_key_env:
        value = os.environ.get(entry.api_key_env, "").strip()
        if not value:
            raise RuntimeError(
                f"Credentiale mancante per namespace {namespace!r}: "
                f"variabile ambiente {entry.api_key_env} non impostata"
            )
        api_key = value

    return ResolvedModel(
        provider=namespace,
        namespace=namespace,
        logical_model=logical_model,
        api_model=api_model,
        interface=entry.interface,
        region=entry.region,
        base_url=resolved_base_url,
        timeout_s=entry.timeout_s,
        verify_tls=entry.verify_tls,
        max_retries=entry.max_retries,
        api_key_env=entry.api_key_env,
        api_key=api_key,
    )


def create_anthropic_client(target: ResolvedModel) -> Any:
    if target.interface not in ("anthropic", "bedrock-anthropic"):
        raise ValueError(
            "Interfaccia non Anthropic passata a create_anthropic_client: "
            f"{target.interface}"
        )

    if target.interface == "anthropic":
        kwargs: dict[str, Any] = {}
        if target.api_key:
            kwargs["api_key"] = target.api_key
        if target.base_url:
            kwargs["base_url"] = target.base_url
        if target.timeout_s is not None:
            kwargs["timeout"] = target.timeout_s
        if target.max_retries is not None:
            kwargs["max_retries"] = target.max_retries
        return anthropic.Anthropic(**kwargs)

    bedrock_cls = getattr(anthropic, "AnthropicBedrock", None)
    if bedrock_cls is None:
        raise RuntimeError(
            "AnthropicBedrock non disponibile. Installa le dipendenze con "
            "pip install -e . (anthropic[bedrock])."
        )

    if not target.api_key:
        missing = target.api_key_env or "<api_key_env>"
        raise RuntimeError(
            f"Credenziale Bedrock Anthropic mancante per namespace {target.namespace!r}: {missing}"
        )
    if not target.region:
        raise RuntimeError(
            f"Region mancante per namespace bedrock Anthropic {target.namespace!r}"
        )

    kwargs = {
        "api_key": target.api_key,
        "aws_region": target.region,
    }
    if target.timeout_s is not None:
        kwargs["timeout"] = target.timeout_s
    if target.max_retries is not None:
        kwargs["max_retries"] = target.max_retries
    return bedrock_cls(**kwargs)


def create_openai_client(target: ResolvedModel) -> OpenAI:
    if target.interface not in ("openai", "bedrock-openai"):
        raise ValueError(
            f"Interfaccia non OpenAI passata a create_openai_client: {target.interface}"
        )

    kwargs: dict[str, Any] = {}
    if target.api_key:
        kwargs["api_key"] = target.api_key
    if target.base_url:
        kwargs["base_url"] = target.base_url
    if target.timeout_s is not None:
        kwargs["timeout"] = target.timeout_s
    if target.max_retries is not None:
        kwargs["max_retries"] = target.max_retries
    return OpenAI(**kwargs)
