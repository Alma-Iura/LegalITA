"""
Configurazione globale del benchmark.
Tutte le costanti condivise tra i moduli stanno qui.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from provider_runtime import parse_model_target

from taxonomy import (
    CANONICAL_MACRO_AREAS,
    MACRO_AREA_LABELS,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths - tutti relativi alla root del progetto
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PRIVATE_DIR = DATA_DIR / "private"
TASKS_DIR = ROOT_DIR / "tasks"
RESULTS_DIR = ROOT_DIR / "results"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# Current benchmark/gold defaults.
BULLSHIT_GOLD_PATH = PRIVATE_DIR / "bullshit_tasks_v3_missing_documents_40.json"

CORPUS_ZIP = RAW_DIR / "sentenze.zip"
CORPUS_JSONL = PROCESSED_DIR / "corpus.jsonl"

# ---------------------------------------------------------------------------
# Corpus e tassonomia
# ---------------------------------------------------------------------------

MACRO_AREE: dict[str, str] = MACRO_AREA_LABELS
MACRO_AREE_CANONICHE: tuple[str, ...] = CANONICAL_MACRO_AREAS

EXCLUDED_DIVISIONS: set[str] = {"Sez. 7"}

# ---------------------------------------------------------------------------
# Benchmark - parametri di generazione
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42
MAX_SOURCE_PRINCIPLES: int = 5
MAX_CRITERIA: int = 4
MIN_FACTS_LENGTH: int = 200
MIN_PRINCIPLES: int = 1

# ---------------------------------------------------------------------------
# Evaluation - parametri del judge e del criteria builder
# ---------------------------------------------------------------------------

JUDGE_TEMPERATURE: float = 0.0
JUDGE_MAX_TOKENS: int = int(os.environ.get("JUDGE_MAX_TOKENS", "3000"))
JUDGE_RETRIES: int = 3

GENERATOR_MODEL: str = "anthropic:claude-sonnet-4-6"
JUDGE_MODEL: str = "anthropic:claude-sonnet-4-6"

JudgeStrategy = Literal["single", "adaptive_majority"]
JudgeId = Literal["A", "B", "C"]

SUPPORTED_JUDGE_STRATEGIES: tuple[str, ...] = ("single", "adaptive_majority")
DEFAULT_JUDGE_STRATEGY: JudgeStrategy = "adaptive_majority"
DEFAULT_JUDGE_A_MODEL: str = JUDGE_MODEL
DEFAULT_JUDGE_B_MODEL: str = "openai:gpt-5.5"
DEFAULT_JUDGE_C_MODEL: str = "anthropic:claude-opus-4-8"


@dataclass(frozen=True)
class JudgeEndpointConfig:
    judge_id: JudgeId
    provider: str
    model: str


@dataclass(frozen=True)
class JudgeRuntimeConfig:
    strategy: JudgeStrategy
    judge_a: JudgeEndpointConfig
    judge_b: JudgeEndpointConfig
    judge_c: JudgeEndpointConfig


def _env_value(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _provider_from_model(value: str) -> str:
    namespace, _ = parse_model_target(value)
    return namespace


def _build_endpoint(judge_id: JudgeId, model: str) -> JudgeEndpointConfig:
    clean_model = model.strip()
    provider = _provider_from_model(clean_model)
    return JudgeEndpointConfig(judge_id=judge_id, provider=provider, model=clean_model)


def build_judge_runtime_config(
    *,
    judge_strategy: str | None = None,
    judge_a_model: str | None = None,
    judge_b_model: str | None = None,
    judge_c_model: str | None = None,
    legacy_judge_model: str | None = None,
) -> JudgeRuntimeConfig:
    strategy = (
        judge_strategy
        or _env_value("JUDGE_STRATEGY")
        or DEFAULT_JUDGE_STRATEGY
    ).strip().lower()

    if any(
        value is not None and value.strip()
        for value in (
            _env_value("JUDGE_A_PROVIDER"),
            _env_value("JUDGE_B_PROVIDER"),
            _env_value("JUDGE_C_PROVIDER"),
        )
    ):
        raise RuntimeError(
            "Configurazione judge non valida: le variabili JUDGE_[A|B|C]_PROVIDER sono rimosse; "
            "usa solo JUDGE_[A|B|C]_MODEL in formato namespace:model"
        )

    a_model = (
        judge_a_model
        or _env_value("JUDGE_A_MODEL")
        or legacy_judge_model
        or DEFAULT_JUDGE_A_MODEL
    )
    b_model = judge_b_model or _env_value("JUDGE_B_MODEL") or DEFAULT_JUDGE_B_MODEL
    c_model = judge_c_model or _env_value("JUDGE_C_MODEL") or DEFAULT_JUDGE_C_MODEL

    return JudgeRuntimeConfig(
        strategy=strategy,  # type: ignore[arg-type]
        judge_a=_build_endpoint("A", a_model),
        judge_b=_build_endpoint("B", b_model),
        judge_c=_build_endpoint("C", c_model),
    )


def validate_judge_runtime_config(config: JudgeRuntimeConfig) -> None:
    if config.strategy not in SUPPORTED_JUDGE_STRATEGIES:
        raise RuntimeError(
            "Strategia judge non supportata: "
            f"{config.strategy!r}. Valori ammessi: {', '.join(SUPPORTED_JUDGE_STRATEGIES)}"
        )

    endpoints = [config.judge_a]
    if config.strategy == "adaptive_majority":
        endpoints.extend([config.judge_b, config.judge_c])

    errors: list[str] = []

    for endpoint in endpoints:
        if not endpoint.model:
            errors.append(f"JUDGE_{endpoint.judge_id}_MODEL mancante")
            continue
        try:
            parse_model_target(endpoint.model)
        except RuntimeError as exc:
            errors.append(f"Judge {endpoint.judge_id}: {exc}")

    if errors:
        raise RuntimeError("Configurazione judge non valida: " + "; ".join(errors))

    if (
        config.strategy == "adaptive_majority"
        and config.judge_a.model == config.judge_b.model
    ):
        log.warning(
            "Judge A e Judge B usano lo stesso modello (%s): "
            "la maggioranza non e metodologicamente eterogenea.",
            config.judge_a.model,
        )


BUILDER_TEMPERATURE: float = 0.0

# ---------------------------------------------------------------------------
# Structured court-rulings S3 resolver.
# ---------------------------------------------------------------------------

COURT_RULINGS_S3_BUCKET: str = os.environ.get(
    "COURT_RULINGS_S3_BUCKET",
    "",
).strip()
COURT_RULINGS_S3_PREFIX: str = os.environ.get(
    "COURT_RULINGS_S3_PREFIX",
    "",
).strip()

# ---------------------------------------------------------------------------
# Run - parametri dei modelli sotto esame
# ---------------------------------------------------------------------------

MODEL_MAX_TOKENS: int = int(os.environ.get("MODEL_MAX_TOKENS", "16000"))
MODEL_RETRIES: int = 3
