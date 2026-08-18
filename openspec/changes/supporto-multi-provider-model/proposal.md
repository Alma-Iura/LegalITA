## Why

LegalITA currently routes LLM providers through duplicated, partially hardcoded logic that infers provider identity from model-name prefixes and treats Bedrock behavior as scattered special branches. This blocks reliable support for arbitrary self-hosted endpoints and makes pricing, retry, and credential behavior inconsistent across judge and benchmark flows.

## What Changes

- Introduce a unified provider registry file (`.env.providers.yaml`) that declares endpoint topology, adapter interface type, credential source, and optional pricing for each namespace.
- Require namespaced model selection (`namespace:model`) across judge and benchmark configuration and CLI inputs.
- Replace prefix-based provider inference and hardcoded Bedrock model mapping with runtime resolution from the registry.
- Replace `interface + transport` pairing with a single explicit interface enum:
  - `anthropic`
  - `openai`
  - `bedrock-anthropic`
  - `bedrock-openai`
- Standardize runtime behavior through provider adapter classes implementing one shared Python adapter interface, so callers depend on a uniform method contract.
- Move pricing source of truth from environment JSON blobs to per-provider `pricing` declarations in the registry.
- Support two explicit pricing modes when pricing is configured:
  - `token`: model-specific token pricing only.
  - `hourly`: self-hosted-style cost per hour with optional `default` fallback in `pricing.models`.
- Keep persisted result schema shape stable (`provider` and `model` remain separate fields; cost stays USD-named fields).
- **BREAKING**: Remove non-namespaced provider/model routing, remove legacy provider-specific judge env vars, and remove legacy global backend/transport env routing.

## Capabilities

### New Capabilities
- `llm-provider-registry`: Registry-driven model routing, adapter selection, credential resolution, and pricing for all LegalITA LLM call paths.

### Modified Capabilities
- None.

## Impact

- Affected code: `provider_runtime.py`, `model_query.py`, `model_runtime.py`, `model_request_config.py`, `usage_tracking.py`, `config.py`, judge and benchmark execution modules, and related tests.
- Configuration impact: introduces `.env.providers.yaml` as required runtime input and deprecates legacy routing env knobs.
- Dependency impact: YAML parsing dependency required for loading the provider registry.
- Operational impact: model selection and judge endpoint config become explicitly namespaced and deterministic across cloud and self-hosted targets.
- Documentation impact: public docs (`README.md`, `docs/CITATION_GROUNDING.md`) must be aligned to namespaced model syntax and registry-driven configuration, removing legacy backend/provider routing guidance.
