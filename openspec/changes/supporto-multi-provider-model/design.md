## Context

See `proposal.md` for motivation. Current runtime behavior is spread across `provider_runtime.py`, `model_query.py`, judge implementations, and benchmark runners. Provider identity is partly inferred from model name prefixes and Bedrock support is mixed with provider logic, while pricing is loaded from JSON environment variables in `usage_tracking.py`. The change needs a single deterministic configuration source that supports cloud and self-hosted namespaces consistently.

## Goals / Non-Goals

**Goals:**
- Establish a single provider registry (`.env.providers.yaml`) that controls namespace resolution, adapter interface dispatch, credentials, endpoint options, and pricing.
- Enforce `namespace:model` in runtime config and CLI entry points.
- Preserve existing output field shape (`provider`, `model`, `model_call_estimated_cost_usd`) while widening provider typing where needed.
- Remove prefix-based provider inference, global backend toggles, and environment-JSON pricing source.
- Keep retry semantics while improving namespace-aware error context.

**Non-Goals:**
- Introduce model discovery or model availability catalogs.
- Add endpoint preflight checks or health probes.
- Migrate historical stored result files.
- Introduce EUR conversion or multi-currency logic.

## Decisions

### 1) Registry schema and loader
- Add a typed loader for `.env.providers.yaml` in runtime code path.
- Registry root is a map keyed by namespace; each namespace includes:
  - `interface`: `anthropic | openai | bedrock-anthropic | bedrock-openai` (required)
  - optional endpoint fields: `base_url`, `api_key_env`, `prefix`, `region`, `timeout_s`, `verify_tls`, `max_retries`
  - optional `pricing` block (see Decision 4)
- Validation occurs before first call; failures include file path, namespace, and invalid field.
- Provide test hook to reset/reload registry so tests can isolate scenarios.
- `transport` is removed from the registry schema and replaced by explicit interface variants.

**Alternatives considered:**
- Keep built-in Python defaults for anthropic/openai: rejected because it reintroduces split behavior and hidden precedence.
- Keep `LLM_BACKEND` with registry overlays: rejected because target behavior should be namespace-local, not global backend-driven.
- Keep `interface + transport`: rejected because OpenAI Bedrock behavior is not just transport wiring; it has distinct call/parsing semantics better represented as an explicit interface variant.

### 2) Namespace target contract
- All configurable model targets become `namespace:model`; parsing splits on first colon.
- Runtime resolves namespace entry first, then passes model segment unchanged through adapter path.
- Non-namespaced targets are rejected as configuration errors.

**Alternatives considered:**
- Keep legacy bare models and infer provider from prefix: rejected due to collisions and self-hosted ambiguity.
- Use two flags (`provider` + `model`) at every call site: rejected because it duplicates config and drifts from registry-based resolution.

### 3) Adapter dispatch model
- Dispatch call behavior by one adapter interface enum value, not by model naming convention.
- Introduce a shared Python adapter interface (Protocol or equivalent abstract contract) implemented by concrete adapter classes:
  - `anthropic`
  - `openai`
  - `bedrock-anthropic`
  - `bedrock-openai`
- Caller modules (`model_query`, judge flows, benchmark flows) call adapters through this common method contract instead of provider-specific branches.
- Bedrock API model ID is computed as `prefix + model`; no hardcoded model map.
- `region` is mandatory for bedrock interface variants (`bedrock-anthropic`, `bedrock-openai`).

**Alternatives considered:**
- Treat bedrock as a generic transport toggle under two interfaces: rejected because current openai-bedrock path has behavior differences that become cleaner and less error-prone as a first-class adapter implementation.

### 4) Pricing model integrated in registry
- Remove env JSON pricing inputs and source pricing from namespace entries only.
- `pricing` is optional. If present:
  - `mode: token | hourly` (required)
  - `models` map (required)
- `mode: token`:
  - model entries may include `input_per_1m`, `output_per_1m`, `cached_input_per_1m`.
  - `models.default` is forbidden.
- `mode: hourly`:
  - model entries may include only `cost_per_hour`.
  - `models.default` is allowed as fallback for undeclared models.
- Lookup order:
  1. exact model key
  2. `models.default` only for hourly mode
  3. no match => cost `None`
- Cost remains stored in USD-named fields.

**Alternatives considered:**
- Maintain env var override for quick edits: rejected by decision to keep a single source of truth.
- Allow mixed token/hourly fields in same provider: rejected because semantics become ambiguous and validation weak.

### 5) Config and schema compatibility
- Keep persisted output shape unchanged, but widen strict provider literals where needed to `str`.
- `config.py` judge runtime config stops using `JUDGE_[A|B|C]_PROVIDER`; each judge only reads namespaced `JUDGE_[A|B|C]_MODEL`.
- CLI docs/help examples in runners update to namespaced models.
- Public repository docs (`README.md`, `docs/CITATION_GROUNDING.md`) are updated to describe registry-driven provider selection and namespaced model targets, and to remove obsolete `LLM_BACKEND` / provider-routing instructions.

**Alternatives considered:**
- Rename persisted cost fields to EUR or neutral names: rejected now to avoid mixed historical semantics and migration burden.

## Risks / Trade-offs

- [Breaking input syntax] Existing `.env` and CLI values without namespace fail. -> Mitigation: update `.env.example`, add clear validation errors, and include migration notes in proposal/tasks.
- [Registry misconfiguration] A typo can block all LLM calls. -> Mitigation: fail-fast validation with precise namespace/field errors and dedicated tests.
- [Bedrock composition mistakes] Wrong `prefix`/`region` can silently call wrong endpoint. -> Mitigation: strict bedrock validation plus tests asserting one-time prefix composition and region requirement.
- [Pricing completeness] Missing model entries yield `None` cost. -> Mitigation: explicit behavior in spec and reporting, plus optional hourly `default` for self-hosted fleets.
- [Adapter abstraction quality] A weak adapter contract can preserve hidden branching in callers. -> Mitigation: enforce one shared adapter method contract and integration tests that run all four interface variants through identical caller paths.

## Migration Plan

1. Add registry loader/types and validation, plus test fixtures and sample `.env.providers.example.yaml`.
2. Refactor provider resolution and client creation to consume namespaced target + registry.
3. Introduce adapter interface contract and concrete adapter implementations for the four interface variants.
4. Migrate shared query/adapters and judge/benchmark call sites to unified adapter dispatch.
5. Move pricing lookup to registry-backed mode and remove env JSON pricing path.
6. Update config parsing and CLI docs/examples to namespaced targets only.
7. Expand tests for routing, bedrock behavior, pricing modes, adapter contract behavior, and compatibility reads.
8. Remove dead legacy code paths and env variable references after tests pass.

## Open Questions

None.
