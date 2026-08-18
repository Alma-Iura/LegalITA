## 1. Registry foundation

- [ ] 1.1 Add YAML loading dependency and implement typed registry loader for `.env.providers.yaml` with cache reset support for tests.
- [ ] 1.2 Implement schema validation for namespace entries (`interface` enum, endpoint options, pricing block) with fail-fast error messages that include file path, namespace, and field.
- [ ] 1.3 Enforce bedrock-specific validation for `bedrock-anthropic` and `bedrock-openai` (`region` required, `prefix` handling) and explicit credential env resolution via `api_key_env`.

## 2. Runtime target resolution and adapter dispatch

- [ ] 2.1 Replace legacy provider/backend resolution with `namespace:model` parsing (split on first colon) and namespace lookup from registry.
- [ ] 2.2 Refactor resolved-target data model to carry namespace provider, logical model, API model, interface, region, and endpoint options.
- [ ] 2.3 Introduce a shared Python adapter interface (Protocol/abstract contract) that defines the common methods required by callers.
- [ ] 2.4 Implement concrete adapters for `anthropic`, `openai`, `bedrock-anthropic`, and `bedrock-openai` and register them in a central adapter registry.

## 3. Judge and benchmark integration

- [ ] 3.1 Migrate `model_query.py` adapter entry points to consume resolved namespace targets and remove `NOVITA_PROVIDERS`/`GEMINI_PROVIDER_PREFIX`/`startswith("claude")` routing.
- [ ] 3.2 Update judge and bullshit judge flows to call providers through the shared adapter contract, removing duplicated provider-specific call scaffolding.
- [ ] 3.3 Update benchmark generator/task builder and runner scripts to accept namespaced model values in all model-selection paths and use unified adapter dispatch.

## 4. Configuration and pricing migration

- [ ] 4.1 Remove legacy judge provider env inputs (`JUDGE_A_PROVIDER`, `JUDGE_B_PROVIDER`, `JUDGE_C_PROVIDER`) and validate only namespaced `JUDGE_[A|B|C]_MODEL` values.
- [ ] 4.2 Replace env JSON pricing lookup with registry-driven pricing and implement discriminated `pricing.mode` behavior (`token` vs `hourly`).
- [ ] 4.3 Implement pricing lookup precedence: exact model key, then `models.default` only for hourly mode, else `None`; keep cost output fields in USD naming.

## 5. Examples, docs, and compatibility wiring

- [ ] 5.1 Update `.env.example` to reflect namespaced judge model configuration and remove obsolete backend/provider routing knobs.
- [ ] 5.2 Add `.env.providers.example.yaml` with representative `anthropic`, `openai`, `bedrock-anthropic`, `bedrock-openai`, and self-hosted hourly examples with no real secrets.
- [ ] 5.3 Widen strict provider literals in persisted-score schemas to `str` where required while preserving separate `provider` and `model` fields.

## 6. Test coverage and validation

- [ ] 6.1 Add/adjust unit tests for registry loader validation, namespace parsing, bedrock prefix+region behavior, adapter registry lookup, and credential failure modes.
- [ ] 6.2 Add/adjust tests for pricing modes, hourly default fallback, unknown pricing (`None`), and hourly latency-based cost calculation.
- [ ] 6.3 Update integration-oriented tests for judge and benchmark routing to namespaced inputs and verify all four interface variants run through the shared adapter contract with correct retry/context behavior.
- [ ] 6.4 Run full test suite and ensure OpenSpec change validates cleanly.
