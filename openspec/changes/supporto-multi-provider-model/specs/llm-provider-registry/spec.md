## Purpose

Define a single runtime contract for resolving provider namespace, model target, adapter interface, credentials, and pricing across all LegalITA LLM call flows, including cloud and self-hosted endpoints.

## ADDED Requirements

### Requirement: Namespaced model targets
The runtime MUST accept model targets only in `namespace:model` format for judge and benchmark execution paths.

#### Scenario: Parse model with additional colons
- **WHEN** the input target is `gpu-office:qwen2.5:72b`
- **THEN** the namespace is resolved as `gpu-office` and the model is resolved as `qwen2.5:72b`

#### Scenario: Reject non-namespaced target
- **WHEN** the input target is `claude-sonnet-4-6` without a namespace separator
- **THEN** the runtime fails before any provider call with a validation error that states the expected `namespace:model` format

### Requirement: Registry-driven namespace resolution
The runtime MUST resolve namespace behavior exclusively from `.env.providers.yaml` and MUST fail fast when the namespace is not declared.

#### Scenario: Unknown namespace
- **WHEN** a target uses namespace `unknown-lab` that is absent from the registry
- **THEN** the runtime fails before any network request with an error including namespace name and registry file path

#### Scenario: Registry parse error
- **WHEN** the registry file contains invalid YAML
- **THEN** the runtime fails startup validation with a readable parse error that references the registry file path

### Requirement: Interface dispatch
Each registry entry MUST declare `interface` as one of `anthropic`, `openai`, `bedrock-anthropic`, or `bedrock-openai`; the runtime MUST dispatch client and call behavior from that field.

#### Scenario: OpenAI-compatible direct dispatch
- **WHEN** a namespace declares `interface: openai`
- **THEN** the runtime uses OpenAI-compatible direct request and response handling for that namespace

#### Scenario: Bedrock OpenAI dispatch
- **WHEN** a namespace declares `interface: bedrock-openai`
- **THEN** the runtime uses the Bedrock OpenAI-specific request and response handling path for that namespace

### Requirement: Bedrock model ID composition
For `interface` values `bedrock-anthropic` and `bedrock-openai`, the runtime MUST compute API model IDs as `prefix + model` without hardcoded model catalog restrictions.

#### Scenario: Prefix applied once
- **WHEN** namespace `bedrock-anthropic` declares `prefix: us.anthropic.` and target model `claude-sonnet-4-6`
- **THEN** the API model ID used for calls is `us.anthropic.claude-sonnet-4-6`

#### Scenario: Missing Bedrock region
- **WHEN** a namespace with `interface: bedrock-anthropic` omits `region`
- **THEN** validation fails before provider calls with an error that points to the missing `region` field

### Requirement: Explicit credential and endpoint options
Registry credential and endpoint options MUST be applied explicitly, including `api_key_env`, `base_url`, `timeout_s`, `verify_tls`, and `max_retries` when provided.

#### Scenario: Missing credential env value
- **WHEN** a namespace references `api_key_env` and the environment variable is unset
- **THEN** the runtime fails before request dispatch without exposing secret values

#### Scenario: Optional endpoint overrides
- **WHEN** a namespace declares `base_url`, `timeout_s`, and `verify_tls: false`
- **THEN** the runtime configures client calls with those values for that namespace

### Requirement: Adapter contract consistency
All provider adapter implementations MUST expose one shared Python interface so callers use the same adapter method contract regardless of provider interface type.

#### Scenario: Unified caller contract
- **WHEN** a caller resolves any supported interface type
- **THEN** it can execute model calls through the same adapter method set without provider-specific branching in caller code

### Requirement: Retry context and precedence
Retry behavior MUST preserve existing transient error handling and include namespace context in failure messages; provider-level `max_retries` MUST override caller defaults when configured.

#### Scenario: Provider retry override
- **WHEN** caller default retries are 3 and namespace sets `max_retries: 1`
- **THEN** runtime executes at most one retry attempt for that namespace

#### Scenario: Error message context
- **WHEN** retries are exhausted for namespace `gpu-office`
- **THEN** the raised error includes namespace and target model context without leaking credential material

### Requirement: Pricing from provider registry
Cost estimation MUST read pricing from provider registry and MUST not read `MODEL_PRICING_USD_PER_1M` or `MODEL_PRICING_USD_PER_MILLION`.

#### Scenario: Token pricing lookup
- **WHEN** namespace pricing mode is `token` and model has explicit prices
- **THEN** cost is computed from input/output/cached token usage and returned in USD fields

#### Scenario: Hourly pricing default fallback
- **WHEN** namespace pricing mode is `hourly`, model has no explicit entry, and `pricing.models.default.cost_per_hour` exists
- **THEN** cost is computed from call latency and default hourly rate

#### Scenario: Unknown pricing
- **WHEN** pricing is absent or model has no applicable pricing entry
- **THEN** estimated cost remains `None`

### Requirement: Pricing mode constraints
When pricing is configured, the runtime MUST require `pricing.mode` as either `token` or `hourly` and MUST enforce field exclusivity per mode.

#### Scenario: Token mode rejects default fallback
- **WHEN** `pricing.mode` is `token` and `pricing.models.default` is present
- **THEN** configuration validation fails before runtime execution

#### Scenario: Hourly mode rejects token fields
- **WHEN** `pricing.mode` is `hourly` and a model entry includes `input_per_1m`
- **THEN** configuration validation fails with a field-level error

### Requirement: Persisted output compatibility
Runtime outputs MUST keep provider and model as separate persisted fields and MUST preserve USD-named cost fields.

#### Scenario: Stored provider/model shape
- **WHEN** a call is recorded for `gpu-office:qwen2.5:72b`
- **THEN** persisted output stores provider namespace in `provider` and logical model in `model` as separate fields

#### Scenario: Historical result readability
- **WHEN** historical records contain provider values such as `anthropic` or `openai`
- **THEN** score and report readers continue to parse those records without schema migration
