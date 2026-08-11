# Amazon Bedrock integration

This version of LegalITA adds an optional Amazon Bedrock transport while keeping
first-party Anthropic/OpenAI behavior as the default.

## Architecture

Provider identity and transport are deliberately separate:

- provider: `anthropic` or `openai` (unchanged benchmark semantics);
- backend: `native` or `bedrock` (`LLM_BACKEND`);
- logical model: the model name stored in LegalITA results, for example
  `claude-sonnet-4-6`;
- API model: the identifier sent to the selected endpoint, for example
  `us.anthropic.claude-sonnet-4-6`.

`provider_runtime.py` is the single place responsible for:

- backend selection;
- Bedrock region/scope resolution;
- logical-model to Bedrock-model mapping;
- Anthropic/OpenAI client creation;
- Bedrock OpenAI base URL construction.

The benchmark runners continue to route models by the original logical names,
so result files remain comparable with native-provider runs.

## Included model mappings

| Provider | Logical model | Bedrock API model | Transport |
|---|---|---|---|
| Anthropic | `claude-sonnet-4-6` | `<scope>.anthropic.claude-sonnet-4-6` | Bedrock Runtime via `AnthropicBedrock` |
| Anthropic | `claude-opus-4-8` | `<scope>.anthropic.claude-opus-4-8` | Bedrock Runtime via `AnthropicBedrock` |
| OpenAI | `gpt-5.5` | `openai.gpt-5.5` | Bedrock Mantle OpenAI-compatible API |

`<scope>` is controlled by `BEDROCK_ANTHROPIC_SCOPE` (`global`, `us`, `eu`,
`au`, or `jp`). AWS model availability varies by Region and inference profile.

## Configuration

### Native mode (backward-compatible default)

```text
LLM_BACKEND=native
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

### Bedrock mode

```text
LLM_BACKEND=bedrock
AWS_BEARER_TOKEN_BEDROCK=...
BEDROCK_REGION=us-east-2
BEDROCK_ANTHROPIC_SCOPE=us
```

The native `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` variables are not required
when `LLM_BACKEND=bedrock`.

The OpenAI-compatible URL defaults to:

```text
https://bedrock-mantle.<region>.api.aws/openai/v1
```

and can be overridden with `BEDROCK_OPENAI_BASE_URL` if AWS changes or a proxy
requires a different endpoint.

## Files changed

- `provider_runtime.py` - new backend/model/client runtime layer.
- `config.py` - backend-aware judge validation and credential checks.
- `model_query.py` - evaluated Anthropic/OpenAI models now use the runtime layer.
- `model_runtime.py` - Bedrock configuration errors are classified as
  non-retryable.
- `evaluation/judge.py` - judge clients/models resolved through the runtime
  layer; Bedrock OpenAI responses are JSON-parsed locally.
- `evaluation/bullshit_judge.py` - same Bedrock behavior for adversarial judges.
- `benchmark/generator.py` - generator Anthropic client is backend-aware.
- `benchmark/task_builder.py` - criteria-builder Anthropic client is
  backend-aware.
- `requirements.txt` / `pyproject.toml` - Bedrock SDK dependencies.
- `.env.example` - safe native/Bedrock configuration template.
- `.gitignore` - allows `.env.example` and delivered tests.
- `README.md` - Bedrock setup and runtime notes.
- `tests/` - focused backend/model/judge tests.

## OpenAI judge behavior on Bedrock

Native OpenAI mode preserves the existing `responses.parse(...,
text_format=PydanticModel)` path.

For Bedrock, LegalITA uses `responses.create()` and validates the JSON returned
by the model locally. This is intentional: the Bedrock Mantle endpoint does not
provide identical support for every OpenAI SDK structured-output helper. The
existing judge prompt already requires JSON-only output, and Pydantic validation
is still applied before a vote is accepted.

## Tests performed without live credentials

The delivered test suite covers:

- native backend leaves model IDs unchanged;
- Bedrock model mapping for all three required models;
- Anthropic regional/global scope mapping;
- Bedrock OpenAI endpoint construction;
- missing/unsupported Bedrock configuration failures;
- native credential validation remains unchanged;
- evaluated-model requests use Bedrock API IDs while metrics keep logical IDs;
- judge requests use Bedrock API IDs while result metadata keeps logical IDs;
- Bedrock OpenAI judge takes the non-`parse` Responses path.

Static Python compilation and import smoke checks completed successfully. All
14 focused unit tests passed. The tests/import smoke were executed using small
temporary SDK stubs because the build sandbox could not reach PyPI to install
`anthropic` and `openai`. Those stubs are not included in the delivered
project. A live Bedrock smoke test was not possible without credentials and
model access.

## Limitations

1. No live Amazon Bedrock inference can be executed without a valid Bedrock API
   key and model access in the selected AWS Region.
2. Bedrock model availability, inference profiles, quotas and supported API
   features can change independently of this repository. Verify them in the AWS
   model cards before a production benchmark run.
3. `LLM_BACKEND=bedrock` applies to Anthropic/OpenAI model traffic. Gemini and
   Novita keep using their existing endpoints.
4. Only the mappings listed above are guaranteed by this change. A different
   Anthropic/OpenAI model in Bedrock mode fails fast with a clear mapping error
   instead of silently calling the wrong endpoint.
5. GPT-5.5 uses the Bedrock Mantle OpenAI-compatible endpoint and therefore must
   be available in the selected Region. Data-residency requirements should be
   reviewed before sending non-public benchmark content.
6. The repository does not contain the separately distributed LegalITA task
   bundle or the private citation-grounding infrastructure; this change does
   not alter that limitation.

## Official references used for the implementation

- AWS Bedrock API keys: https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html
- AWS OpenAI-compatible Responses/Mantle endpoint: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
- AWS API compatibility by model: https://docs.aws.amazon.com/bedrock/latest/userguide/models-api-compatibility.html
- AWS Claude Sonnet 4.6 model card: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-4-6.html
- AWS Claude Opus 4.8 model card: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-8.html
- AWS GPT-5.5 model card: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-55.html
