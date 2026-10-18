# Model Regression Detection System

Evaluate LLM prompt/model changes against a golden dataset and baseline, with
regression reports and a CI gate. Gemini classifies and summarizes support emails;
a separate Groq-hosted judge evaluates summary relevance.

## Key Features

- **Versioned YAML prompts** with metadata and few-shot examples.
- **Golden dataset:** 92 reviewed emails across four balanced categories, with ambiguous,
  short, typo-heavy, mixed-language, and sarcastic cases.
- **Bounded async evaluation:** configurable concurrency, ordered results, raw and
  validated outputs, per-case errors, and request latency.
- **Multi-dimensional scoring:** exact category match and 1–5 summary relevance
  from a Groq judge (`openai/gpt-oss-120b`).
- **Token tracking:** separate Gemini and Groq input/output/total counters and
  baseline deltas; missing values remain unknown, not zero.
- **Regression comparison:** pass-rate and per-category accuracy deltas, specific
  regressions and improvements, and configurable warning/critical thresholds.
- **Reports and alerts:** JSON results, HTML diffs with trends, Slack webhooks,
  and seven-run moving-average drift detection with supplied history.
- **GitHub Actions:** offline tests, a quota-controlled evaluation gate, PR
  summaries, and report artifacts. See [CI setup](ci/SETUP.md).
- **Docker and HTTP interface:** a non-root image serving FastAPI and Swagger UI.

## How It Works

1. Load a versioned prompt and a golden dataset or selected subset.
2. Run emails through Gemini, retaining outputs, errors, latency, and token usage.
3. Check categories against reference labels and score summaries with the judge.
4. Compare compatible cases against an explicit baseline. A case passes when its
   category is correct and its valid judge score is at least 4/5.
5. Generate reports and assess thresholds: drops **greater than 3 percentage
   points** warn; drops **greater than 8 points** are critical by default.
6. In CI, fail the gate for critical regressions or incomplete evaluation and
   optionally notify Slack. Merge blocking requires GitHub required-check rules.

## Tech Stack

Python 3.12 · Google GenAI SDK · Groq API via HTTPX · Pydantic · FastAPI/Uvicorn ·
asyncio · YAML/JSON files · unittest · GitHub Actions · Slack webhooks · Docker

## Quick Start

Source: `app/`; CI: `ci/`; HTTP entry: `main.py`. Tests, prompts, and datasets
remain in `tests/`, `prompts/`, and `datasets/`.

From the repository root in Windows PowerShell, with Python 3.12 installed:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create a local `.env` file (never commit it):

```dotenv
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-3.6-flash
```

Use a model available to your Gemini account. For evaluation with the judge, add
`GROQ_API_KEY` and optionally `GROQ_MODEL` (default: `openai/gpt-oss-120b`).
Slack delivery uses `SLACK_WEBHOOK_URL`; `HTML_REPORT_URL` optionally supplies a
hosted report link. Existing environment variables take precedence over `.env`.

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

Open [Swagger UI](http://localhost:8000/docs) and send `POST /classify`:

```json
{"email": "I was charged twice this month. Please refund the extra charge."}
```

The response contains `category` (`billing`, `technical`, `account`, or `general`)
and a summary. This endpoint uses `prompts/v1.yaml` and calls Gemini only; it does
not run the evaluation pipeline or judge. Executing it consumes API quota.
Opening `/docs` or `/openapi.json` does not call a provider. Stop with `Ctrl+C`.

## Docker

With Docker's Linux engine running, execute from the repository root:
```powershell
docker build -t model-regression-detection-system:local .
docker run --detach --rm --name model-regression-api --env-file .env -p 127.0.0.1:8000:8000 model-regression-detection-system:local
docker stop model-regression-api
```

Open `/docs` before stopping the container. Omit `--env-file .env` for a startup
check without credentials. Use unquoted `KEY=value` entries for Docker env files.
Secrets are supplied at runtime, excluded from the build context, and not baked
into the image. The app runs as non-root; startup makes no provider requests.

## Testing

Run the offline suite from the repository root:
```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Latest local verification: **181/181 tests passed**, with mocked provider calls.
Node.js is needed for the CI quota test (skipped if unavailable).
`tests/test_classifier.py` is a manual live-API script, excluded from discovery calls.

Imports use `app.*`. From the repository root, run `python -m app.html_report`
or `python -m app.slack_alerts` (`--help` lists arguments; Slack sends only with
`--send`). The `python ci_pipeline.py` CI entry point is preserved.

## Limitations

- Live CI uses **three fixed cases**, not the full 92-record dataset. PRs targeting
  `main` run offline checks; live evaluation is conditional on prompt changes,
  repository trust, configuration, and available CI quota.
- Runner concurrency is not an RPM/RPD limiter. Live evaluations consume provider
  quota; token counters are usage metadata, not billing or cost estimates.
- Baselines and drift history must be supplied explicitly; there is no automatic
  historical run store. Missing historical token counts cannot be reconstructed.
- Thresholds are heuristic, not statistical significance tests. Judge ratings are
  model assessments; incomplete results are not evidence of passing quality.
