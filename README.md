# Human vs Bot AI Service

FastAPI service for the AI subject in Human vs Bot.

## Local Development

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\uvicorn main:app --reload
```

For local smoke tests without calling Gemini:

```bash
$env:MOCK_AI_RESPONSE="true"
venv\Scripts\uvicorn main:app --reload
```

## Configuration

Copy `.env.example` to `.env` and fill in values as needed.

Important variables:

- `GEMINI_API_KEY`: required for real model calls.
- `AI_MODEL_NAME`: model used by the maintained Google GenAI SDK.
- `MOCK_AI_RESPONSE`: set to `true` for local smoke tests.
- `PROMPT_VERSION`: prompt file name under `prompts/` without `.txt`.
- `ALLOWED_ORIGINS`: comma-separated frontend/middleware origins.
- `SESSION_TTL_SECONDS`: cleanup TTL for chat sessions.
- `RATE_LIMIT_COUNT`: messages allowed per session in the rate-limit window.

## Verification

```bash
venv\Scripts\python -m unittest discover tests
venv\Scripts\python evals\run_evals.py
venv\Scripts\python check_models.py
```

## Health

- `GET /health`: process liveness.
- `GET /ready`: model/prompt readiness.

## SDK

This service uses the maintained `google-genai` package and `from google import genai` client API. Do not reintroduce the deprecated `google.generativeai` import.
