# Monsoon Guard

GenAI-powered monsoon preparedness assistant — built for the **PromptWars: Monsoon
Preparedness (Citizen Assistance)** challenge.

## Problem statement alignment

| Requirement | Where it's implemented |
|---|---|
| Personalized preparedness plans | `POST /api/preparedness-plan` — tailored to household type, dwelling type (flood/landslide-aware), size, vehicle access, special needs |
| Emergency checklists | `POST /api/checklist` + checklist embedded in the plan response |
| Travel advisories | `POST /api/travel-advisory` — route + mode + date aware |
| Safety recommendations | `action_items` in plan/travel responses |
| Multilingual assistance | English / Hindi / Kannada / Marathi via `language` field on every endpoint, plus a standalone `POST /api/translate` |
| Real-time alerts | `GET /api/alerts?location=` — live forecast data (see below) |

**Note on "real-time":** `/api/alerts` is backed by live weather data from
[Open-Meteo](https://open-meteo.com) (free, no API key required — geocodes
the location, then reads the day's precipitation forecast to derive a
severity level). If that lookup fails for any reason (network issue,
unrecognized location, provider outage) it degrades to the original
deterministic simulated feed (`services/alerts_service.py`) so the endpoint
never hard-fails a demo. The endpoint contract (`List[Alert]`) is designed so
a different provider (IMD, OpenWeatherMap, data.gov.in) can be dropped in
behind it with no API changes.

## Architecture

```
monsoon-guard/
├── backend/
│   ├── main.py              # FastAPI routes, CORS, rate limiting, security headers
│   ├── models.py            # Pydantic request/response schemas (validation)
│   ├── prompts.py           # Pure prompt-building functions (testable, no I/O)
│   ├── services/
│   │   ├── gemini_service.py   # Gemini API client, structured JSON output, error handling
│   │   └── alerts_service.py   # Alert feed (swappable for a live provider)
│   └── tests/test_main.py   # pytest suite, Gemini calls mocked
└── frontend/
    └── index.html           # Zero-build vanilla HTML/CSS/JS, WCAG-conscious
```

## Design choices (why each scoring pillar is covered)

- **Code quality**: layered structure (routes / schemas / prompts / services),
  docstrings explaining *why* not just *what*, type hints throughout.
- **Security**: server-side Pydantic validation with length/range limits on
  every field, API key never exposed to the client, CORS locked to configured
  origins (not `*`), in-memory rate limiting, defensive security headers,
  no secrets committed (`.env.example` only), errors don't leak internals.
- **Efficiency**: a single `httpx.AsyncClient` is created once at app startup
  (FastAPI `lifespan`) and reused for every Gemini and weather-API call,
  instead of opening/closing a connection per request. Transient upstream
  failures (timeouts, 429/5xx) retry with exponential backoff. Identical
  Gemini requests and repeated alert lookups for the same location are
  served from a short-TTL in-memory cache. `response_mime_type:
  application/json` avoids fragile markdown-fence parsing.
- **Testing**: 14 pytest cases covering happy paths, validation failures,
  upstream-failure handling, rate limiting, and security headers — all mocked
  so they run offline and deterministically.
- **Accessibility**: semantic HTML5, associated `<label>`s, `aria-live`
  regions for async results and alerts, visible focus states, responsive
  layout, sufficient color contrast.

## Local setup

```bash
cd backend
python -m pip install -r requirements.txt --break-system-packages   # Windows Application Control: use python -m pip, not pip.exe
cp .env.example .env   # then add your real GEMINI_API_KEY
python -m uvicorn main:app --reload   # not uvicorn.exe directly
```

Open `frontend/index.html` directly, or serve it:

```bash
cd frontend
python -m http.server 5500
```

Set `window.MONSOON_GUARD_API_BASE` in `index.html` (or before it loads) to
your backend URL when not running on `localhost:8000`.

## Tests

```bash
cd backend
python -m pytest -q
```

## Deploy

- **Backend → Render**: uses `render.yaml` at repo root (`rootDir: backend`).
  Set `GEMINI_API_KEY` in the Render dashboard (not committed). Update
  `ALLOWED_ORIGINS` to your live Vercel URL.
- **Frontend → Vercel**: uses `vercel.json` at repo root, zero build step,
  serves `frontend/`.
