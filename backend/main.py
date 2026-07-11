"""Monsoon Guard API.

Run locally:  python -m uvicorn main:app --reload
Env required: GEMINI_API_KEY (see .env.example)
"""
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import AsyncIterator, Deque, Dict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models import (
    Alert,
    ChecklistRequest,
    GeneratedPlan,
    PreparednessRequest,
    TranslateRequest,
    TravelAdvisoryRequest,
)
from prompts import (
    CHECKLIST_SYSTEM_INSTRUCTION,
    PLAN_SYSTEM_INSTRUCTION,
    TRANSLATE_SYSTEM_INSTRUCTION,
    TRAVEL_SYSTEM_INSTRUCTION,
    build_checklist_prompt,
    build_plan_prompt,
    build_translate_prompt,
    build_travel_prompt,
)
from services import alerts_service
from services.gemini_service import GeminiConfigError, GeminiRequestError, generate_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("monsoon_guard")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # One shared HTTP client for the app's lifetime instead of opening/closing
    # a new connection (TCP + TLS handshake) on every single request — this
    # was the main efficiency cost in the previous version. httpx pools and
    # reuses keep-alive connections across requests to the same host.
    app.state.http_client = httpx.AsyncClient()
    try:
        yield
    finally:
        await app.state.http_client.aclose()


app = FastAPI(
    title="Monsoon Guard API",
    description="GenAI-powered monsoon preparedness assistant for PromptWars.",
    version="1.1.0",
    lifespan=lifespan,
)

# --- CORS -------------------------------------------------------------
# Locked to configured origins rather than "*" so the API can't be embedded
# and hammered from arbitrary third-party pages in production.
_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# --- Minimal in-memory rate limiting ------------------------------------
# Good enough for a single-instance hackathon deployment; swap for
# Redis-backed limiting (e.g. slowapi + Redis) before scaling horizontally.
_RATE_LIMIT = 20  # requests
_RATE_WINDOW = 60  # seconds
_hits: Dict[str, Deque[float]] = defaultdict(deque)


@app.middleware("http")
async def rate_limit_and_security_headers(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _hits[client_ip]
    while bucket and now - bucket[0] > _RATE_WINDOW:
        bucket.popleft()
    if len(bucket) >= _RATE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={"error": "Too many requests. Please slow down and try again shortly."},
        )
    bucket.append(now)
    # Evict IPs with no recent activity so this dict doesn't grow unbounded
    # over the process's lifetime (each entry was previously kept forever).
    if not bucket:
        _hits.pop(client_ip, None)

    response = await call_next(request)
    # Defensive security headers (secure practice, near-zero cost).
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def _handle_gemini_errors(exc: Exception):
    if isinstance(exc, GeminiConfigError):
        logger.error("Config error: %s", exc)
        raise HTTPException(status_code=500, detail="Server misconfiguration. Contact the admin.")
    if isinstance(exc, GeminiRequestError):
        logger.error("Upstream error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))
    logger.exception("Unexpected error")
    raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@app.get("/health")
async def health() -> dict:
    """Liveness/readiness probe for Render/Vercel and uptime checks."""
    return {"status": "ok"}


@app.post("/api/preparedness-plan", response_model=GeneratedPlan)
async def preparedness_plan(req: PreparednessRequest, request: Request):
    """Personalized, weather-aware monsoon preparedness plan for a household."""
    try:
        data = await generate_json(
            PLAN_SYSTEM_INSTRUCTION, build_plan_prompt(req), request.app.state.http_client
        )
        return GeneratedPlan(
            summary=data.get("summary", ""),
            action_items=data.get("action_items", []),
            checklist=data.get("checklist", []),
            emergency_contacts_hint=data.get("emergency_contacts_hint", []),
            language=req.language,
        )
    except (GeminiConfigError, GeminiRequestError) as exc:
        _handle_gemini_errors(exc)


@app.post("/api/checklist")
async def checklist(req: ChecklistRequest, request: Request):
    """Standalone emergency checklist generator."""
    try:
        data = await generate_json(
            CHECKLIST_SYSTEM_INSTRUCTION, build_checklist_prompt(req), request.app.state.http_client
        )
        return {"checklist": data.get("checklist", [])}
    except (GeminiConfigError, GeminiRequestError) as exc:
        _handle_gemini_errors(exc)


@app.post("/api/travel-advisory", response_model=GeneratedPlan)
async def travel_advisory(req: TravelAdvisoryRequest, request: Request):
    """Monsoon-aware travel advisory for a given route/date/mode."""
    try:
        data = await generate_json(
            TRAVEL_SYSTEM_INSTRUCTION, build_travel_prompt(req), request.app.state.http_client
        )
        return GeneratedPlan(
            summary=data.get("summary", ""),
            action_items=data.get("action_items", []),
            checklist=data.get("checklist", []),
            emergency_contacts_hint=data.get("emergency_contacts_hint", []),
            language=req.language,
        )
    except (GeminiConfigError, GeminiRequestError) as exc:
        _handle_gemini_errors(exc)


@app.post("/api/translate")
async def translate(req: TranslateRequest, request: Request):
    """Multilingual assistance: translate any guidance text on demand."""
    try:
        data = await generate_json(
            TRANSLATE_SYSTEM_INSTRUCTION,
            build_translate_prompt(req.text, req.target_language.value),
            request.app.state.http_client,
        )
        return {"translated_text": data.get("translated_text", "")}
    except (GeminiConfigError, GeminiRequestError) as exc:
        _handle_gemini_errors(exc)


@app.get("/api/alerts", response_model=list[Alert])
async def alerts(location: str, request: Request):
    """Real-time weather alerts for a location, backed by live forecast data
    (falls back to a deterministic simulated feed if the provider is
    unreachable). See alerts_service docstring for details."""
    location = location.strip()
    if not location or len(location) > 100:
        raise HTTPException(status_code=422, detail="location must be 1-100 characters.")
    return await alerts_service.get_alerts(location, request.app.state.http_client)
