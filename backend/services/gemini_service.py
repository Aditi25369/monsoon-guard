"""Thin wrapper around the Gemini API.

Isolated in its own module so:
- main.py stays free of prompt strings / HTTP details (readability)
- it can be mocked in tests without hitting the network (testability)
- API-key / network failures are converted into typed exceptions (security/robustness)

The httpx.AsyncClient is created once at app startup (see main.py's lifespan)
and passed in here, rather than opened/closed per-request. Opening a fresh
TCP+TLS connection for every call was the single biggest efficiency cost in
this module; reusing a client lets httpx pool and keep-alive connections.
"""
import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("monsoon_guard.gemini")

GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)
REQUEST_TIMEOUT_SECONDS = 20

# Transient upstream statuses worth a short retry (rate limit / overload /
# gateway hiccups). Anything else (4xx client errors, auth failures) is not
# retried since retrying won't change the outcome.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 0.5

# Tiny in-memory TTL cache so identical requests (same prompt+instruction,
# e.g. two people checking the same location/dwelling combo within a few
# minutes) don't each burn a fresh Gemini call. Fine for a single-instance
# deployment; swap for Redis before scaling horizontally.
_CACHE_TTL_SECONDS = 300
_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}


class GeminiConfigError(RuntimeError):
    """Raised when the API key is missing/misconfigured."""


class GeminiRequestError(RuntimeError):
    """Raised when the upstream call fails or returns unusable content."""


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise GeminiConfigError(
            "GEMINI_API_KEY is not set. Add it to your environment / .env file."
        )
    return key


def _cache_key(system_instruction: str, user_prompt: str) -> str:
    return f"{hash(system_instruction)}:{hash(user_prompt)}"


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    hit = _cache.get(key)
    if hit is None:
        return None
    expires_at, value = hit
    if time.monotonic() > expires_at:
        _cache.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: Dict[str, Any]) -> None:
    _cache[key] = (time.monotonic() + _CACHE_TTL_SECONDS, value)


async def generate_json(
    system_instruction: str,
    user_prompt: str,
    client: httpx.AsyncClient,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Call Gemini and parse a strict-JSON response.

    Uses response_mime_type=application/json so the model is constrained to
    valid JSON, avoiding brittle regex/markdown-fence stripping. Retries
    transient upstream failures with exponential backoff, and serves cached
    results for identical (instruction, prompt) pairs within the TTL window.
    """
    key = _cache_key(system_instruction, user_prompt)
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            logger.info("Serving cached Gemini response")
            return cached

    api_key = _get_api_key()
    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "response_mime_type": "application/json",
        },
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    last_exc: Optional[Exception] = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            resp = await client.post(
                GEMINI_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
            )
        except httpx.TimeoutException as exc:
            last_exc = exc
            logger.warning("Gemini request timed out (attempt %d/%d)", attempt, _MAX_ATTEMPTS)
        except httpx.HTTPError as exc:
            last_exc = exc
            logger.warning("Gemini transport error (attempt %d/%d): %s", attempt, _MAX_ATTEMPTS, exc)
        else:
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    parsed = json.loads(text)
                except (KeyError, IndexError, json.JSONDecodeError) as exc:
                    logger.error("Failed to parse Gemini response: %s", exc)
                    raise GeminiRequestError("The AI service returned an unexpected response.") from exc
                if use_cache:
                    _cache_set(key, parsed)
                return parsed

            if resp.status_code not in _RETRYABLE_STATUS:
                logger.error("Gemini returned status %s: %s", resp.status_code, resp.text[:500])
                raise GeminiRequestError(f"AI service error (status {resp.status_code}).")

            last_exc = GeminiRequestError(f"AI service error (status {resp.status_code}).")
            logger.warning(
                "Gemini returned retryable status %s (attempt %d/%d)",
                resp.status_code, attempt, _MAX_ATTEMPTS,
            )

        if attempt < _MAX_ATTEMPTS:
            await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))

    if isinstance(last_exc, httpx.TimeoutException):
        raise GeminiRequestError("The AI service timed out. Please retry.") from last_exc
    if isinstance(last_exc, httpx.HTTPError):
        raise GeminiRequestError("Could not reach the AI service.") from last_exc
    raise last_exc or GeminiRequestError("The AI service failed after retries.")
