"""Real-time alert feed, backed by live weather data.

Uses Open-Meteo (https://open-meteo.com) for geocoding + forecast — it's
free and needs no API key, so it can run in a hackathon/demo environment
without provisioning another secret. Severity is derived from the day's
forecast precipitation volume and probability.

If the live lookup fails for any reason (network issue, location not found,
provider outage) this falls back to the original deterministic simulated
feed, so the endpoint stays fully functional even when offline. The public
interface (`get_alerts(location, client) -> List[Alert]`) is unchanged, so
main.py doesn't need to know which path served the response.
"""
import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Optional

import httpx

from models import Alert, AlertSeverity

logger = logging.getLogger("monsoon_guard.alerts")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 8

# Small in-memory TTL cache: weather doesn't meaningfully change minute to
# minute, so repeated checks of the same location within this window are
# served from cache instead of re-hitting the provider.
_CACHE_TTL_SECONDS = 600
_cache: dict = {}

_SEVERITY_CYCLE = [
    AlertSeverity.LOW,
    AlertSeverity.MODERATE,
    AlertSeverity.HIGH,
    AlertSeverity.SEVERE,
]

_HEADLINES = {
    AlertSeverity.LOW: "Light rain expected",
    AlertSeverity.MODERATE: "Moderate to heavy rain expected",
    AlertSeverity.HIGH: "Heavy rainfall warning",
    AlertSeverity.SEVERE: "Severe weather / flood risk",
}

_MESSAGES = {
    AlertSeverity.LOW: "Carry an umbrella; minor waterlogging possible on low roads.",
    AlertSeverity.MODERATE: "Avoid low-lying underpasses; keep emergency kit accessible.",
    AlertSeverity.HIGH: "Avoid non-essential travel; secure loose outdoor items.",
    AlertSeverity.SEVERE: "Stay indoors; follow local authority evacuation guidance if issued.",
}


def _simulated_alert(location: str) -> Alert:
    """Deterministic fallback feed (original hackathon implementation)."""
    seed = int(hashlib.sha256(location.lower().encode()).hexdigest(), 16)
    severity = _SEVERITY_CYCLE[seed % len(_SEVERITY_CYCLE)]
    now = datetime.now(timezone.utc).isoformat()
    return Alert(
        id=hashlib.sha256(f"{location}-{now[:13]}".encode()).hexdigest()[:12],
        location=location,
        severity=severity,
        headline=_HEADLINES[severity],
        message=_MESSAGES[severity],
        issued_at=now,
    )


def _severity_from_forecast(precip_mm: float, precip_prob: float) -> AlertSeverity:
    if precip_mm >= 60 or precip_prob >= 90:
        return AlertSeverity.SEVERE
    if precip_mm >= 30 or precip_prob >= 70:
        return AlertSeverity.HIGH
    if precip_mm >= 10 or precip_prob >= 40:
        return AlertSeverity.MODERATE
    return AlertSeverity.LOW


async def _fetch_live_alert(location: str, client: httpx.AsyncClient) -> Optional[Alert]:
    geo_resp = await client.get(
        GEOCODE_URL,
        params={"name": location, "count": 1, "language": "en", "format": "json"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    geo_resp.raise_for_status()
    results = geo_resp.json().get("results") or []
    if not results:
        return None
    place = results[0]
    lat, lon = place["latitude"], place["longitude"]
    resolved_name = place.get("name", location)

    fc_resp = await client.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 1,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    fc_resp.raise_for_status()
    daily = fc_resp.json().get("daily") or {}
    precip_mm = (daily.get("precipitation_sum") or [0.0])[0] or 0.0
    precip_prob = (daily.get("precipitation_probability_max") or [0.0])[0] or 0.0

    severity = _severity_from_forecast(precip_mm, precip_prob)
    now = datetime.now(timezone.utc).isoformat()
    return Alert(
        id=hashlib.sha256(f"{resolved_name}-{now[:13]}".encode()).hexdigest()[:12],
        location=resolved_name,
        severity=severity,
        headline=_HEADLINES[severity],
        message=(
            f"{_MESSAGES[severity]} (forecast: {precip_mm:.0f}mm rain, "
            f"{precip_prob:.0f}% chance)"
        ),
        issued_at=now,
    )


async def get_alerts(location: str, client: httpx.AsyncClient) -> List[Alert]:
    location = location.strip()
    cache_key = location.lower()
    cached = _cache.get(cache_key)
    if cached and cached[0] > datetime.now(timezone.utc).timestamp():
        return [cached[1]]

    alert: Optional[Alert] = None
    try:
        alert = await _fetch_live_alert(location, client)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: this is a
        # resilience boundary. Any failure talking to the weather provider
        # (network, malformed response, unexpected shape) should degrade to
        # the simulated feed rather than surface a 502 to the user.
        logger.warning("Live weather lookup failed for %r, falling back to simulated feed: %s", location, exc)

    if alert is None:
        alert = _simulated_alert(location)

    _cache[cache_key] = (datetime.now(timezone.utc).timestamp() + _CACHE_TTL_SECONDS, alert)
    return [alert]
