#!/usr/bin/env python3
"""Fetch Everest weather and produce a small simulation-friendly JSON payload."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://api.open-meteo.com/v1/forecast"

LOCATIONS = {
    "lukla": {"name": "Lukla", "latitude": 27.6869, "longitude": 86.7297, "elevation_m": 2860},
    "namche": {"name": "Namche Bazaar", "latitude": 27.8069, "longitude": 86.7140, "elevation_m": 3440},
    "base-camp": {"name": "Everest Base Camp", "latitude": 28.0026, "longitude": 86.8528, "elevation_m": 5364},
    "south-col": {"name": "South Col", "latitude": 27.9717, "longitude": 86.9292, "elevation_m": 7906},
    "summit": {"name": "Mount Everest Summit", "latitude": 27.9881, "longitude": 86.9250, "elevation_m": 8849},
}

CURRENT_FIELDS = (
    "temperature_2m",
    "apparent_temperature",
    "surface_pressure",
    "precipitation",
    "snowfall",
    "weather_code",
    "cloud_cover",
    "visibility",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
)

WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

AIR_GAS_CONSTANT = 287.05
G1_DRAG_COEFFICIENT = 1.2
G1_FRONTAL_AREA_M2 = 0.7
MAX_WIND_FORCE_N = 120.0
BASE_SNOW_FRICTION = 0.35


@dataclass(frozen=True)
class Risk:
    score: int
    level: str
    reasons: list[str]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def derive_simulation_parameters(current: dict[str, Any], risk: Risk) -> dict[str, Any]:
    """Convert a weather observation into bounded simulation priors.

    The snow values are an initial material prior, not a site-calibrated snow
    profile. They become active friction immediately; deformable coefficients
    remain metadata until the Newton MPM path is running.
    """
    temperature_c = float(current.get("temperature_2m") or 0.0)
    pressure_hpa = max(1.0, float(current.get("surface_pressure") or 1013.25))
    precipitation_mm = max(0.0, float(current.get("precipitation") or 0.0))
    snowfall_cm = max(0.0, float(current.get("snowfall") or 0.0))
    wind_speed_kmh = max(0.0, float(current.get("wind_speed_10m") or 0.0))
    wind_gust_kmh = max(wind_speed_kmh, float(current.get("wind_gusts_10m") or 0.0))
    visibility_m = max(0.0, float(current.get("visibility") or 0.0))

    temperature_k = max(180.0, temperature_c + 273.15)
    air_density = pressure_hpa * 100.0 / (AIR_GAS_CONSTANT * temperature_k)
    gust_ms = wind_gust_kmh / 3.6
    dynamic_pressure = 0.5 * air_density * gust_ms**2
    wind_force_n = clamp(
        dynamic_pressure * G1_DRAG_COEFFICIENT * G1_FRONTAL_AREA_M2,
        0.0,
        MAX_WIND_FORCE_N,
    )

    fresh_snow = clamp(snowfall_cm / 1.0, 0.0, 1.0)
    wind_packing = clamp((max(wind_speed_kmh, wind_gust_kmh) - 15.0) / 45.0, 0.0, 1.0)
    melt_wetness = clamp((temperature_c + 10.0) / 10.0, 0.0, 1.0)
    rain_wetness = clamp(precipitation_mm / 2.0, 0.0, 1.0) if temperature_c > -3.0 else 0.0
    wetness = max(melt_wetness, rain_wetness)

    density = clamp(90.0 + 180.0 * wind_packing + 140.0 * wetness, 60.0, 450.0)
    stiffness = clamp(
        30_000.0 * (density / 100.0) ** 2.4 * (1.0 + 2.0 * wind_packing),
        20_000.0,
        4_000_000.0,
    )
    compressive_strength = clamp(
        2_500.0 * (density / 100.0) ** 2.0 * (1.0 + wind_packing),
        1_000.0,
        250_000.0,
    )
    shear_strength = clamp(compressive_strength * (0.28 + 0.20 * wind_packing), 300.0, 100_000.0)
    cohesion = clamp(250.0 + 6.0 * density + 5_000.0 * wetness, 500.0, 10_000.0)
    surface_friction = clamp(
        0.38 + 0.07 * wind_packing - 0.10 * fresh_snow - 0.14 * wetness,
        0.12,
        0.48,
    )
    friction_scale = clamp(surface_friction / BASE_SNOW_FRICTION, 0.35, 1.0)

    if wetness >= 0.55:
        snow_type = "WET_SNOW"
    elif wind_packing >= 0.45:
        snow_type = "WIND_PACK"
    elif fresh_snow >= 0.2:
        snow_type = "POWDER"
    elif temperature_c <= -20.0:
        snow_type = "COLD_DRY"
    else:
        snow_type = "PACKED_SNOW"

    confidence = clamp(0.30 + 0.30 * fresh_snow + 0.15 * wind_packing + 0.10 * wetness, 0.30, 0.75)
    return {
        "wind_force_n": wind_force_n,
        "wind_force_scale": wind_force_n / MAX_WIND_FORCE_N,
        "air_density_kg_m3": air_density,
        "dynamic_pressure_pa": dynamic_pressure,
        "terrain_friction_scale": friction_scale,
        "visibility_scale": clamp(visibility_m / 10_000.0, 0.0, 1.0),
        "movement_allowed": risk.level not in {"HIGH", "EXTREME"},
        "snow_prior": {
            "model": "heuristic-weather-prior/v1",
            "snow_type": snow_type,
            "fresh_layer_increment_m": snowfall_cm / 100.0,
            "active_layer_thickness_m": clamp(0.08 + snowfall_cm / 100.0, 0.05, 0.30),
            "snowfall_rate_mm_h": snowfall_cm * 10.0,
            "density_kg_m3": density,
            "stiffness_pa": stiffness,
            "compressive_strength_pa": compressive_strength,
            "shear_strength_pa": shear_strength,
            "compaction_hardening": 2.0 + 8.0 * wind_packing,
            "bond_strength_below_pa": cohesion,
            "surface_friction": surface_friction,
            "wind_packing_index": wind_packing,
            "wetness_index": wetness,
            "confidence": confidence,
            "calibration": "weather-derived prior; requires robot sensing or incident system ID",
        },
        "derivation": {
            "wind": "0.5 * air_density * gust_speed^2 * Cd(1.2) * area(0.7m2)",
            "snow": "bounded heuristic from snowfall, temperature, precipitation, wind speed and gust",
        },
    }


def calculate_risk(current: dict[str, Any]) -> Risk:
    """Calculate a transparent heuristic risk score for the demo."""
    score = 0
    reasons: list[str] = []
    gust = float(current.get("wind_gusts_10m") or 0)
    visibility = float(current.get("visibility") or 999_999)
    snowfall = float(current.get("snowfall") or 0)
    temperature = float(current.get("temperature_2m") or 0)

    if gust >= 60:
        score += 3
        reasons.append("dangerous wind gusts")
    elif gust >= 40:
        score += 2
        reasons.append("strong wind gusts")

    if visibility < 1_000:
        score += 3
        reasons.append("very low visibility")
    elif visibility < 5_000:
        score += 2
        reasons.append("reduced visibility")

    if snowfall > 0:
        score += 2
        reasons.append("active snowfall")

    if temperature <= -20:
        score += 2
        reasons.append("extreme cold")
    elif temperature <= -10:
        score += 1
        reasons.append("severe cold")

    if score >= 9:
        level = "EXTREME"
    elif score >= 6:
        level = "HIGH"
    elif score >= 3:
        level = "MODERATE"
    else:
        level = "LOW"
    return Risk(score, level, reasons or ["no configured hazard threshold exceeded"])


def fetch_weather(location: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "elevation": location["elevation_m"],
        "current": ",".join(CURRENT_FIELDS),
        "timezone": "Asia/Kathmandu",
    }
    request = Request(
        f"{API_URL}?{urlencode(params)}",
        headers={"User-Agent": "everest-dream-weather/1.0"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def build_payload(location: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    current = response["current"]
    risk = calculate_risk(current)
    weather_code = int(current.get("weather_code", -1))
    return {
        "schema": "everest-weather/v1",
        "source": "Open-Meteo forecast simulation data",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "location": {
            "name": location["name"],
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "elevation_m": location["elevation_m"],
        },
        "observed_at": current["time"],
        "conditions": {
            "summary": WEATHER_CODES.get(weather_code, f"WMO weather code {weather_code}"),
            "weather_code": weather_code,
            "temperature_c": current.get("temperature_2m"),
            "apparent_temperature_c": current.get("apparent_temperature"),
            "surface_pressure_hpa": current.get("surface_pressure"),
            "precipitation_mm": current.get("precipitation"),
            "snowfall_cm": current.get("snowfall"),
            "cloud_cover_percent": current.get("cloud_cover"),
            "visibility_m": current.get("visibility"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "wind_direction_deg": current.get("wind_direction_10m"),
            "wind_gust_kmh": current.get("wind_gusts_10m"),
        },
        "risk": asdict(risk),
        "simulation": derive_simulation_parameters(current, risk),
    }


def apply_to_dashboard(payload: dict[str, Any], dashboard_url: str, timeout: float = 10.0) -> dict[str, Any]:
    """Apply this weather payload to a running Everest dashboard."""
    endpoint = f"{dashboard_url.rstrip('/')}/api/control"
    body = json.dumps({"action": "weather", "value": payload}).encode()
    request = Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "everest-dream-weather/1.0"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("location", nargs="?", choices=LOCATIONS, default="base-camp")
    parser.add_argument("--compact", action="store_true", help="Print compact JSON")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds")
    parser.add_argument(
        "--dashboard-url",
        help="Apply the resulting simulation parameters to a running dashboard",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    location = LOCATIONS[args.location]
    try:
        response = fetch_weather(location, args.timeout)
        payload = build_payload(location, response)
        if args.dashboard_url:
            apply_to_dashboard(payload, args.dashboard_url, args.timeout)
    except (HTTPError, URLError, TimeoutError, KeyError, ValueError) as error:
        print(json.dumps({"schema": "everest-weather/v1", "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=None if args.compact else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
