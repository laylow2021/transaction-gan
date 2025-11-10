"""Utility helpers for working with geographic columns."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

Coordinate = Tuple[float, float]


KNOWN_CITY_COORDINATES: Dict[str, Coordinate] = {
    "springfield": (39.7817, -89.6501),
    "riverton": (40.7990, -74.4824),
    "fairview": (35.5097, -86.0733),
    "lakeside": (32.8579, -117.2755),
    "greenville": (34.8526, -82.3940),
    "hillcrest": (37.7749, -122.4194),
    "seattle": (47.6062, -122.3321),
    "san francisco": (37.7749, -122.4194),
    "new york": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "miami": (25.7617, -80.1918),
    "dallas": (32.7767, -96.7970),
    "denver": (39.7392, -104.9903),
    "atlanta": (33.7490, -84.3880),
    "boston": (42.3601, -71.0589),
    "orlando": (28.5384, -81.3789),
    "houston": (29.7604, -95.3698),
}

STATE_CENTRES: Dict[str, Coordinate] = {
    "ca": (36.7783, -119.4179),
    "ny": (43.0000, -75.0000),
    "tx": (31.0000, -100.0000),
    "wa": (47.4009, -121.4905),
    "fl": (27.6648, -81.5158),
    "il": (40.0000, -89.0000),
    "co": (39.1130, -105.3589),
    "ga": (32.1656, -82.9001),
    "az": (34.0489, -111.0937),
    "pa": (41.2033, -77.1945),
}

COUNTRY_CENTRES: Dict[str, Coordinate] = {
    "usa": (39.8283, -98.5795),
    "us": (39.8283, -98.5795),
    "united states": (39.8283, -98.5795),
    "canada": (56.1304, -106.3468),
}

LAT_LON_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def _extract_lat_lon(text: str) -> Coordinate | None:
    values = [float(match) for match in LAT_LON_RE.findall(text)]
    if len(values) >= 2:
        return values[0], values[1]
    return None


@dataclass
class GeoResolver:
    """Resolve free-form addresses to lat/lon pairs with caching."""

    default_coordinate: Coordinate = (0.0, 0.0)
    _cache: Dict[str, Coordinate] = field(default_factory=dict, init=False)

    def resolve(self, raw: str | None) -> Coordinate:
        if raw is None:
            return self.default_coordinate

        text = str(raw).strip()
        if not text:
            return self.default_coordinate

        cached = self._cache.get(text)
        if cached:
            return cached

        lat_lon = _extract_lat_lon(text)
        if not lat_lon:
            lat_lon = self._lookup_from_tokens(text)
        if not lat_lon:
            lat_lon = self.default_coordinate

        self._cache[text] = lat_lon
        return lat_lon

    def reverse(self, latitude: float, longitude: float) -> str:
        """Return a best-effort location name for provided coordinates."""

        best_name = ""
        best_distance = float("inf")

        for name, coord in KNOWN_CITY_COORDINATES.items():
            distance = self._distance((latitude, longitude), coord)
            if distance < best_distance:
                best_distance = distance
                best_name = name.title()

        if best_name and best_distance < 5.0:
            return best_name

        return f"{latitude:.5f},{longitude:.5f}"

    def _lookup_from_tokens(self, text: str) -> Coordinate | None:
        tokens = [token.strip() for token in text.split(",") if token.strip()]
        lookup_tokens: Iterable[str] = (token.lower() for token in tokens[::-1])

        for token in lookup_tokens:
            if token in KNOWN_CITY_COORDINATES:
                return KNOWN_CITY_COORDINATES[token]
            if token in STATE_CENTRES:
                return STATE_CENTRES[token]
            if token in COUNTRY_CENTRES:
                return COUNTRY_CENTRES[token]

        return None

    @staticmethod
    def _distance(a: Coordinate, b: Coordinate) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


__all__ = ["GeoResolver", "Coordinate"]
