from typing import Dict, Any, Optional, Tuple
from django.conf import settings
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
import json
import time

# Simple in-memory cache with coarse rounding to reduce repeated reverse geocodes
_REV_CACHE: Dict[Tuple[float, float], Dict[str, Any]] = {}
_REV_ORDER = []
_REV_MAX = 64


def _round_coord(value: float) -> float:
    try:
        return round(float(value), 4)  # ~11m precision
    except Exception:
        return value


def reverse_geocode(latitude: float, longitude: float) -> Dict[str, Any]:
    key = (_round_coord(latitude), _round_coord(longitude))
    cached = _REV_CACHE.get(key)
    if cached is not None:
        return cached

    country: Optional[str] = None
    region: Optional[str] = None
    locality: Optional[str] = None
    full: Optional[str] = None

    # Prefer Google Geocoding API if key available
    api_key = getattr(settings, 'GOOGLE_MAPS_API_KEY', None)
    if api_key:
        try:
            params = {
                'latlng': f"{key[0]},{key[1]}",
                'language': 'en',
                'key': api_key,
            }
            url = f"https://maps.googleapis.com/maps/api/geocode/json?{urlencode(params)}"
            req = Request(url, headers={'User-Agent': 'hackmit2025/1.0'})
            with urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            results = (data or {}).get('results') or []
            if results:
                full = results[0].get('formatted_address')
                comps = results[0].get('address_components') or []
                def get_comp(t: str) -> Optional[str]:
                    for c in comps:
                        if t in (c.get('types') or []):
                            return c.get('long_name') or c.get('short_name')
                    return None
                country = get_comp('country')
                region = get_comp('administrative_area_level_1')
                locality = get_comp('locality') or get_comp('administrative_area_level_2')
        except (HTTPError, URLError, TimeoutError, Exception):
            pass

    # Fallback to OpenStreetMap Nominatim (polite user-agent)
    if not (country or region or locality or full):
        try:
            params = {
                'format': 'jsonv2',
                'lat': str(key[0]),
                'lon': str(key[1]),
                'accept-language': 'en',
            }
            url = f"https://nominatim.openstreetmap.org/reverse?{urlencode(params)}"
            req = Request(url, headers={'User-Agent': 'hackmit2025/1.0 (reverse geocode)'})
            with urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            addr = (data or {}).get('address') or {}
            full = (data or {}).get('display_name')
            country = addr.get('country')
            region = addr.get('state') or addr.get('region') or addr.get('county')
            locality = addr.get('city') or addr.get('town') or addr.get('village') or addr.get('hamlet')
        except (HTTPError, URLError, TimeoutError, Exception):
            pass

    hint_parts = []
    if country:
        hint_parts.append(f"country={country}")
    if region:
        hint_parts.append(f"region={region}")
    if locality:
        hint_parts.append(f"locality={locality}")
    location_hint = "; ".join(hint_parts) if hint_parts else (full or None)

    result = {
        'country': country,
        'region': region,
        'locality': locality,
        'full': full,
        'location_hint': location_hint,
    }

    # Cache
    _REV_CACHE[key] = result
    _REV_ORDER.append(key)
    if len(_REV_ORDER) > _REV_MAX:
        old = _REV_ORDER.pop(0)
        _REV_CACHE.pop(old, None)
    return result


