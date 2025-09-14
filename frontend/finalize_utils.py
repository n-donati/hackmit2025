from django.conf import settings
from smolagents import CodeAgent, LiteLLMModel
import json
import os
from typing import Dict, Any, List
import hashlib
import time
import requests

# Plain-text reference ranges for strip analytes (for qualitative guidance only)
_REFERENCE_RANGES_TEXT = (
    "Total Alkalinity: 40 - 240 mg/L\n"
    "pH: 6.8 - 8.4\n"
    "Hardness: TBD (To Be Determined)\n"
    "Hydrogen Sulfide: 0 mg/L\n"
    "Iron: 0 - 0.3 mg/L\n"
    "Copper: 0 - 1 mg/L\n"
    "Lead: 0 - 15 µg/L\n"
    "Manganese: 0 - 0.1 mg/L\n"
    "Total Chlorine: 0 - 3 mg/L\n"
    "Free Chlorine: 0 - 3 mg/L\n"
    "Nitrate: 0 - 10 mg/L\n"
    "Nitrite: 0 - 1 mg/L\n"
    "Sulfate: 0 - 200 mg/L\n"
    "Zinc: 0 - 5 mg/L\n"
    "Sodium Chloride: 0 - 250 mg/L\n"
    "Fluoride: 0 - 4 mg/L"
)

# Best-effort formatter to convert any strip-related content to a plain-text summary
def _format_strip_context_text(obj: Dict[str, Any] | None) -> str:
    try:
        if not isinstance(obj, dict):
            return ""
        # Accept preformatted text if provided
        pre = (
            obj.get('strip_text')
            or (obj.get('strip') or {}).get('text')
            or (obj.get('strip') or {}).get('analysis_text')
            or (obj.get('strip') or {}).get('analysis')
        )
        if isinstance(pre, str) and pre.strip():
            return pre.strip()

        # Otherwise flatten any values into a readable single-line string
        values = (obj.get('strip') or {}).get('values')
        if isinstance(values, dict) and values:
            parts: List[str] = []
            for key, val in values.items():
                try:
                    parts.append(f"{str(key)}: {str(val)}")
                except Exception:
                    continue
            if parts:
                return "Strip test results — " + "; ".join(parts)
        return ""
    except Exception:
        return ""

# Reuse quieting helpers from waterbody utils if available
try:
    from waterbody.utils import suppress_logs_and_output, suppress_stdout_stderr  # type: ignore
except Exception:
    from contextlib import contextmanager
    @contextmanager
    def suppress_logs_and_output():
        # No-op to show logs
        yield
    @contextmanager
    def suppress_stdout_stderr():
        # No-op to show stdout/stderr
        yield


_MODEL_SINGLETON = None

def _ensure_model() -> LiteLLMModel:
    if getattr(settings, 'GEMINI_API_KEY', None) and 'GEMINI_API_KEY' not in os.environ:
        os.environ['GEMINI_API_KEY'] = settings.GEMINI_API_KEY
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is None:
        _MODEL_SINGLETON = LiteLLMModel(
            model_id="gemini/gemini-2.5-flash",
            temperature=0,
        )
    return _MODEL_SINGLETON


_FINALIZE_CACHE = {}
_FINALIZE_ORDER = []
_API_RESPONSE_CACHE = {}  # Shared cache for full API responses

def finalize_report(combined: Dict[str, Any], use_case: str) -> Dict[str, Any]:
    """
    Use the external API endpoint to synthesize the final JSON from combined analysis + user use-case.
    Returns a Python dict with the exact keys required by the UI.
    """
    # Simple LRU cache to avoid repeated finalizations on same input
    combined_json = json.dumps(combined, ensure_ascii=False)
    cache_key = hashlib.sha256((combined_json + "\n" + (use_case or '')).encode('utf-8')).hexdigest()
    
    # Always try to make a fresh API call first, use cache only as fallback
    result = None
    
    # Prepare the API request payload - sending the exact same info
    api_payload = {
        'combined': combined,
        'use_case': use_case
    }

    try:
        t0 = time.time()
        print(f"[FINALIZE] Starting fresh API request to judge endpoint...")
        print(f"[FINALIZE] Payload keys: {list(api_payload.keys())}")
        
        # Make the API request with extended timeout
        response = requests.post(
            'http://35.233.224.11/judge/',
            json=api_payload,
            timeout=120,  # 2 minute timeout to allow for API processing
            headers={'Content-Type': 'application/json'}
        )
        
        t1 = time.time()
        print(f"[FINALIZE] api_request={t1 - t0:.2f}s, status_code={response.status_code}")
        
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the response
        api_response = response.json()
        print(f"[FINALIZE] API response received successfully")
        print(f"[FINALIZE] Full API response: {json.dumps(api_response, indent=2)}")
        print(f"[FINALIZE] API response keys: {list(api_response.keys())}")
        
        # Cache the full API response for use by generate_detailed_plan
        _API_RESPONSE_CACHE[cache_key] = api_response
        
        # Extract the result from the nested structure
        if 'result' in api_response:
            result = api_response['result']
            print(f"[FINALIZE] Extracted result from 'result' key: {json.dumps(result, indent=2)}")
        else:
            print(f"[FINALIZE] No 'result' key found, using full response as result")
            result = api_response  # Fallback if structure changes
        
        print(f"[FINALIZE] Final result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        print(f"[FINALIZE] water_health_percent value: {result.get('water_health_percent', 'NOT_FOUND')}")
        print(f"[FINALIZE] current_water_use_cases value: {result.get('current_water_use_cases', 'NOT_FOUND')}")
        print(f"[FINALIZE] potential_dangers value: {result.get('potential_dangers', 'NOT_FOUND')}")
        print(f"[FINALIZE] purify_for_selected_use value: {result.get('purify_for_selected_use', 'NOT_FOUND')}")
        
        # Validate that we have the required keys (but don't fail if missing, just log)
        required_keys = ['water_health_percent', 'current_water_use_cases', 'potential_dangers', 'purify_for_selected_use']
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            print(f"[FINALIZE] WARNING: Missing required keys in API response: {missing_keys}")
            # Don't raise an error, just log and continue with what we have
        else:
            print(f"[FINALIZE] SUCCESS: All required keys present in API response")
        
    except requests.exceptions.RequestException as e:
        print(f"[FINALIZE] API request failed: {e}")
        # Try to use cached result as fallback
        cached = _FINALIZE_CACHE.get(cache_key)
        if cached:
            print(f"[FINALIZE] Using cached result as fallback")
            return cached
        # Fallback to default response if API fails and no cache
        result = {
            'water_health_percent': "50%",
            'current_water_use_cases': "Use with caution; treat before sensitive uses.",
            'potential_dangers': "Possible microbial or chemical contaminants.",
            'purify_for_selected_use': "Filter and disinfect before your selected use.",
        }
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"[FINALIZE] API response parsing failed: {e}")
        # Try to use cached result as fallback
        cached = _FINALIZE_CACHE.get(cache_key)
        if cached:
            print(f"[FINALIZE] Using cached result as fallback")
            return cached
        # Fallback to default response if parsing fails and no cache
        result = {
            'water_health_percent': "50%",
            'current_water_use_cases': "Use with caution; treat before sensitive uses.",
            'potential_dangers': "Possible microbial or chemical contaminants.",
            'purify_for_selected_use': "Filter and disinfect before your selected use.",
        }

    # Update cache with successful result
    if result and 'water_health_percent' in result and result['water_health_percent'] != "50%":
        try:
            _FINALIZE_CACHE[cache_key] = result
            _FINALIZE_ORDER.append(cache_key)
            if len(_FINALIZE_ORDER) > 32:
                old = _FINALIZE_ORDER.pop(0)
                _FINALIZE_CACHE.pop(old, None)
            print(f"[FINALIZE] Cached successful API result")
        except Exception:
            pass

    return result


def generate_detailed_plan(final_result: Dict[str, Any], analysis: Dict[str, Any] | None = None) -> List[Dict[str, str]]:
    """
    Get the detailed purification plan from the external API endpoint.
    Returns a list of {title, description} items from the API's purification_plan.
    """
    # Use the same cache key format as finalize_report
    combined = analysis or {}
    combined_json = json.dumps(combined, ensure_ascii=False)
    use_case = final_result.get('selected_use', 'human')
    cache_key = hashlib.sha256((combined_json + "\n" + (use_case or '')).encode('utf-8')).hexdigest()
    
    # First check if we have the full API response cached from finalize_report
    api_response_cached = _API_RESPONSE_CACHE.get(cache_key)
    if api_response_cached:
        print(f"[DETAILED] Using cached API response from finalize_report")
        # Extract the purification_plan from the cached response
        if 'purification_plan' in api_response_cached:
            purification_plan = api_response_cached['purification_plan']
            if isinstance(purification_plan, list):
                # Normalize the plan format
                normalized_plan = []
                for item in purification_plan:
                    if isinstance(item, dict) and 'title' in item and 'description' in item:
                        normalized_plan.append({
                            'title': str(item['title']).strip(),
                            'description': str(item['description']).strip()
                        })
                
                if normalized_plan:
                    return normalized_plan
    
    # If we don't have it cached, make the API call
    api_payload = {
        'combined': combined,
        'use_case': use_case
    }

    try:
        t0 = time.time()
        print(f"[DETAILED] Starting API request for purification plan...")
        
        # Make the API request with extended timeout
        response = requests.post(
            'http://35.233.224.11/judge/',
            json=api_payload,
            timeout=120,  # 2 minute timeout to allow for API processing
            headers={'Content-Type': 'application/json'}
        )
        
        t1 = time.time()
        print(f"[DETAILED] api_request={t1 - t0:.2f}s, status_code={response.status_code}")
        
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the response
        api_response = response.json()
        print(f"[DETAILED] API response received successfully")
        
        # Cache the full API response for future use
        _API_RESPONSE_CACHE[cache_key] = api_response
        
        # Extract the purification_plan from the API response
        if 'purification_plan' in api_response:
            purification_plan = api_response['purification_plan']
            if isinstance(purification_plan, list):
                # Normalize the plan format
                normalized_plan = []
                for item in purification_plan:
                    if isinstance(item, dict) and 'title' in item and 'description' in item:
                        normalized_plan.append({
                            'title': str(item['title']).strip(),
                            'description': str(item['description']).strip()
                        })
                
                if normalized_plan:
                    return normalized_plan
        
        # If we can't extract the plan, fall back to default
        print(f"[DETAILED] Could not extract purification_plan from API response")
        
    except Exception as e:
        print(f"[DETAILED] API request failed: {e}")
    
    # Fallback default plan
    return [
        {
            "title": "Filter water",
            "description": "Use clean cloth or filter to remove visible particles and sediment."
        },
        {
            "title": "Disinfect",
            "description": "Boil water for 1 minute, use purification tablets, or add unscented bleach (1/4 tsp per gallon)."
        },
        {
            "title": "Safe storage",
            "description": "Store in clean, covered containers. Use within 24 hours if not refrigerated."
        }
    ]


