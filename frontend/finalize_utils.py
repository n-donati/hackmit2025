from django.conf import settings
from smolagents import CodeAgent, LiteLLMModel
import json
import os
from typing import Dict, Any, List
import hashlib
import time

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

def finalize_report(combined: Dict[str, Any], use_case: str) -> Dict[str, Any]:
    """
    Use a small smolagents flow to synthesize the final JSON from combined analysis + user use-case.
    Returns a Python dict with the exact keys required by the UI.
    """
    model = _ensure_model()
    agent = CodeAgent(tools=[], model=model, additional_authorized_imports=["json"], max_steps=1)

    combined_json = json.dumps(combined, ensure_ascii=False)
    prompt = (
        "Given combined_json (string), optional location hint at data.location.hint, and selected_use in {drinking, irrigation, human, animals}, produce dict result with EXACT KEYS: \n"
        "- water_health_percent: '<int>%';\n"
        "- current_water_use_cases: one concise sentence;\n"
        "- potential_dangers: one concise sentence;\n"
        "- purify_for_selected_use: one concise sentence tailored to selected_use.\n"
        "Policy: realistic, non-alarmist; if evidence weak, use 55–75%. Weigh strong visual evidence higher; consider benign causes. Tailor phrasing to selected_use. If selected_use == 'human', interpret as non-consumptive hygiene/cleaning only (e.g., showering, bathing, laundry); never recommend or imply drinking. If location.hint exists (e.g., country/region), you may adapt guidance to typical local constraints; do not hallucinate precise places.\n"
        "Output Python only: import json; data=json.loads(combined_json); selected=selected_use; final_answer(json.dumps(result, ensure_ascii=False))."
    )

    # Simple LRU cache to avoid repeated finalizations on same input
    cache_key = hashlib.sha256((combined_json + "\n" + (use_case or '')).encode('utf-8')).hexdigest()
    cached = _FINALIZE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with suppress_logs_and_output(), suppress_stdout_stderr():
        t0 = time.time()
        out = str(agent.run(prompt, additional_args={
            'combined_json': combined_json,
            'selected_use': use_case,
        }))
        t1 = time.time()
        try:
            # Use print so it always shows up even if logging is configured differently
            print(f"[FINALIZE] agent_run={t1 - t0:.2f}s")
        except Exception:
            pass

    # Robust parse
    try:
        parsed = json.loads(out)
    except Exception:
        start = out.find('{')
        end = out.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(out[start:end+1])
            except Exception:
                parsed = {
                    'water_health_percent': "50%",
                    'current_water_use_cases': "Use with caution; treat before sensitive uses.",
                    'potential_dangers': "Possible microbial or chemical contaminants.",
                    'purify_for_selected_use': "Filter and disinfect before your selected use.",
                }
        else:
            parsed = {
                'water_health_percent': "50%",
                'current_water_use_cases': "Use with caution; treat before sensitive uses.",
                'potential_dangers': "Possible microbial or chemical contaminants.",
                'purify_for_selected_use': "Filter and disinfect before your selected use.",
            }

    # Harmonize messaging with selected use (minimal filtering; keep LLM as final arbiter)
    try:
        selected = (use_case or '').lower()
        def norm(s):
            return (s or '').strip()
        current = norm(parsed.get('current_water_use_cases'))
        purify = norm(parsed.get('purify_for_selected_use'))
        dangers = norm(parsed.get('potential_dangers'))

        def remove_unrelated_mentions(s: str) -> str:
            if not s:
                return s
            lowered = s.lower()
            if selected in ('irrigation', 'human'):
                for term in ['drink', 'consum', 'animal']:
                    if term in lowered:
                        return s.split('.')[0] + '.'
            if selected == 'animals':
                for term in ['drink', 'human']:
                    if term in lowered:
                        return s.split('.')[0] + '.'
            return s

        parsed['current_water_use_cases'] = remove_unrelated_mentions(current) or current
        parsed['potential_dangers'] = remove_unrelated_mentions(dangers) or dangers
        parsed['purify_for_selected_use'] = remove_unrelated_mentions(purify) or purify

        result = parsed
    except Exception:
        result = parsed

    # Update cache
    try:
        _FINALIZE_CACHE[cache_key] = result
        _FINALIZE_ORDER.append(cache_key)
        if len(_FINALIZE_ORDER) > 32:
            old = _FINALIZE_ORDER.pop(0)
            _FINALIZE_CACHE.pop(old, None)
    except Exception:
        pass

    return result


def generate_detailed_plan(final_result: Dict[str, Any], analysis: Dict[str, Any] | None = None) -> List[Dict[str, str]]:
    """
    Generate a step-by-step purification plan using smolagents. Returns a list of
    {title, description} items. Uses the same model singleton for efficiency.
    """
    model = _ensure_model()
    agent = CodeAgent(tools=[], model=model, additional_authorized_imports=["json"], max_steps=1)

    # Build compact context
    ctx = {
        'final': final_result or {},
        'strip_values': ((analysis or {}).get('strip') or {}).get('values') if isinstance(analysis, dict) else None,
        'waterbody': (analysis or {}).get('waterbody') if isinstance(analysis, dict) else None,
        'location': (analysis or {}).get('location') if isinstance(analysis, dict) else None,
    }
    ctx_json = json.dumps(ctx, ensure_ascii=False)

    prompt = (
        "You are a water safety practitioner. Given ctx_json, generate a detailed purification plan.\n"
        "Inputs: ctx_json has keys: final (LLM summary with selected use, dangers, purify guidance, percent),\n"
        "optional strip_values (mapping of analyte->value), optional waterbody (evaluation & observations), optional location (lat,lng,hint).\n\n"
        "Task: Produce a Python list named steps of 6–10 items. Each item is a dict:\n"
        "{ 'title': short string, 'description': 2–4 sentences }.\n\n"
        "Policy:\n"
        "- Align with final.selected_use (non-consumptive if 'human': showering, bathing, laundry; do not imply drinking).\n"
        "- Be realistic, non-alarmist. Adapt to location.hint only if present (country/region).\n"
        "- Start with low-cost actions (settling, cloth/sand filtration), then progressive disinfection options (chlorine/boil/UV) as appropriate.\n"
        "- If strong evidence of risk exists, include extra caution and monitoring.\n"
        "- Avoid brand names; keep steps actionable and safe.\n\n"
        "Output Python only: import json; ctx=json.loads(ctx_json); final_answer(json.dumps(steps, ensure_ascii=False))."
    )

    with suppress_logs_and_output(), suppress_stdout_stderr():
        t0 = time.time()
        out = str(agent.run(prompt, additional_args={
            'ctx_json': ctx_json,
        }))
        t1 = time.time()
        try:
            print(f"[DETAILED] agent_run={t1 - t0:.2f}s")
        except Exception:
            pass

    # Parse robustly to list[dict]
    def _try_parse_list(text: str):
        try:
            data = json.loads(text)
            return data if isinstance(data, list) else None
        except Exception:
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start:end+1])
                    return data if isinstance(data, list) else None
                except Exception:
                    return None
            return None

    steps = _try_parse_list(out) or []
    normalized: List[Dict[str, str]] = []
    for idx, item in enumerate(steps):
        if not isinstance(item, dict):
            continue
        title = str(item.get('title') or f"Step {idx+1}").strip()
        desc = str(item.get('description') or "").strip()
        if not desc:
            continue
        normalized.append({'title': title, 'description': desc})

    if not normalized:
        normalized = [
            {'title': 'Filter water through cloth/sand', 'description': 'Use a clean cloth or sand/gravel filter to remove visible particles and sediments. Repeat until water looks clear.'},
            {'title': 'Disinfect appropriately', 'description': 'Use chlorine, UV, or boiling depending on availability. Adjust dose and contact time; if boiling, bring to a rolling boil for at least 1 minute.'},
            {'title': 'Safe storage', 'description': 'Store in clean, covered containers. Avoid recontamination; use ladles or taps rather than dipping hands.'},
        ]

    return normalized


