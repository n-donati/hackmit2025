from django.conf import settings
from smolagents import CodeAgent, LiteLLMModel
import json
import os
from typing import Dict, Any
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
        "Given combined_json (string) and selected_use in {drinking, irrigation, human, animals}, produce dict result with EXACT KEYS: \n"
        "- water_health_percent: '<int>%';\n"
        "- current_water_use_cases: one concise sentence;\n"
        "- potential_dangers: one concise sentence;\n"
        "- purify_for_selected_use: one concise sentence tailored to selected_use.\n"
        "Policy: realistic, non-alarmist; if evidence weak, use 55–75%. Weigh strong visual evidence higher; consider benign causes. Tailor phrasing to selected_use.\n"
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


