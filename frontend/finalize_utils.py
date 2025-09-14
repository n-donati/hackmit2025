from django.conf import settings
from smolagents import CodeAgent, LiteLLMModel
import json
import os
from typing import Dict, Any, List
import hashlib
import time

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

def finalize_report(combined: Dict[str, Any], use_case: str) -> Dict[str, Any]:
    """
    Use a small smolagents flow to synthesize the final JSON from combined analysis + user use-case.
    Returns a Python dict with the exact keys required by the UI.
    """
    model = _ensure_model()
    agent = CodeAgent(tools=[], model=model, additional_authorized_imports=["json"], max_steps=1)

    combined_json = json.dumps(combined, ensure_ascii=False)
    strip_context_text = _format_strip_context_text(combined)
    # Embed plain-text guidance directly in the prompt so the agent sees it without parsing
    prompt = (
        "Given combined_json (string), optional location hint at data.location.hint, and selected_use in {drinking, irrigation, human, animals}, produce dict result with EXACT KEYS: \n"
        "- water_health_percent: '<int>%';\n"
        "- current_water_use_cases: one concise sentence;\n"
        "- potential_dangers: one concise sentence;\n"
        "- purify_for_selected_use: one concise sentence tailored to selected_use.\n"
        "Policy: realistic, non-alarmist; if evidence weak, use 55–75%. Weigh strong visual evidence higher; consider benign causes. Tailor phrasing to selected_use. If selected_use == 'human', interpret as non-consumptive hygiene/cleaning only (e.g., showering, bathing, laundry); never recommend or imply drinking. If location.hint exists (e.g., country/region), you may adapt guidance to typical local constraints; do not hallucinate precise places.\n"
        "Strip guidance: treat strip test context as LOW-CONFIDENCE, supplementary only. Prefer non-strip sources (visual waterbody analysis, general knowledge, location). If strip context conflicts or is ambiguous, ignore it. Do not change water_health_percent by more than ±5% based solely on strip context.\n\n"
        "Additional context (plain text; do not parse into structured data):\n"
        "Reference ranges:\n" + _REFERENCE_RANGES_TEXT + "\n\n"
        "Strip test context (low-confidence):\n" + (strip_context_text or "(none)") + "\n\n"
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

    # Build compact context with plain-text strip info
    strip_text = _format_strip_context_text(analysis if isinstance(analysis, dict) else None)
    ctx = {
        'final': final_result or {},
        'strip_text': strip_text,
        'reference_ranges': _REFERENCE_RANGES_TEXT,
        'waterbody': (analysis or {}).get('waterbody') if isinstance(analysis, dict) else None,
        'location': (analysis or {}).get('location') if isinstance(analysis, dict) else None,
    }
    ctx_json = json.dumps(ctx, ensure_ascii=False)

    prompt = (
        "You are a water safety practitioner. Given ctx_json, generate a detailed purification plan.\n"
        "Inputs: ctx_json has keys: final (LLM summary with selected use, dangers, purify guidance, percent),\n"
        "optional strip_text (plain text summary of test strip results), reference_ranges (plain text table for qualitative guidance), optional waterbody (evaluation & observations), optional location (lat,lng,hint).\n\n"
        "Note: strip_text and reference_ranges are LOW-CONFIDENCE, for qualitative reasoning only; do not attempt to parse into structured data. Prefer non-strip evidence (final summary, waterbody, general knowledge). If strip hints conflict or are unclear, ignore them.\n\n"
        "Task: Produce a Python list named steps of 4–6 items. Each item is a dict:\n"
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


