from django.conf import settings
from smolagents import CodeAgent, LiteLLMModel
import json
import os
from typing import Dict, Any

# Reuse quieting helpers from waterbody utils if available
try:
    from waterbody.utils import suppress_logs_and_output, suppress_stdout_stderr  # type: ignore
except Exception:
    from contextlib import contextmanager
    @contextmanager
    def suppress_logs_and_output():
        yield
    @contextmanager
    def suppress_stdout_stderr():
        yield


def _ensure_model() -> LiteLLMModel:
    if getattr(settings, 'GEMINI_API_KEY', None) and 'GEMINI_API_KEY' not in os.environ:
        os.environ['GEMINI_API_KEY'] = settings.GEMINI_API_KEY
    return LiteLLMModel(model_id="gemini/gemini-2.5-flash")


def finalize_report(combined: Dict[str, Any], use_case: str) -> Dict[str, Any]:
    """
    Use a small smolagents flow to synthesize the final JSON from combined analysis + user use-case.
    Returns a Python dict with the exact keys required by the UI.
    """
    model = _ensure_model()
    agent = CodeAgent(tools=[], model=model, additional_authorized_imports=["json"])

    combined_json = json.dumps(combined, ensure_ascii=False)
    prompt = (
        "You are a water safety expert. You will receive two variables: combined_json (a JSON string) "
        "and selected_use (one of 'drinking', 'irrigation', 'human', 'animals').\n\n"
        "Task: Parse combined_json, then construct a dict named result with EXACTLY these keys: \n"
        "- water_health_percent: string like '55%' (choose an integer percent and add %).\n"
        "- current_water_use_cases: concise sentence describing safe uses now.\n"
        "- potential_dangers: concise sentence on key risks (pathogens, metals, chemicals).\n"
        "- purify_for_selected_use: one-sentence, actionable purification guidance tailored to selected_use.\n\n"
        "Calibration: Be realistic and not pessimistic. Minor imperfections (visual tint, small strip deviations) should only slightly reduce the health percent.\n"
        "If evidence is weak or ambiguous, prefer a moderate score (55–75%) with practical guidance rather than severe warnings.\n"
        "If combined.waterbody.evaluation contains STRONG EVIDENCE (e.g., visible trash piles, discharge pipes, oil sheen, dead fish), weight risks higher.\n"
        "Otherwise, treat waterbody classification as a hint and balance with strip analytes. Consider benign explanations (natural sediments, plant reflections).\n"
        "Tailoring: Align phrasing to selected_use. Do NOT mention drinking or animal consumption if selected_use is irrigation or human.\n"
        "Solutions: prioritize free/low-cost, accessible steps (settling, cloth filtration, sand/gravel filtration, basic chlorination, SODIS/sun for drinking only, boiling as last resort).\n\n"
        "Write Python code only. The code must:\n"
        "- import json\n"
        "- data = json.loads(combined_json)\n"
        "- selected = selected_use\n"
        "- result = { ...exact schema above... }\n"
        "- final_answer(json.dumps(result, ensure_ascii=False))\n"
    )

    with suppress_logs_and_output(), suppress_stdout_stderr():
        out = str(agent.run(prompt, additional_args={
            'combined_json': combined_json,
            'selected_use': use_case,
        }))

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

        return parsed
    except Exception:
        return parsed


