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
        "Guidance: If combined.waterbody.evaluation.water_usage_classification exists, use it to anchor the percent; "
        "else estimate from strip analytes (e.g., high sulfate or chlorine -> lower percent). Keep tone factual.\n\n"
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
        return json.loads(out)
    except Exception:
        start = out.find('{')
        end = out.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(out[start:end+1])
            except Exception:
                return {
                    'water_health_percent': "50%",
                    'current_water_use_cases': "Use with caution; treat before sensitive uses.",
                    'potential_dangers': "Possible microbial or chemical contaminants.",
                    'purify_for_selected_use': "Filter and disinfect before your selected use.",
                }
        return {
            'water_health_percent': "50%",
            'current_water_use_cases': "Use with caution; treat before sensitive uses.",
            'potential_dangers': "Possible microbial or chemical contaminants.",
            'purify_for_selected_use': "Filter and disinfect before your selected use.",
        }


