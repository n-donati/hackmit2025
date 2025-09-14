from django.conf import settings
from google import genai
from PIL import Image
from smolagents import CodeAgent, LiteLLMModel, Tool
import io
import json
import logging
import os
import sys
import hashlib
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Reuse singletons to avoid re-initialization overhead on each request
_MODEL_SINGLETON = None
_VISION_CLIENT_SINGLETON = None


@contextmanager
def suppress_logs_and_output():
    """
    No-op: allow normal logging and propagation.
    """
    yield


@contextmanager
def suppress_stdout_stderr():
    """
    No-op: keep stdout and stderr visible.
    """
    yield


class FinalizeWaterReportTool(Tool):
    name = "finalize_water_report"
    description = (
        "Assembles a validated, strictly structured water report JSON. "
        "Inputs for environment_context, surface_clarity, color_chemistry, issues_identified, recommendations, "
        "recommended_uses, usage_parameters, caveats can be either JSON strings or Python dict/list objects. "
        "Enforces allowed classification and exactly 10 usage_parameters. Returns a JSON string."
    )
    inputs = {
        "scene_description": {"type": "string", "description": "Overall scene description text."},
        "environment_context": {"type": "string", "description": "JSON object with potential_sources: [str], notes: str"},
        "surface_clarity": {"type": "string", "description": "JSON object with clarity,turbidity,surface_contaminants:[str],notes"},
        "color_chemistry": {"type": "string", "description": "JSON object with observed_colors:[str], inferred_risks:[str], notes"},
        "issues_identified": {"type": "string", "description": "JSON array of strings"},
        "recommendations": {"type": "string", "description": "JSON array of strings"},
        "water_usage_classification": {"type": "string", "description": "one of safe_for_drinking|agricultural_only|recreational_only|unsafe|requires_purification"},
        "recommended_uses": {"type": "string", "description": "JSON array of strings"},
        "usage_parameters": {"type": "string", "description": "JSON array of 10 objects: {name,value,rationale}"},
        "confidence": {"type": "number", "description": "0..1"},
        "caveats": {"type": "string", "description": "JSON array of strings"},
    }
    output_type = "string"

    def forward(
        self,
        scene_description: str,
        environment_context,
        surface_clarity,
        color_chemistry,
        issues_identified,
        recommendations,
        water_usage_classification: str,
        recommended_uses,
        usage_parameters,
        confidence: float,
        caveats,
    ) -> str:
        allowed = {"safe_for_drinking", "agricultural_only", "recreational_only", "unsafe", "requires_purification"}

        def _parse_json(value, fallback):
            if isinstance(value, (dict, list)):
                return value
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except Exception:
                    return fallback
            return fallback

        env_obj = _parse_json(environment_context, {"potential_sources": [], "notes": ""})
        surf_obj = _parse_json(surface_clarity, {"clarity": "", "turbidity": "", "surface_contaminants": [], "notes": ""})
        color_obj = _parse_json(color_chemistry, {"observed_colors": [], "inferred_risks": [], "notes": ""})
        issues = _parse_json(issues_identified, [])
        recs = _parse_json(recommendations, [])
        uses = _parse_json(recommended_uses, [])
        params = _parse_json(usage_parameters, [])
        cavs = _parse_json(caveats, [])

        if water_usage_classification not in allowed:
            water_usage_classification = "requires_purification"

        if not isinstance(params, list):
            params = []
        if len(params) < 10:
            while len(params) < 10:
                params.append({"name": "param", "value": "unknown", "rationale": "insufficient data"})
        elif len(params) > 10:
            params = params[:10]

        if not isinstance(uses, list):
            uses = []

        try:
            conf = float(confidence)
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))

        final_obj = {
            "environment_context": {
                "potential_sources": env_obj.get("potential_sources", []),
                "notes": env_obj.get("notes", ""),
            },
            "surface_clarity": {
                "clarity": surf_obj.get("clarity", ""),
                "turbidity": surf_obj.get("turbidity", ""),
                "surface_contaminants": surf_obj.get("surface_contaminants", []),
                "notes": surf_obj.get("notes", ""),
            },
            "color_chemistry": {
                "observed_colors": color_obj.get("observed_colors", []),
                "inferred_risks": color_obj.get("inferred_risks", []),
                "notes": color_obj.get("notes", ""),
            },
            "evaluation": {
                "issues_identified": issues,
                "recommendations": recs,
                "water_usage_classification": water_usage_classification,
                "recommended_uses": uses,
                "usage_parameters": params,
                "confidence": conf,
                "caveats": cavs,
            },
        }

        return json.dumps(final_obj, ensure_ascii=False)


def analyze_water_image(image_bytes: bytes) -> dict:
    """
    Shared analyzer for water body images. Accepts raw image bytes and returns
    a JSON-serializable dict with the final structured report.
    """
    # Lightweight in-memory cache by image hash to avoid recomputation
    _CACHE_MAX = 16
    if not hasattr(analyze_water_image, "_cache"):
        analyze_water_image._cache = {}  # type: ignore[attr-defined]
        analyze_water_image._cache_order = []  # type: ignore[attr-defined]

    image_hash = hashlib.sha256(image_bytes).hexdigest()
    cache = analyze_water_image._cache  # type: ignore[attr-defined]
    order = analyze_water_image._cache_order  # type: ignore[attr-defined]
    if image_hash in cache:
        return cache[image_hash]

    with suppress_logs_and_output():
        image = Image.open(io.BytesIO(image_bytes))
        try:
            image = image.convert('RGB')
        except Exception:
            pass
        # Downscale very large images to reduce upload+compute cost while preserving detail
        try:
            max_dim = 1280
            if max(image.size) > max_dim:
                ratio = max_dim / float(max(image.size))
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size)
        except Exception:
            pass

        global _VISION_CLIENT_SINGLETON
        if _VISION_CLIENT_SINGLETON is None:
            _VISION_CLIENT_SINGLETON = genai.Client(api_key=settings.GEMINI_API_KEY)
        vision_client = _VISION_CLIENT_SINGLETON
        vision_prompt = (
            "Describe this waterbody image. Be concise (<=120 words). Include: surroundings; visible pollution sources; water appearance (color, clarity, surface patterns); weather/lighting; any explicit strong evidence (trash piles, discharge pipes, oil sheen, dead fish/wildlife, algal mats)."
        )
        vision_response = vision_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[vision_prompt, image],
            config={
                'temperature': 0.1,
                'top_p': 0.9,
            }
        )
        scene_description = vision_response.text or ""

        if getattr(settings, 'GEMINI_API_KEY', None) and 'GEMINI_API_KEY' not in os.environ:
            os.environ['GEMINI_API_KEY'] = settings.GEMINI_API_KEY

        global _MODEL_SINGLETON
        if _MODEL_SINGLETON is None:
            _MODEL_SINGLETON = LiteLLMModel(
                model_id="gemini/gemini-2.5-flash",
                temperature=0,
                request_timeout=25,
            )
        model = _MODEL_SINGLETON

        agent = CodeAgent(
            tools=[],
            model=model,
            additional_authorized_imports=["json"],
            max_steps=1,
        )

        final_prompt = (
            "Using the given scene_description, produce a single Python dict named result with this exact structure: \n"
            "environment_context: {potential_sources:[str], notes:str};\n"
            "surface_clarity: {clarity:str, turbidity:str, surface_contaminants:[str], notes:str};\n"
            "color_chemistry: {observed_colors:[str], inferred_risks:[str], notes:str};\n"
            "evaluation: {issues_identified:[str], recommendations:[str], water_usage_classification: one of safe_for_drinking|agricultural_only|recreational_only|unsafe|requires_purification, recommended_uses:[str], usage_parameters:[{name,value,rationale}] (exactly 10), confidence: float 0..1, caveats:[str]}.\n"
            "Rules: be realistic, avoid speculation, only flag high risk with strong visible evidence.\n"
            "Output Python code only: import json; build result; final_answer(json.dumps(result, ensure_ascii=False)).\n"
        )

        with suppress_stdout_stderr():
            final_report = str(
                agent.run(
                    final_prompt,
                    additional_args={
                        'scene_description': scene_description,
                    }
                )
            )

    def _try_parse_json(text: str):
        try:
            return json.loads(text)
        except Exception:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end+1])
                except Exception:
                    return None
            return None

    final_obj = _try_parse_json(final_report)
    if final_obj is None:
        try:
            final_obj = json.loads(final_report)
        except Exception:
            final_obj = {'final_report': final_report}

    # Update cache (simple LRU)
    cache[image_hash] = final_obj
    order.append(image_hash)
    if len(order) > _CACHE_MAX:
        oldest = order.pop(0)
        cache.pop(oldest, None)

    return final_obj


