from django.conf import settings
from google import genai
from PIL import Image
from smolagents import CodeAgent, LiteLLMModel, Tool
import io
import json
import logging
import os
import sys
from contextlib import contextmanager
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
import os

logger = logging.getLogger(__name__)

def static_map_image(request):
    """
    Return a Google Static Maps image (satellite/hybrid) for given coordinates or center.

    Query params:
      - lat, lng: coordinates (preferred)
      - center: fallback text or "lat,lng"
      - zoom: default 16 (required if no markers/path/visible)
      - size: WxH (default 640x400)
      - maptype: satellite|hybrid|roadmap|terrain (default satellite)
      - scale, format, language, region, map_id: optional
      - markers, path, style, visible: repeatable parameters, forwarded verbatim
      - signature: optional, forwarded
    """
    # Prefer settings; fallback to environment; last resort: allow passing via ?key=...
    api_key = (
        getattr(settings, 'GOOGLE_MAPS_API_KEY', None)
        or os.getenv('GOOGLE_MAPS_API_KEY')
        or request.GET.get('key')
    )
    if not api_key:
        return JsonResponse({
            'error': 'Missing GOOGLE_MAPS_API_KEY. Set it in environment.'
        }, status=500)

    latitude = request.GET.get('lat')
    longitude = request.GET.get('lng')
    center_text = request.GET.get('center')

    if (latitude and longitude):
        center_value = f"{latitude},{longitude}"
    else:
        center_value = center_text

    markers_list = request.GET.getlist('markers')
    path_list = request.GET.getlist('path')
    style_list = request.GET.getlist('style')
    visible_list = request.GET.getlist('visible')

    # If there are no overlay parameters, require center
    if not center_value and not markers_list and not path_list and not visible_list:
        return JsonResponse({'error': 'Provide lat and lng, or center, or overlays (markers/path/visible).'}, status=400)

    size = request.GET.get('size', '640x400')
    zoom = request.GET.get('zoom', '16')
    maptype = request.GET.get('maptype', 'satellite')
    scale = request.GET.get('scale')
    img_format = request.GET.get('format')
    language = request.GET.get('language')
    region = request.GET.get('region')
    map_id = request.GET.get('map_id')
    signature = request.GET.get('signature')

    base_url = 'https://maps.googleapis.com/maps/api/staticmap'

    query_params = {
        'size': size,
        'maptype': maptype,
        'key': api_key,
    }

    # center/zoom are optional if markers/path/visible are provided; include when present
    if center_value:
        query_params['center'] = center_value
        if zoom is not None:
            query_params['zoom'] = zoom

    if scale is not None:
        query_params['scale'] = scale
    if img_format is not None:
        query_params['format'] = img_format
    if language is not None:
        query_params['language'] = language
    if region is not None:
        query_params['region'] = region
    if map_id is not None:
        query_params['map_id'] = map_id
    if signature is not None:
        query_params['signature'] = signature

    # Repeated parameters
    if markers_list:
        query_params['markers'] = markers_list
    if path_list:
        query_params['path'] = path_list
    if style_list:
        query_params['style'] = style_list
    if visible_list:
        query_params['visible'] = visible_list

    url = f"{base_url}?{urlencode(query_params, doseq=True)}"

    try:
        request_obj = Request(url)
        with urlopen(request_obj, timeout=10) as response:
            content_type = response.headers.get('Content-Type', 'image/png')
            image_bytes = response.read()
            return HttpResponse(image_bytes, content_type=content_type)
    except HTTPError as http_error:
        status_code = getattr(http_error, 'code', 502)
        try:
            error_body = http_error.read()
        except Exception:
            error_body = b''
        return HttpResponse(error_body or b'Upstream error from Google Static Maps API', status=status_code)
    except URLError:
        return JsonResponse({'error': 'Failed to reach Google Static Maps API'}, status=502)


@contextmanager
def suppress_logs_and_output():
    """
    Temporarily reduce logging verbosity for noisy libraries while preserving stdout/stderr
    so agents can return their final answers.
    """
    previous_levels = {}
    noisy_loggers = [
        'smolagents',
        'httpx',
        'litellm',
        'urllib3',
        'google',
    ]
    try:
        # Store and reduce levels
        for name in noisy_loggers:
            logger_obj = logging.getLogger(name)
            previous_levels[name] = logger_obj.level
            logger_obj.setLevel(logging.ERROR)
            logger_obj.propagate = False
        yield
    finally:
        # Restore levels
        for name in noisy_loggers:
            logger_obj = logging.getLogger(name)
            if name in previous_levels:
                logger_obj.setLevel(previous_levels[name])
            logger_obj.propagate = True


@contextmanager
def suppress_stdout_stderr():
    """
    Temporarily redirect stdout and stderr to devnull.
    """
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        with open(os.devnull, 'w') as devnull:
            sys.stdout, sys.stderr = devnull, devnull
            yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


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


def analyze_satellite_image(image_bytes: bytes) -> dict:
    """
    Shared analyzer for water body images. Accepts raw image bytes and returns
    a JSON-serializable dict with the final structured report.
    """
    with suppress_logs_and_output():
        image = Image.open(io.BytesIO(image_bytes))

        vision_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        vision_prompt = (
            "You are a visual expert. Given a satellite image view that contains a water body, describe in deep detail: "
            "- Surroundings (landscape, vegetation, infrastructure, shore conditions).\n"
            "- Potential pollution sources (industrial, agricultural, urban runoff, trash/debris).\n"
            "- Any visible signage, bridges, landmarks that might hint the approximate location (do NOT guess an address).\n"
            "- Water appearance (color, clarity, surface patterns), without making chemical claims.\n"
            "- Weather/lighting and river flow hints (direction, turbulence).\n"
            "Keep it objective, visual-only, and comprehensive so downstream text-only agents can reason on it."
        )
        vision_response = vision_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[vision_prompt, image]
        )
        scene_description = vision_response.text or ""

        if getattr(settings, 'GEMINI_API_KEY', None) and 'GEMINI_API_KEY' not in os.environ:
            os.environ['GEMINI_API_KEY'] = settings.GEMINI_API_KEY
        model = LiteLLMModel(model_id="gemini/gemini-2.5-flash")

        def run_agent(name: str, instruction: str, schema: str) -> str:
            agent = CodeAgent(tools=[], model=model, additional_authorized_imports=["json"])
            prompt = (
                f"You are {name}. Use ONLY the provided scene description below.\n"
                f"Scene description:\n{scene_description}\n\n"
                f"Task:\n{instruction}\n"
                "Write Python code only (no prose). The code must:\n"
                "- Construct a dict named result that matches this schema exactly: "
                f"{schema}\n"
                "- Use concise, factual phrasing; avoid speculation.\n"
                "- Import json and call final_answer(json.dumps(result, ensure_ascii=False)).\n"
            )
            with suppress_stdout_stderr():
                result = agent.run(prompt)
            return str(result)

        agent1_instruction = (
            "Focus on land, vegetation, and surroundings near the water. Detect signs of pollution sources: "
            "factories, pipes, industrial zones; agricultural runoff; trash on shores; erosion; algae blooms; dead vegetation. "
            "Report geolocation-relevant features and potential contamination sources factually and non-biased."
        )
        env_schema = "{\\n  \\\"potential_sources\\\": [string,...],\\n  \\\"notes\\\": string\\n}"
        env_context = run_agent(
            "Environmental Context Analyzer",
            agent1_instruction,
            env_schema,
        )

        agent2_instruction = (
            "Focus on the water surface. Assess clarity vs murkiness, turbidity, floating oil/foam/scum, "
            "suspended particles/sediment, and reflections/light behavior for depth/turbidity hints. "
            "Provide a surface condition report with possible contamination types without speculation."
        )
        surface_schema = "{\\n  \\\"clarity\\\": string,\\n  \\\"turbidity\\\": string,\\n  \\\"surface_contaminants\\\": [string,...],\\n  \\\"notes\\\": string\\n}"
        surface_report = run_agent(
            "Water Surface & Clarity Inspector",
            agent2_instruction,
            surface_schema,
        )

        agent3_instruction = (
            "Focus on the water color spectrum and visible signatures. Consider green shades (algae/eutrophication), "
            "brown/red tints (iron, clay, pollution), blue/clear (generally clean), and iridescence (oil/metals). "
            "Infer likely chemical/biological risks with cautious language and clear uncertainty."
        )
        color_schema = "{\\n  \\\"observed_colors\\\": [string,...],\\n  \\\"inferred_risks\\\": [string,...],\\n  \\\"notes\\\": string\\n}"
        color_chem = run_agent(
            "Color & Chemical Signature Estimator",
            agent3_instruction,
            color_schema,
        )

        final_agent = CodeAgent(tools=[FinalizeWaterReportTool()], model=model, additional_authorized_imports=["json"])
        final_prompt = (
            "You will be given variables: scene_description (str), env_data (dict), surface_data (dict), color_data (dict).\n"
            "Steps: 1) derive issues_identified, recommendations, recommended_uses from inputs;\n"
            "2) choose water_usage_classification from {safe_for_drinking, agricultural_only, recreational_only, unsafe, requires_purification};\n"
            "3) build exactly 10 usage_parameters with name,value,rationale; 4) set confidence in [0,1] and caveats;\n"
            "5) call finalize_water_report(scene_description, env_data, surface_data, color_data, issues_identified, recommendations, water_usage_classification, recommended_uses, usage_parameters, confidence, caveats) and return it with final_answer.\n"
            "Output Python code only (no prose)."
        )
        with suppress_stdout_stderr():
            final_report = str(
                final_agent.run(
                    final_prompt,
                    additional_args={
                        'scene_description': scene_description,
                        'env_data': json.loads(env_context) if isinstance(env_context, str) else env_context,
                        'surface_data': json.loads(surface_report) if isinstance(surface_report, str) else surface_report,
                        'color_data': json.loads(color_chem) if isinstance(color_chem, str) else color_chem,
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
    if final_obj is not None:
        return final_obj
    try:
        return json.loads(final_report)
    except Exception:
        return {'final_report': final_report}


