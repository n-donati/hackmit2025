from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
from google import genai
from PIL import Image
import json
import logging
import io
import os

# smolagents
from smolagents import CodeAgent, LiteLLMModel, Tool

logger = logging.getLogger(__name__)
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
        # enforce exactly 10
        if len(params) < 10:
            # pad with generic placeholders
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
            # "scene_description": scene_description,
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

@csrf_exempt
@require_http_methods(["POST"])
def analyze_water_agents(request):
    """
    Runs a 4-agent chain using smolagents on an uploaded image to analyze water.

    POST form-data:
      - photo: image file
    Returns: JSON with each agent's output and a final report.
    """
    try:
        if 'photo' not in request.FILES:
            return JsonResponse({'success': False, 'error': 'No photo provided'}, status=400)

        photo_file = request.FILES['photo']
        if photo_file.size > 10 * 1024 * 1024:
            return JsonResponse({'success': False, 'error': 'Photo too large (max 10MB)'}, status=400)

        image_bytes = photo_file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # 1) Vision pre-pass with Gemini to produce a detailed scene description
        #    This avoids agents attempting direct CV; they consume this text context.
        vision_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        vision_prompt = (
            "You are a vision expert. Given an image of a river or water body, describe in deep detail: "
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

        # Initialize Gemini 2.5 Flash via LiteLLM using Google AI Studio (API key)
        # Ensure API key is available to LiteLLM to avoid ADC/Vertex AI path
        if getattr(settings, 'GEMINI_API_KEY', None) and 'GEMINI_API_KEY' not in os.environ:
            os.environ['GEMINI_API_KEY'] = settings.GEMINI_API_KEY
        model = LiteLLMModel(model_id="gemini/gemini-2.5-flash")

        # Helper to run an agent with a specific instruction
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
            result = agent.run(prompt)
            return str(result)

        # Agent 1: Environmental Context Analyzer
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

        # Agent 2: Water Surface & Clarity Inspector
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

        # Agent 3: Color & Chemical Signature Estimator
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

        # Agent 4: Water Quality Evaluator & Report Generator
        agent4_instruction = (
            "Synthesize prior findings to evaluate overall water quality and risks. Provide: "
            "1) Issues Identified; 2) Recommendations (testing/treatment/restrictions); 3) Water Usage Classification "
            "(safe_for_drinking | agricultural_only | recreational_only | unsafe | requires_purification); 4) Recommended uses; "
            "5) Exactly 10 usage parameters with name, value, and rationale appropriate to the classification."
        )
        synthesis_context = (
            "Use the following analyses as input to your synthesis.\n\n"
            f"[Environmental Context]\n{env_context}\n\n"
            f"[Surface & Clarity]\n{surface_report}\n\n"
            f"[Color & Chemical]\n{color_chem}\n\n"
        )
        # Final synthesizer via smolagents tool: assemble validated JSON without relying on free-form output
        final_agent = CodeAgent(tools=[FinalizeWaterReportTool()], model=model, additional_authorized_imports=["json"])
        final_prompt = (
            "You will be given variables: scene_description (str), env_data (dict), surface_data (dict), color_data (dict).\n"
            "Steps: 1) derive issues_identified, recommendations, recommended_uses from inputs;\n"
            "2) choose water_usage_classification from {safe_for_drinking, agricultural_only, recreational_only, unsafe, requires_purification};\n"
            "3) build exactly 10 usage_parameters with name,value,rationale; 4) set confidence in [0,1] and caveats;\n"
            "5) call finalize_water_report(scene_description, env_data, surface_data, color_data, issues_identified, recommendations, water_usage_classification, recommended_uses, usage_parameters, confidence, caveats) and return it with final_answer.\n"
            "Output Python code only (no prose)."
        )
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

        # Parse final JSON robustly
        def _try_parse_json(text: str):
            try:
                return json.loads(text)
            except Exception:
                # try to extract first {...} block
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1 and end > start:
                    try:
                        return json.loads(text[start:end+1])
                    except Exception:
                        return None
                return None

        final_obj = _try_parse_json(final_report)

        # Return only the final structured JSON
        if final_obj is not None:
            return JsonResponse(final_obj)
        # Fallback: attempt to parse once more
        try:
            return JsonResponse(json.loads(final_report))
        except Exception:
            # Last resort: wrap raw string
            return JsonResponse({'final_report': final_report})

    except Exception as e:
        logger.error(f"Error in analyze_water_agents: {str(e)}")
        return JsonResponse({'success': False, 'error': 'Internal server error'}, status=500)
