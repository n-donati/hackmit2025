from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import logging

from .utils import analyze_water_image

logger = logging.getLogger(__name__)


@csrf_exempt
@require_http_methods(["POST"])
def analyze_water_agents(request):
    """
    Runs a 4-agent chain using smolagents on an uploaded image to analyze water.

    POST form-data:
      - photo: image file
    Returns: JSON with final structured report.
    """
    try:
        if 'photo' not in request.FILES:
            return JsonResponse({'success': False, 'error': 'No photo provided'}, status=400)

        photo_file = request.FILES['photo']
        if photo_file.size > 10 * 1024 * 1024:
            return JsonResponse({'success': False, 'error': 'Photo too large (max 10MB)'}, status=400)

        image_bytes = photo_file.read()
        final_obj = analyze_water_image(image_bytes)
        return JsonResponse(final_obj)
    except Exception as e:
        logger.error(f"Error in analyze_water_agents: {str(e)}")
        return JsonResponse({'success': False, 'error': 'Internal server error'}, status=500)
