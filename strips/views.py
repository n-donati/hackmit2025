from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from .process_image import process_image

@csrf_exempt
@require_http_methods(["POST"])
def analyze_strip(request):
    try:
        data = json.loads(request.body)
        base64_image = data.get('image')
        if not base64_image:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        result = process_image(base64_image)
        return JsonResponse(result)
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=400)
    except Exception as e:
        return JsonResponse({'error': 'Internal server error'}, status=500)
