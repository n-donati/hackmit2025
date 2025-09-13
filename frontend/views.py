from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import base64
import json
import logging
import sys

from strips.process_image import process_image as process_strip_base64
from strips.utils import build_strip_final
from waterbody.utils import analyze_water_image


def home(request):
    return render(request, 'tester.html', {
        'GOOGLE_MAPS_API_KEY': getattr(settings, 'GOOGLE_MAPS_API_KEY', None)
    })


logger = logging.getLogger(__name__)

# Dedicated logger to always show aggregator outputs on stdout
agg_logger = logging.getLogger('aggregate')
if not agg_logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setLevel(logging.INFO)
    _formatter = logging.Formatter('[AGGREGATE] %(levelname)s %(message)s')
    _handler.setFormatter(_formatter)
    agg_logger.addHandler(_handler)
agg_logger.setLevel(logging.INFO)
agg_logger.propagate = False


@csrf_exempt
@require_http_methods(["POST"])
def aggregate_analysis(request):
    try:
        # Files
        strip_file = request.FILES.get('strip')
        water_files = request.FILES.getlist('waterbody')

        if not strip_file:
            return JsonResponse({'error': 'Missing test strip image (field name: strip)'}, status=400)
        if not water_files:
            return JsonResponse({'error': 'Missing water body image(s) (field name: waterbody)'}, status=400)

        # Location
        lat_str = request.POST.get('lat')
        lng_str = request.POST.get('lng')
        if lat_str is None or lng_str is None:
            return JsonResponse({'error': 'Missing location lat/lng'}, status=400)
        try:
            latitude = float(lat_str)
            longitude = float(lng_str)
        except ValueError:
            return JsonResponse({'error': 'Invalid lat/lng values'}, status=400)

        # 1) Strips expects base64 JSON payload
        strip_result = None
        strip_error = None
        strip_received_bytes = 0
        try:
            strip_bytes = strip_file.read()
            strip_received_bytes = len(strip_bytes) if strip_bytes else 0
            strip_b64 = base64.b64encode(strip_bytes).decode('utf-8') if strip_bytes else ''
            if not strip_b64:
                strip_error = 'Empty strip image payload'
            else:
                strip_result = process_strip_base64(strip_b64)
        except Exception as exc:
            logger.exception("Strip analysis failed")
            strip_error = str(exc)

        # 2) Waterbody analysis: use first image for now
        water_result = None
        water_error = None
        water_received_bytes = 0
        try:
            water_image_bytes = water_files[0].read()
            water_received_bytes = len(water_image_bytes) if water_image_bytes else 0
            water_result = analyze_water_image(water_image_bytes)
        except Exception as exc:
            logger.exception("Waterbody analysis failed")
            water_error = str(exc)

        # 3) Location static map URL (proxied via our backend endpoint that injects API key)
        static_map_url = f"/location/aerial/?lat={latitude}&lng={longitude}&zoom=16&size=640x400&maptype=satellite"

        # Clean, minimal response (no overlapping fields)
        response = {
            'strip': build_strip_final(strip_result or {}),
            'waterbody': water_result or None,
            'location': {
                'lat': latitude,
                'lng': longitude,
                'static_map_url': static_map_url,
            },
            'errors': {
                'strip': strip_error,
                'waterbody': water_error,
            },
            'meta': {
                'strip_received_bytes': strip_received_bytes,
                'water_received_bytes': water_received_bytes,
            }
        }

        # Save minimal object to session
        request.session['last_analysis'] = response
        request.session.modified = True

        # Log final response to backend console
        try:
            agg_logger.info("Received strip_bytes=%d water_bytes=%d lat=%s lng=%s",
                            strip_received_bytes, water_received_bytes, latitude, longitude)
            agg_logger.info("Final aggregated response: %s", json.dumps(response, ensure_ascii=False))
        except Exception:
            pass

        return JsonResponse(response)
    except Exception as e:
        logger.exception("Error aggregating analysis")
        return JsonResponse({'error': 'Internal server error'}, status=500)
