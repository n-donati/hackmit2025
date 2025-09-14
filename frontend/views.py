from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import base64
import json
import logging
import sys
import time

from strips.process_image import process_image as process_strip_base64
from strips.utils import build_strip_final
from waterbody.utils import analyze_water_image
from .finalize_utils import finalize_report


def home(request):
    return render(request, 'landing.html')


def choices(request):
    return render(request, 'choices.html')


def health(request):
    return render(request, 'health.html')


def analysis(request):
    # Reuse tester as the analysis entry point
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

        # 1) Run strip and water analyses in parallel-ish (simple threads)
        import threading

        strip_result = None
        strip_error = None
        strip_received_bytes = 0

        water_result = None
        water_error = None
        water_received_bytes = 0

        def _analyze_strip():
            nonlocal strip_result, strip_error, strip_received_bytes
            try:
                t0 = time.time()
                strip_bytes = strip_file.read()
                strip_received_bytes = len(strip_bytes) if strip_bytes else 0
                strip_b64 = base64.b64encode(strip_bytes).decode('utf-8') if strip_bytes else ''
                if not strip_b64:
                    strip_error = 'Empty strip image payload'
                else:
                    strip_result = process_strip_base64(strip_b64)
                t1 = time.time()
                agg_logger.info("[AGG] strip_analysis=%.2fs", t1 - t0)
            except Exception as exc:
                logger.exception("Strip analysis failed")
                strip_error = str(exc)

        def _analyze_water():
            nonlocal water_result, water_error, water_received_bytes
            try:
                t0 = time.time()
                water_image_bytes = water_files[0].read()
                water_received_bytes = len(water_image_bytes) if water_image_bytes else 0
                if water_image_bytes:
                    water_result = analyze_water_image(water_image_bytes)
                t1 = time.time()
                agg_logger.info("[AGG] water_analysis=%.2fs", t1 - t0)
            except Exception as exc:
                logger.exception("Waterbody analysis failed")
                water_error = str(exc)

        t1 = threading.Thread(target=_analyze_strip)
        t2 = threading.Thread(target=_analyze_water)
        t1.start(); t2.start(); t1.join(); t2.join()

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

        # Log concise summary (avoid dumping full JSON)
        try:
            num_analytes = (response.get('strip') or {}).get('num_analytes')
            agg_logger.info(
                "Aggregated summary: strip_bytes=%d water_bytes=%d analytes=%s water_present=%s lat=%s lng=%s",
                strip_received_bytes,
                water_received_bytes,
                num_analytes,
                bool(response.get('waterbody')),
                latitude,
                longitude,
            )
        except Exception:
            pass

        return JsonResponse(response)
    except Exception as e:
        logger.exception("Error aggregating analysis")
        return JsonResponse({'error': 'Internal server error'}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def aggregate_stash(request):
    try:
        # Store current payload in session without returning large JSON to client again
        body = json.loads(request.body.decode('utf-8'))
        request.session['last_analysis'] = body
        request.session.modified = True
        return JsonResponse({'ok': True})
    except Exception:
        return JsonResponse({'ok': False}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def aggregate_finalize(request):
    try:
        # Support two flows:
        # A) Multipart with files + lat/lng + use_case (single-shot)
        # B) JSON with use_case only (uses stashed session analysis)

        if request.FILES.get('strip') or request.FILES.get('waterbody'):
            agg_logger.info("[FINALIZE] Flow A (files) detected")
            # Flow A: perform analysis now, then finalize
            strip_file = request.FILES.get('strip')
            water_files = request.FILES.getlist('waterbody') or ([request.FILES.get('waterbody')] if request.FILES.get('waterbody') else [])
            lat_str = request.POST.get('lat')
            lng_str = request.POST.get('lng')
            user_use_case = request.POST.get('use_case', '')

            if not strip_file or not water_files:
                return JsonResponse({'error': 'Missing files'}, status=400)
            if lat_str is None or lng_str is None:
                return JsonResponse({'error': 'Missing lat/lng'}, status=400)
            try:
                latitude = float(lat_str)
                longitude = float(lng_str)
            except ValueError:
                return JsonResponse({'error': 'Invalid lat/lng values'}, status=400)

            try:
                agg_logger.info(
                    "[FINALIZE] Received strip_bytes=%s water_bytes=%s lat=%s lng=%s use_case=%s",
                    getattr(strip_file, 'size', None),
                    getattr(water_files[0], 'size', None) if water_files else None,
                    latitude,
                    longitude,
                    user_use_case,
                )
            except Exception:
                pass

            # Analyze strip and water in parallel
            import threading
            strip_result = {}
            water_result = None
            strip_error = None
            water_error = None

            def _s():
                nonlocal strip_result, strip_error
                try:
                    t0 = time.time()
                    strip_bytes = strip_file.read()
                    strip_b64 = base64.b64encode(strip_bytes).decode('utf-8') if strip_bytes else ''
                    if not strip_b64:
                        strip_error = 'Empty strip image payload'
                    else:
                        strip_result = process_strip_base64(strip_b64)
                    t1 = time.time()
                    agg_logger.info("[FINALIZE] strip_analysis=%.2fs", t1 - t0)
                except Exception as exc:
                    agg_logger.exception("[FINALIZE] Strip analysis failed: %s", str(exc))
                    strip_result = {}

            def _w():
                nonlocal water_result, water_error
                try:
                    t0 = time.time()
                    water_image_bytes = water_files[0].read()
                    water_result = analyze_water_image(water_image_bytes) if water_image_bytes else None
                    t1 = time.time()
                    agg_logger.info("[FINALIZE] water_analysis=%.2fs", t1 - t0)
                except Exception as exc:
                    agg_logger.exception("[FINALIZE] Waterbody analysis failed: %s", str(exc))
                    water_result = None

            t1 = threading.Thread(target=_s)
            t2 = threading.Thread(target=_w)
            t1.start(); t2.start(); t1.join(); t2.join()

            static_map_url = f"/location/aerial/?lat={latitude}&lng={longitude}&zoom=16&size=640x400&maptype=satellite"
            combined = {
                'strip': build_strip_final(strip_result or {}),
                'waterbody': water_result or None,
                'location': {
                    'lat': latitude,
                    'lng': longitude,
                    'static_map_url': static_map_url,
                },
                'errors': {
                    'strip': None if strip_result else 'strip_failed',
                    'waterbody': None if water_result else 'waterbody_failed',
                }
            }
            try:
                agg_logger.info("[FINALIZE] Combined summary: has_strip=%s has_water=%s", bool(strip_result), bool(water_result))
            except Exception:
                pass
            try:
                # Pass a minimized input to the AI to avoid duplicate/noisy strip payloads
                t0 = time.time()
                ai_input = {
                    'strip': {'values': strip_result or {}},
                    'waterbody': water_result or None,
                    'location': {'lat': latitude, 'lng': longitude},
                }
                ai_result = finalize_report(ai_input, user_use_case)
                t1 = time.time()
                agg_logger.info("[FINALIZE] finalize_agent=%.2fs", t1 - t0)
                # Attach selected use and a title hint for the UI
                use_titles = {
                    'drinking': 'Purify for Drinking Use',
                    'irrigation': 'Purify for Irrigation Use',
                    'human': 'Purify for Human Use',
                    'animals': 'Purify for Animal Use',
                }
                ai_result['selected_use'] = user_use_case
                ai_result['purify_title'] = use_titles.get(user_use_case or '', 'Purify Guidance')
                try:
                    agg_logger.info("[FINALIZE] AI result sample: percent=%s", ai_result.get('water_health_percent'))
                except Exception:
                    pass
                return JsonResponse(ai_result)
            except Exception:
                # Fallback heuristic
                water_class = None
                try:
                    water_class = (water_result or {}).get('evaluation', {}).get('water_usage_classification')
                except Exception:
                    pass
                percent_map = {
                    'safe_for_drinking': 85,
                    'recreational_only': 65,
                    'agricultural_only': 60,
                    'requires_purification': 45,
                    'unsafe': 25,
                    None: 50,
                }
                health_percent = percent_map.get(water_class, 50)
                current_use_cases = "Unsafe to drink; treat before use. Can be used for irrigation, cleaning, or animals after proper treatment." if water_class in (None, 'requires_purification', 'unsafe') else "Use with caution according to local guidelines."
                dangers = (
                    "This water may contain bacteria, viruses, heavy metals, or chemical contaminants. "
                    "Consuming it can lead to gastrointestinal illness, skin irritation, or long-term health issues."
                )
                use_case_texts = {
                    'drinking': "Purify for drinking: filter, disinfect (boil/chemical/UV), then store safely.",
                    'irrigation': "Purify for irrigation: coarse filter to remove sediment, then disinfect before use.",
                    'human': "Purify for human contact: filter, disinfect; avoid contact if skin irritation occurs.",
                    'animals': "Purify for animals: filter and disinfect; monitor animals for signs of illness.",
                }
                purify_for = use_case_texts.get(user_use_case or '', "Filter and disinfect before your selected use.")
                return JsonResponse({
                    'water_health_percent': f"{health_percent}%",
                    'current_water_use_cases': current_use_cases,
                    'potential_dangers': dangers,
                    'purify_for_selected_use': purify_for,
                    'selected_use': user_use_case,
                    'purify_title': use_case_texts.get(user_use_case or '', 'Purify Guidance').replace('Purify for ', 'Purify for ') if user_use_case else 'Purify Guidance',
                })

        # Flow B: JSON from session
        payload = json.loads(request.body.decode('utf-8'))
        user_use_case = payload.get('use_case')
        agg_logger.info("[FINALIZE] Flow B (session) detected; use_case=%s", user_use_case)
        data = request.session.get('last_analysis') or {}
        try:
            agg_logger.info("[FINALIZE] Session last_analysis present=%s", bool(data))
        except Exception:
            pass

        # Minimal, heuristic finalization (placeholder before smolagents flow):
        # Map waterbody classification to percent, uses, dangers; personalize with selected use-case
        water_class = None
        try:
            water_class = (data.get('waterbody') or {}).get('evaluation', {}).get('water_usage_classification')
        except Exception:
            water_class = None

        percent_map = {
            'safe_for_drinking': 85,
            'recreational_only': 65,
            'agricultural_only': 60,
            'requires_purification': 45,
            'unsafe': 25,
            None: 50,
        }
        health_percent = percent_map.get(water_class, 50)

        current_use_cases = "Unsafe to drink; treat before use. Can be used for irrigation, cleaning, or animals after proper treatment." if water_class in (None, 'requires_purification', 'unsafe') else "Use with caution according to local guidelines."

        dangers = (
            "This water may contain bacteria, viruses, heavy metals, or chemical contaminants. "
            "Consuming it can lead to gastrointestinal illness, skin irritation, or long-term health issues."
        )

        use_case_texts = {
            'drinking': "Purify for drinking: filter, disinfect (boil/chemical/UV), then store safely.",
            'irrigation': "Purify for irrigation: coarse filter to remove sediment, then disinfect before use.",
            'human': "Purify for human contact: filter, disinfect; avoid contact if skin irritation occurs.",
            'animals': "Purify for animals: filter and disinfect; monitor animals for signs of illness.",
        }
        purify_for = use_case_texts.get(user_use_case, "Treat the water in three simple steps: filter, disinfect, and use for your selected purpose.")

        # Use smolagents synthesizer for final JSON; fallback to heuristic result if it fails
        try:
            # Build minimized AI input: only pass strip values, basic location, and waterbody
            t0 = time.time()
            strip_values = {}
            try:
                strip_values = ((data.get('strip') or {}).get('values') or {}) if isinstance(data.get('strip'), dict) else {}
            except Exception:
                strip_values = {}
            loc = data.get('location') or {}
            ai_input = {
                'strip': {'values': strip_values},
                'waterbody': data.get('waterbody') or None,
                'location': {'lat': loc.get('lat'), 'lng': loc.get('lng')},
            }
            ai_result = finalize_report(ai_input, user_use_case or '')
            t1 = time.time()
            agg_logger.info("[FINALIZE] finalize_agent=%.2fs", t1 - t0)
            use_titles = {
                'drinking': 'Purify for Drinking Use',
                'irrigation': 'Purify for Irrigation Use',
                'human': 'Purify for Human Use',
                'animals': 'Purify for Animal Use',
            }
            ai_result['selected_use'] = user_use_case
            ai_result['purify_title'] = use_titles.get(user_use_case or '', 'Purify Guidance')
            try:
                agg_logger.info("[FINALIZE] AI result sample: percent=%s", ai_result.get('water_health_percent'))
            except Exception:
                pass
            return JsonResponse(ai_result)
        except Exception as exc:
            agg_logger.exception("[FINALIZE] AI finalize failed: %s", str(exc))
            result = {
                'water_health_percent': f"{health_percent}%",
                'current_water_use_cases': current_use_cases,
                'potential_dangers': dangers,
                'purify_for_selected_use': purify_for,
                'selected_use': user_use_case,
                'purify_title': use_case_texts.get(user_use_case or '', 'Purify Guidance').replace('Purify for ', 'Purify for ') if user_use_case else 'Purify Guidance',
            }
            return JsonResponse(result)
    except Exception:
        agg_logger.exception("[FINALIZE] Unhandled error")
        return JsonResponse({'error': 'finalization_failed'}, status=500)
