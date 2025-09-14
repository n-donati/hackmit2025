# from django.http import HttpResponse, JsonResponse
# from django.conf import settings
# from urllib.parse import urlencode
# from urllib.request import urlopen, Request
# from urllib.error import HTTPError, URLError
# import os

# def static_map_image(request):
#     """
#     Return a Google Static Maps image (satellite/hybrid) for given coordinates or center.

#     Query params:
#       - lat, lng: coordinates (preferred)
#       - center: fallback text or "lat,lng"
#       - zoom: default 16 (required if no markers/path/visible)
#       - size: WxH (default 640x400)
#       - maptype: satellite|hybrid|roadmap|terrain (default satellite)
#       - scale, format, language, region, map_id: optional
#       - markers, path, style, visible: repeatable parameters, forwarded verbatim
#       - signature: optional, forwarded
#     """
#     # Prefer settings; fallback to environment; last resort: allow passing via ?key=...
#     api_key = (
#         getattr(settings, 'GOOGLE_MAPS_API_KEY', None)
#         or os.getenv('GOOGLE_MAPS_API_KEY')
#         or request.GET.get('key')
#     )
#     if not api_key:
#         return JsonResponse({
#             'error': 'Missing GOOGLE_MAPS_API_KEY. Set it in environment.'
#         }, status=500)

#     latitude = request.GET.get('lat')
#     longitude = request.GET.get('lng')
#     center_text = request.GET.get('center')

#     if (latitude and longitude):
#         center_value = f"{latitude},{longitude}"
#     else:
#         center_value = center_text

#     markers_list = request.GET.getlist('markers')
#     path_list = request.GET.getlist('path')
#     style_list = request.GET.getlist('style')
#     visible_list = request.GET.getlist('visible')

#     # If there are no overlay parameters, require center
#     if not center_value and not markers_list and not path_list and not visible_list:
#         return JsonResponse({'error': 'Provide lat and lng, or center, or overlays (markers/path/visible).'}, status=400)

#     size = request.GET.get('size', '640x400')
#     zoom = request.GET.get('zoom', '16')
#     maptype = request.GET.get('maptype', 'satellite')
#     scale = request.GET.get('scale')
#     img_format = request.GET.get('format')
#     language = request.GET.get('language')
#     region = request.GET.get('region')
#     map_id = request.GET.get('map_id')
#     signature = request.GET.get('signature')

#     base_url = 'https://maps.googleapis.com/maps/api/staticmap'

#     query_params = {
#         'size': size,
#         'maptype': maptype,
#         'key': api_key,
#     }

#     # center/zoom are optional if markers/path/visible are provided; include when present
#     if center_value:
#         query_params['center'] = center_value
#         if zoom is not None:
#             query_params['zoom'] = zoom

#     if scale is not None:
#         query_params['scale'] = scale
#     if img_format is not None:
#         query_params['format'] = img_format
#     if language is not None:
#         query_params['language'] = language
#     if region is not None:
#         query_params['region'] = region
#     if map_id is not None:
#         query_params['map_id'] = map_id
#     if signature is not None:
#         query_params['signature'] = signature

#     # Repeated parameters
#     if markers_list:
#         query_params['markers'] = markers_list
#     if path_list:
#         query_params['path'] = path_list
#     if style_list:
#         query_params['style'] = style_list
#     if visible_list:
#         query_params['visible'] = visible_list

#     url = f"{base_url}?{urlencode(query_params, doseq=True)}"

#     try:
#         request_obj = Request(url)
#         with urlopen(request_obj, timeout=10) as response:
#             content_type = response.headers.get('Content-Type', 'image/png')
#             image_bytes = response.read()
#             return HttpResponse(image_bytes, content_type=content_type)
#     except HTTPError as http_error:
#         status_code = getattr(http_error, 'code', 502)
#         try:
#             error_body = http_error.read()
#         except Exception:
#             error_body = b''
#         return HttpResponse(error_body or b'Upstream error from Google Static Maps API', status=status_code)
#     except URLError:
#         return JsonResponse({'error': 'Failed to reach Google Static Maps API'}, status=502)
