from django.urls import path
from . import views

urlpatterns = [
    path('aerial/', views.static_map_image, name='static_map_image'),
]
