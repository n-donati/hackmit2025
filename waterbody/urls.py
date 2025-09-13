from django.urls import path
from . import views

urlpatterns = [
    path('analyze/', views.analyze_water_agents, name='analyze_water_agents'),
]