from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('analysis/', views.analysis, name='analysis'),
    path('aggregate/', views.aggregate_analysis, name='aggregate_analysis'),
    path('choices/', views.choices, name='choices'),
    path('aggregate/stash/', views.aggregate_stash, name='aggregate_stash'),
    path('aggregate/finalize/', views.aggregate_finalize, name='aggregate_finalize'),
    path('health/', views.health, name='health'),
    path('detailed/', views.detailed, name='detailed'),
    path('plan/detailed/', views.detailed_plan, name='detailed_plan'),
]
