from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('aggregate/', views.aggregate_analysis, name='aggregate_analysis'),
]


