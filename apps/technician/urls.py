from django.urls import path
from django.shortcuts import redirect
from . import views

app_name = 'technician'

def redirect_to_dashboard(request):
    """Redirect root technician URL to dashboard"""
    return redirect('technician:dashboard')

urlpatterns = [
    path('', redirect_to_dashboard, name='index'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('farm-locations/', views.farm_locations, name='farm_locations'),
    path('farm/<int:pk>/', views.farm_detail, name='farm_detail'),
    path('farm/<int:pk>/edit/', views.edit_farm, name='edit_farm'),
    path('sensor-config/', views.sensor_config, name='sensor_config'),
    path('analytics/', views.analytics, name='analytics'),
    path('reports/', views.reports, name='reports'),
    path('profile/', views.profile, name='profile'),
    path('settings/', views.settings, name='settings'),
    path('farm/<int:pk>/delete/', views.delete_farm, name='delete_farm'),
    path('soil-readings/', views.technician_soil_readings, name='soil_readings'),
]