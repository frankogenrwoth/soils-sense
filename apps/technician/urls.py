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
    path('farm-locations/add/', views.add_farm, name='add_farm'),
    path('farm/<int:pk>/', views.farm_detail, name='farm_detail'),
    path('farm/<int:pk>/edit/', views.edit_farm, name='edit_farm'),
    path('farm/<int:pk>/delete/', views.delete_farm, name='delete_farm'),
    path('models/', views.models_view, name='models'),
    path('threshold/<int:pk>/edit/', views.edit_threshold, name='edit_threshold'),
    path('threshold/<int:pk>/delete/', views.delete_threshold, name='delete_threshold'),
    path('analytics/', views.analytics, name='analytics'),
    path('reports/', views.reports, name='reports'),
    path('reports/export/', views.export_reports, name='export_reports'),
    path('report/<int:pk>/edit/', views.edit_report, name='edit_report'),
    path('report/<int:pk>/delete/', views.delete_report, name='delete_report'),
    path('report/<int:pk>/download-pdf/', views.download_prediction_pdf, name='download_prediction_pdf'),
    path('profile/', views.profile, name='profile'),
    path('settings/', views.settings, name='settings'),
    path('soil-readings/', views.technician_soil_readings, name='soil_readings'),
]