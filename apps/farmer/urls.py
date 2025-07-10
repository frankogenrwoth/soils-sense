from django.urls import path
from . import views

app_name = 'farmer'

urlpatterns = [
    path('dashboard/', views.dashboard, name='dashboard'),
    path('profile/', views.profile, name='profile'),
    path('farm-management/', views.farm_management, name='farm_management'),
    path('add-farm/', views.add_farm, name='add_farm'),
    path('add-crop/', views.add_crop, name='add_crop'),
    path('analytics/', views.analytics, name='analytics'),
    path('recommendations/', views.recommendations, name='recommendations'),
    
    # New URLs for soil moisture data management
    path('soil-data/', views.soil_data_management, name='soil_data_management'),
    path('soil-data/add/', views.add_soil_reading, name='add_soil_reading'),
    path('soil-data/upload/', views.upload_soil_data, name='upload_soil_data'),
    path('soil-data/filter/', views.filter_soil_data, name='filter_soil_data'),
    path('download-csv-template/', views.download_csv_template, name='download_csv_template'),
] 