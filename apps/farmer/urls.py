from django.urls import path
from . import views

app_name = 'farmer'

urlpatterns = [
    path('dashboard/', views.dashboard, name='dashboard'),
    path('profile/', views.profile, name='profile'),
    path('farm-management/', views.farm_management, name='farm_management'),
    path('farm-management/delete-farm/<int:farm_id>/', views.delete_farm, name='delete_farm'),
    path('farm-management/delete-crop/<int:crop_id>/', views.delete_crop, name='delete_crop'),
    path('add-farm/', views.add_farm, name='add_farm'),
    path('add-crop/', views.add_crop, name='add_crop'),
    path('analytics/', views.analytics, name='analytics'),
    path('recommendations/', views.recommendations, name='recommendations'),
    path('predictions/', views.predictions, name='predictions'),
    
    # New URLs for soil moisture data management
    path('soil-data/', views.soil_data_management, name='soil_data_management'),
    path('soil-data/add/', views.add_soil_reading, name='add_soil_reading'),
    path('soil-data/upload/', views.upload_soil_data, name='upload_soil_data'),
    path('soil-data/filter/', views.filter_soil_data, name='filter_soil_data'),
    path('soil-data/delete/<int:reading_id>/', views.delete_reading, name='delete_reading'),
    path('download-csv-template/', views.download_csv_template, name='download_csv_template'),

    
] 