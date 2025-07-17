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
    path('predictions/delete/<int:prediction_id>/', views.delete_prediction, name='delete_prediction'),
    path('predictions/download/<int:prediction_id>/', views.download_prediction_pdf, name='download_prediction_pdf'),
    path('predictions/download-csv/', views.download_predictions_csv, name='download_predictions_csv'),
    path('get-latest-readings/<int:farm_id>/', views.get_latest_readings, name='get_latest_readings'),
    
    # New URLs for soil moisture data management
    path('soil-data/', views.soil_data_management, name='soil_data_management'),
    path('soil-data/add/', views.add_soil_reading, name='add_soil_reading'),
    path('soil-data/upload/', views.upload_soil_data, name='upload_soil_data'),
    path('soil-data/filter/', views.filter_soil_data, name='filter_soil_data'),
    path('soil-data/delete/<int:reading_id>/', views.delete_reading, name='delete_reading'),
    path('download-csv-template/', views.download_csv_template, name='download_csv_template'),

    # Notification URLs
    path('notifications/', views.notifications, name='notifications'),
    path('notifications/mark-read/<int:notification_id>/', views.mark_notification_read, name='mark_notification_read'),
    path('notifications/unread-count/', views.get_unread_count, name='unread_count'),
    path('notifications/delete/<int:notification_id>/', views.delete_notification, name='delete_notification'),

    # API URLs
    path('api/sensor-data/<int:farm_id>/', views.get_sensor_data, name='get_sensor_data'),
    path('api/sensor-data-json/<int:farm_id>/', views.get_sensor_data_api, name='get_sensor_data_api'),
] 