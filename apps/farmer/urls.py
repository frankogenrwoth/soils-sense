from django.urls import path
from . import views

app_name = 'farmer'

urlpatterns = [
    path('dashboard/', views.dashboard, name='dashboard'),
    path('profile/', views.profile, name='profile'),
    path('manage-files/', views.manage_files, name='manage_files'),
    path('view-history/', views.view_history, name='view_history'),
    path('analytics/', views.analytics, name='analytics'),
    path('recommendations/', views.recommendations, name='recommendations'),
    path('upload-file/', views.upload_file, name='upload_file'),
    path('download-file/', views.download_file, name='download_file'),
    path('delete-file/', views.delete_file, name='delete_file'),
    
    # Farm Management URLs
    path('farm-management/', views.farm_management, name='farm_management'),
    path('add-farm/', views.add_farm, name='add_farm'),
    path('add-crop/', views.add_crop, name='add_crop'),
] 