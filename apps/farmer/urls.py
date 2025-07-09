from django.urls import path
from . import views

app_name = 'farmer'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('profile/', views.profile, name='profile'),
    path('manage-files/', views.manage_files, name='manage_files'),
    path('view-history/', views.view_history, name='view_history'),
    path('analytics/', views.analytics, name='analytics'),
    path('recommendations/', views.recommendations, name='recommendations'),
    path('upload-file/', views.upload_file, name='upload_file'),
    path('download-file/', views.download_file, name='download_file'),
    path('delete-file/', views.delete_file, name='delete_file'),
] 