from django.urls import path
from . import views

app_name = 'farmer'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('view-history/', views.view_history, name='view_history'),
    path('upload-file/', views.upload_file, name='upload_file'),
] 