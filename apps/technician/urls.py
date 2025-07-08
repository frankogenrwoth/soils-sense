from django.urls import path
from . import views

app_name = 'technician'

urlpatterns = [
    path('dashboard/', views.dashboard, name='dashboard'),
] 