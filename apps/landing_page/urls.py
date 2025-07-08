from django.urls import path
from . import views

urlpatterns = [
    # Example:
    # path('', views.landing_page.landing_page, name='home'),
    path('', views.landing_page, name='landing_page'),
    path('about/', views.about_page, name='about_page'),
    path('contact/', views.contact_page, name='contact_page'),

]