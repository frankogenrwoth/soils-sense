from django.urls import path
from . import views

urlpatterns = [
    path('', views.LandingPage.as_view(), name='home'),
    path('contact/', views.ContactPage.as_view(), name='contact'),
    path('about/', views.AboutPage.as_view(), name='about'),
    path('privacy/', views.PrivacyPolicyPage.as_view(), name='privacy'),
    path('terms/', views.TermsOfUsePage.as_view(), name='terms'),
]