from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect
from django.contrib.auth import views as auth_views
from authentication.views import LoginView, SignupView

def redirect_to_farmer(request):
    """Redirect root URL to farmer dashboard"""
    return redirect('farmer:dashboard')

urlpatterns = [
    path("admin/", admin.site.urls),
    path("farmer/", redirect_to_farmer, name='home'),
    path("authentication/", include("authentication.urls")),
    path("signup/", SignupView.as_view(), name="signup"),
    path("login/", LoginView.as_view(), name="login"),
    path("farmer/", include("apps.farmer.urls")),
	path("", include("apps.landing_page.urls")),
]
